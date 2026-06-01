"""
Plan B: Best-of-N clause repair with ClausePRM scoring.

This implements the complete Plan B pipeline:
1. Generate initial SQL with base model
2. Score clauses with trained ClausePRM  
3. Identify faulty clause (lowest PRM score)
4. Generate N repair candidates for faulty clause
5. Oracle selection: execute candidates, pick first correct one

This is pure inference-time repair using the trained reward model.
No PPO training involved.
"""

import os
import sys
import json
import time
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLAUSE_PPO_SRC = os.path.join(_REPO_ROOT, 'clause_ppo', 'src')
sys.path.insert(0, _CLAUSE_PPO_SRC)

from models.prm_inference import PRMScorer
from env.env import NL2SQLEnv
from eval.metrics import execution_accuracy, split_sql_prefixes
from baseline.full_regen import extract_sql, apply_chat_template


def load_generator(model_config: dict):
    """Load base model for SQL generation."""
    print(f"Loading generator: {model_config['base_name']}")
    
    if torch.cuda.is_available():
        return AutoModelForCausalLM.from_pretrained(
            model_config['base_name'],
            device_map='auto',
            dtype=torch.float16,
        )
    else:
        return AutoModelForCausalLM.from_pretrained(
            model_config['base_name'],
            dtype=torch.float32,
        )


def generate_sql(generator, tokenizer, prompt: str, max_tokens: int = 32, temperature: float = 0.0) -> str:
    """
    Generate SQL from a prompt ending with [SQL].
    """
    chat_text = apply_chat_template(tokenizer, prompt)
    inputs = tokenizer(chat_text, return_tensors='pt', truncation=True, max_length=2048)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    do_sample = temperature > 0.0
    gen_kwargs = dict(
        max_new_tokens=max_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs['temperature'] = temperature

    with torch.no_grad():
        output = generator.generate(**inputs, **gen_kwargs)

    generated_tokens = output[0][len(inputs['input_ids'][0]):]
    raw = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return extract_sql(raw)


def build_initial_prompt(question: str, schema: str) -> str:
    """Build prompt for initial SQL generation."""
    return f"[QUESTION] {question} [SCHEMA] {schema} [TASK] Generate the full SQL query. [SQL]"

def build_prm_prompt(question: str, schema: str, sql_prefix: str) -> str:
    """Build prompt for PRM scoring."""
    return f"[QUESTION] {question} [SCHEMA] {schema} [PREFIX] {sql_prefix}"


def score_clauses(scorer: PRMScorer, question: str, schema: str, sql: str) -> dict:
    """
    Score each clause prefix with PRMScorer.

    Splits the SQL into cumulative prefixes (SELECT → SELECT+FROM → …) and
    scores each with the PRM. Returns {clause_label: score}; caller uses
    argmin to identify the faulty clause. Returns {} if SQL has no recognisable
    clauses — callers fall back to 'SELECT' in that case.
    """
    scores = {}
    for label, prefix_sql in split_sql_prefixes(sql):
        try:
            scores[label] = scorer.score(question, schema, prefix_sql)
        except Exception as e:
            print(f"  score_clauses [{label}] error: {e}")
    return scores


def build_repair_prompt(question: str, schema: str, original_sql: str, faulty_clause: str, clause_scores: dict = None) -> str:
    """Build a repair prompt targeting the faulty clause. All variants end with [SQL]."""
    sql_upper = original_sql.upper().strip()

    select_part = from_part = where_part = ""
    try:
        if "SELECT" in sql_upper and "FROM" in sql_upper:
            select_start = sql_upper.find("SELECT")
            from_start   = sql_upper.find("FROM")
            select_part  = original_sql[select_start:from_start].strip()
            where_start  = sql_upper.find("WHERE")
            if where_start > from_start:
                from_part  = original_sql[from_start:where_start].strip()
                where_part = original_sql[where_start:].strip()
            else:
                from_part = original_sql[from_start:].strip()
        elif "SELECT" in sql_upper:
            select_part = original_sql.strip()
    except Exception as e:
        print(f"  repair prompt: SQL parse failed ({e}), using general fallback")

    # Surgical repair when PRM scores identify a clearly bad clause
    if clause_scores:
        worst_clause = min(clause_scores, key=clause_scores.get)
        worst_score  = clause_scores[worst_clause]
        good_clauses = {k: v for k, v in clause_scores.items() if v > 0.6}
        if good_clauses and from_part and where_part and worst_score < 0.4:
            if "SELECT" in worst_clause:
                return (f"[QUESTION] {question} [SCHEMA] {schema} "
                        f"[TASK] Fix the SELECT clause. Keep the existing table and filter: "
                        f"'{from_part} {where_part}'. Generate the corrected SQL. [SQL]")
            if "FROM" in worst_clause:
                return (f"[QUESTION] {question} [SCHEMA] {schema} "
                        f"[TASK] Fix the table references. Keep the selection: '{select_part}'. "
                        f"Generate the corrected SQL. [SQL]")
            if "WHERE" in worst_clause:
                return (f"[QUESTION] {question} [SCHEMA] {schema} "
                        f"[TASK] Fix the filtering conditions. Keep the table selection: "
                        f"'{select_part} {from_part}'. Generate the corrected SQL. [SQL]")

    # General fallbacks
    has_join = any(w.upper() in {'JOIN', 'INNER', 'LEFT', 'RIGHT'} for w in original_sql.split())
    if has_join:
        task = f"Rewrite this query without complex JOINs: '{original_sql}'."
    elif len(original_sql.strip()) < 20 or not from_part:
        task = f"Complete or fix this incomplete SQL: '{original_sql}'."
    elif "count" in original_sql.lower() and "*" not in original_sql and "(" not in original_sql:
        task = f"Fix the COUNT function in: '{original_sql}'."
    else:
        task = f"Rewrite this SQL more accurately: '{original_sql}'."

    return f"[QUESTION] {question} [SCHEMA] {schema} [TASK] {task} [SQL]"


def run_plan_b_for_evaluate(
    samples: List[Dict],
    prm_ckpt: str,
    max_retries: int = 3,  # unused — Plan B uses oracle selection
) -> Tuple[List[str], List[int], List[int]]:
    """
    evaluate.py-compatible entry point for Plan B.

    Loads the eval config, runs eval_best_of_n_direct(), and returns
    (predictions, token_costs, attempt_counts).
    """
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'clause_ppo', 'configs', 'eval_qwen_config.yaml',
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)

    spider_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'clause_ppo', 'data', 'spider',
    )

    predictions, token_costs, attempt_counts, _ = eval_best_of_n_direct(
        samples=samples,
        config=config,
        spider_dir=spider_dir,
        prm_ckpt=prm_ckpt,
    )
    return predictions, token_costs, attempt_counts


def eval_best_of_n_direct(
    samples: List[Dict],
    config: dict,
    spider_dir: str,
    prm_ckpt: str,
) -> Tuple[List[str], List[int], List[int], Dict]:
    """
    Core Plan B evaluation loop that operates on a pre-loaded sample list.

    Returns (predictions, token_costs, attempt_counts, summary).
    """
    tokenizer = AutoTokenizer.from_pretrained(config['model']['base_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prm_scorer = PRMScorer(prm_ckpt, base_model=config['model']['base_name'])
    generator  = load_generator(config['model'])

    env          = NL2SQLEnv(spider_dir=spider_dir)
    n_candidates = config['eval']['n_candidates']
    max_tokens   = config['eval']['max_new_tokens']

    predictions:    List[str] = []
    token_costs:    List[int] = []
    attempt_counts: List[int] = []
    plan_b_correct  = 0

    n_total = len(samples)
    for i, sample in enumerate(samples):
        question = sample['question']
        db_id    = sample['db_id']

        print(f"[{i+1}/{n_total}] {db_id}: {question[:60]}")

        env.reset(sample)

        table_info = env.tables.get(db_id, {})
        schema = (
            f"Database {db_id} tables: {', '.join(table_info['table_names'])}"
            if 'table_names' in table_info
            else f'[schema unavailable for {db_id}]'
        )

        # ── Initial SQL: input to PRM scoring and fallback ──────────────────
        initial_prompt = build_initial_prompt(question, schema)
        initial_sql    = generate_sql(generator, tokenizer, initial_prompt, max_tokens, temperature=0.0)
        print(f"  initial sql: {initial_sql}")

        # ── Plan B: score clauses, repair, oracle-select ────────────────────
        clause_scores = score_clauses(prm_scorer, question, schema, initial_sql)
        faulty_clause = (
            min(clause_scores, key=clause_scores.get)
            if clause_scores else 'SELECT'
        )
        print(f"  plan_b: faulty_clause={faulty_clause}  scores={clause_scores}")

        repair_prompt        = build_repair_prompt(question, schema, initial_sql, faulty_clause, clause_scores)
        repair_prompt_tokens = len(tokenizer.encode(repair_prompt))
        plan_b_total_tokens  = repair_prompt_tokens

        print(f"  plan_b: generating {n_candidates} repair candidate(s)")
        plan_b_candidates = []
        for j in range(n_candidates):
            cand = generate_sql(generator, tokenizer, repair_prompt, max_tokens, temperature=1.0)
            plan_b_candidates.append(cand)
            plan_b_total_tokens += len(tokenizer.encode(cand))
            print(f"  plan_b[{j}]: {cand}")

        best_plan_b = initial_sql
        for cand in plan_b_candidates:
            env.reset(sample)
            if env.step(cand)[0] > 0:
                best_plan_b = cand
                plan_b_correct += 1
                print(f"  plan_b: correct")
                break
        else:
            print(f"  plan_b: all candidates wrong")

        predictions.append(best_plan_b)
        token_costs.append(plan_b_total_tokens)
        attempt_counts.append(1)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    plan_b_acc = plan_b_correct / len(samples)
    avg_tokens = sum(token_costs) / len(token_costs) if token_costs else 0.0

    summary = {
        'plan_b_accuracy': plan_b_acc,
        'avg_tokens':      avg_tokens,
        'total_samples':   len(samples),
        'plan_b_correct':  plan_b_correct,
    }
    return predictions, token_costs, attempt_counts, summary


def eval_best_of_n(config: dict, spider_dir: str, prm_ckpt: str, limit: int = None):
    """Standalone Plan B evaluation pipeline."""
    tokenizer = AutoTokenizer.from_pretrained(config['model']['base_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prm_scorer = PRMScorer(prm_ckpt, base_model=config['model']['base_name'])
    generator  = load_generator(config['model'])

    processed_file = 'data/processed/original_dataset.json'
    if not os.path.exists(processed_file):
        processed_file = os.path.join(spider_dir, 'dev.json')
    print(f"Loading data from: {processed_file}")
    with open(processed_file) as f:
        samples = json.load(f)
    if limit:
        samples = samples[:limit]
    print(f"Evaluating {len(samples)} samples...")

    env          = NL2SQLEnv(spider_dir=spider_dir)
    n_candidates = config['eval']['n_candidates']
    plan_b_predictions: List[str] = []
    plan_b_tokens:      List[int] = []

    for i, sample in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] {sample.get('db_id', '?')}: {sample.get('question', '')[:60]}")
        initial_sql = ''
        try:
            state    = env.reset(sample)
            question = state['question']
            schema   = state['schema']

            initial_prompt = build_initial_prompt(question, schema)
            initial_sql    = generate_sql(generator, tokenizer, initial_prompt,
                                          config['eval']['max_new_tokens'], temperature=0.0)
            print(f"  initial: {initial_sql[:80]}")

            clause_scores = score_clauses(prm_scorer, question, schema, initial_sql)
            faulty_clause = min(clause_scores, key=clause_scores.get) if clause_scores else 'SELECT'
            print(f"  faulty_clause={faulty_clause}  scores={clause_scores}")

            repair_prompt = build_repair_prompt(question, schema, initial_sql, faulty_clause, clause_scores)
            total_tokens  = len(tokenizer.encode(repair_prompt))

            candidates = []
            for j in range(n_candidates):
                cand = generate_sql(generator, tokenizer, repair_prompt,
                                    config['eval']['max_new_tokens'], temperature=1.0)
                candidates.append(cand)
                total_tokens += len(tokenizer.encode(cand))
                print(f"  candidate[{j}]: {cand[:80]}")

            best_sql = initial_sql
            for cand in candidates:
                env.reset(sample)
                if env.step(cand)[0] > 0:
                    best_sql = cand
                    print(f"  correct")
                    break
            else:
                print(f"  all candidates wrong")

            plan_b_predictions.append(best_sql)
            plan_b_tokens.append(total_tokens)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  error: {e}")
            plan_b_predictions.append(initial_sql)
            plan_b_tokens.append(0)

    plan_b_acc = execution_accuracy(plan_b_predictions, samples, spider_dir)
    avg_tokens = sum(plan_b_tokens) / len(plan_b_tokens) if plan_b_tokens else 0.0

    print(f"\nPlan B accuracy: {plan_b_acc:.4f}")
    print(f"Avg tokens:      {avg_tokens:.1f}")

    output_file = config['paths']['output_file']
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    summary = {
        'plan_b_ex':          plan_b_acc,
        'plan_b_mean_tokens': avg_tokens,
        'n_candidates':       n_candidates,
    }
    with open(output_file, 'w') as f:
        f.write(json.dumps(summary) + '\n')
        for i, sample in enumerate(samples):
            f.write(json.dumps({
                'idx':        i,
                'question':   sample['question'],
                'db_id':      sample['db_id'],
                'plan_b_sql': plan_b_predictions[i],
                'gold_sql':   sample.get('query', ''),
            }) + '\n')

    print(f"Results saved to: {output_file}")
    return summary