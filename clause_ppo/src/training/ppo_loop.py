"""
PPO training loop for CLAUSE-PPO Phase 2.

Trains CodeLlama-7B via trl PPOTrainer to repair wrong SQL queries
at the clause level, combining ClausePRM dense rewards with NL2SQLEnv
terminal rewards.

Public API:
  build_rewrite_prompt   — formats the clause-rewrite prompt for the actor
  build_prm_prompt       — formats the scoring prompt for ClausePRM
  compute_reward         — combines terminal + alpha * prm_score
  get_corrupted_sample   — applies corruption engine to a Spider sample
  train_ppo              — main training entry point
"""

import os
import sys

# ── Make sibling packages importable ─────────────────────────────────────────
_SRC = os.path.dirname(os.path.abspath(__file__))          # .../src/training
_CLAUSE_PPO_SRC = os.path.normpath(os.path.join(_SRC, '..'))  # .../src
if _CLAUSE_PPO_SRC not in sys.path:
    sys.path.insert(0, _CLAUSE_PPO_SRC)

from data.clause_splitter import CLAUSE_LABELS, split_into_clauses  # noqa: E402


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_rewrite_prompt(
    question: str,
    schema: str,
    wrong_sql: str,
    faulty_clause: str,
    clause_names: list[str],
) -> str:
    """
    Build the rewrite prompt for the PPO actor.

    Args:
        question:      Natural language question.
        schema:        Formatted DB schema string from schema_to_string().
        wrong_sql:     SQL string with the faulty clause.
        faulty_clause: Spider clause key of the wrong clause (e.g. 'where').
        clause_names:  All clause keys present in the SQL (from split_into_clauses).

    Returns:
        Prompt string ending with '[SQL]'; actor generates everything after that.
    """
    task_parts = []
    for name in clause_names:
        label = CLAUSE_LABELS.get(name, name.upper())
        if name == faulty_clause:
            task_parts.append(f"The {label} clause is wrong.")
        else:
            task_parts.append(f"The {label} clause is correct.")
    faulty_label = CLAUSE_LABELS.get(faulty_clause, faulty_clause.upper())
    task_parts.append(f"Rewrite the full SQL fixing only the {faulty_label} clause.")

    return (
        f"[QUESTION] {question} "
        f"[SCHEMA] {schema} "
        f"[WRONG_SQL] {wrong_sql} "
        f"[TASK] {' '.join(task_parts)} "
        f"[SQL]"
    )


def build_prm_prompt(question: str, schema: str, clause_names_up_to_faulty: list[str]) -> str:
    """
    Build the ClausePRM scoring prompt for a clause.

    Matches PRMDataset prefix_query_str format: space-joined clause labels
    up to and including the faulty clause position (e.g. 'FROM WHERE' for
    position j=1 in [from, where, select] order).

    Args:
        clause_names_up_to_faulty: Ordered clause keys up to and including
            the faulty clause (e.g. ['from', 'where'] when where is faulty).
    """
    labels = [CLAUSE_LABELS.get(n, n.upper()) for n in clause_names_up_to_faulty]
    prefix_str = ' '.join(labels)
    return f"[QUESTION] {question} [SCHEMA] {schema} [PREFIX] {prefix_str}"


# ── Reward ────────────────────────────────────────────────────────────────────

def compute_reward(terminal: float, prm_score: float, alpha: float) -> float:
    """
    Combined PPO reward.

    Args:
        terminal:  env.step() result: +1.0 (correct) or -1.0 (wrong).
        prm_score: ClausePRM score for the rewritten clause, ∈ [0, 1].
        alpha:     Weight on the PRM score (from ppo_config.yaml).

    Returns:
        terminal + alpha * prm_score
    """
    return terminal + alpha * prm_score


# ── Episode helper ────────────────────────────────────────────────────────────

import random

from data.corruption import _CORRUPT_FNS          # noqa: E402
from utils.sql_utils import reconstruct_sql        # noqa: E402


def get_corrupted_sample(
    sample: dict,
    tables_dict: dict,
) -> tuple[str, str] | None:
    """
    Apply the corruption engine to produce a wrong SQL for one Spider sample.

    Calls the per-clause corrupt_* functions directly (not generate_corruptions)
    to avoid the hardcoded spider path in the orchestration function.
    Execution verification is skipped here — env.step() handles that at training time.

    Args:
        sample:      One Spider train entry with 'sql', 'db_id' fields.
        tables_dict: tables.json loaded as {db_id: tables_entry}.

    Returns:
        (wrong_sql_str, faulty_clause_key) or None if no corruption is possible.
    """
    sql_dict = sample.get('sql')
    if not sql_dict:
        return None

    tables = tables_dict.get(sample['db_id'])
    if tables is None:
        return None

    # Get clause names present in this SQL (execution order)
    clauses = split_into_clauses(sql_dict)
    clause_names = [name for name, _ in clauses]

    # Shuffle to avoid always corrupting the same clause
    shuffled = clause_names[:]
    random.shuffle(shuffled)

    for clause_name in shuffled:
        corrupt_fn = _CORRUPT_FNS.get(clause_name)
        if corrupt_fn is None:
            continue
        corrupted_dict = corrupt_fn(sql_dict, tables)
        if corrupted_dict is None:
            continue
        try:
            wrong_sql = reconstruct_sql(corrupted_dict, tables)
            return wrong_sql, clause_name
        except Exception:
            continue

    return None


# ── Main training loop ────────────────────────────────────────────────────────

def train_ppo(config: dict, spider_dir: str, prm_ckpt: str) -> list[dict]:
    """
    Full PPO training loop for clause-level NL2SQL repair.

    Args:
        config:     Parsed ppo_config.yaml as a nested dict.
        spider_dir: Path to the Spider dataset root (contains tables.json,
                    train_spider.json, dev.json, database/).
        prm_ckpt:   Path to the saved ClausePRM checkpoint directory.

    Returns:
        List of log entry dicts written during training.
    """
    import json
    import torch
    from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
    from peft import LoraConfig, get_peft_model, TaskType, create_reference_model
    from transformers import AutoTokenizer, BitsAndBytesConfig

    # Make src/ (env, eval) importable
    _REPO_ROOT = os.path.normpath(os.path.join(_CLAUSE_PPO_SRC, '..', '..'))
    _ENV_SRC   = os.path.join(_REPO_ROOT, 'src')
    if _ENV_SRC not in sys.path:
        sys.path.insert(0, _ENV_SRC)
    from env.env import NL2SQLEnv

    from models.prm import ClausePRM

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    mcfg  = config['model']
    pcfg  = config['ppo']
    tcfg  = config['training']
    paths = config['paths']

    os.makedirs(paths['output_dir'], exist_ok=True)
    log_dir = os.path.dirname(paths['log_file'])
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(mcfg['actor_name'], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Actor with value head ─────────────────────────────────────────────────
    print("Loading actor model...")
    use_4bit = mcfg.get('quantization', '4bit') == '4bit'
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    actor = AutoModelForCausalLMWithValueHead.from_pretrained(
        mcfg['actor_name'],
        quantization_config=bnb_config,
        device_map='auto',
        torch_dtype=torch.bfloat16,
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=mcfg['lora_rank'],
        lora_alpha=mcfg['lora_alpha'],
        lora_dropout=mcfg.get('lora_dropout', 0.05),
        target_modules=mcfg.get('target_modules', ['q_proj', 'v_proj']),
        bias='none',
    )
    actor = get_peft_model(actor, lora_cfg)

    # Reference model: same 4-bit base weights, LoRA adapter disabled.
    # create_reference_model() from peft handles this — no extra VRAM copy.
    ref_model = create_reference_model(actor)

    trainable = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in actor.parameters())
    print(f"Actor trainable params: {trainable:,} / {total:,}")

    # ── ClausePRM ─────────────────────────────────────────────────────────────
    print(f"Loading ClausePRM from {prm_ckpt}...")
    prm = ClausePRM(
        model_name=prm_ckpt,
        lora_rank=mcfg['lora_rank'],
        lora_alpha=mcfg['lora_alpha'],
        lora_dropout=mcfg.get('lora_dropout', 0.05),
        target_modules=mcfg.get('target_modules', ['q_proj', 'v_proj']),
        use_4bit=use_4bit,
    )
    prm.eval()

    # ── Spider data ───────────────────────────────────────────────────────────
    print("Loading Spider data...")
    with open(os.path.join(spider_dir, 'tables.json')) as f:
        tables_dict = {t['db_id']: t for t in json.load(f)}

    with open(os.path.join(spider_dir, 'train_spider.json')) as f:
        all_train = json.load(f)

    ppo_start   = tcfg.get('ppo_split_start', 4000)
    ppo_samples = all_train[ppo_start:]
    print(f"PPO split: {len(ppo_samples)} samples (train_spider[{ppo_start}:])")

    with open(os.path.join(spider_dir, 'dev.json')) as f:
        dev_samples = json.load(f)
    print(f"Dev split: {len(dev_samples)} samples")

    # ── Environment ───────────────────────────────────────────────────────────
    env = NL2SQLEnv(spider_dir=spider_dir, tables=tables_dict)

    # ── trl PPOTrainer ────────────────────────────────────────────────────────
    ppo_config_obj = PPOConfig(
        model_name=mcfg['actor_name'],
        learning_rate=pcfg['learning_rate'],
        batch_size=pcfg['batch_size'],
        mini_batch_size=pcfg['mini_batch_size'],
        gradient_accumulation_steps=pcfg['gradient_accumulation_steps'],
        ppo_epochs=pcfg['ppo_epochs'],
        init_kl_coef=pcfg['kl_coef'],
        max_grad_norm=tcfg['max_grad_norm'],
    )
    ppo_trainer = PPOTrainer(
        config=ppo_config_obj,
        model=actor,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    # ── Episode loop ──────────────────────────────────────────────────────────
    log_entries: list[dict] = []
    num_episodes = tcfg['num_episodes']

    print(f"\nStarting PPO training: {num_episodes} episodes")

    for ep_idx in range(num_episodes):
        sample = ppo_samples[ep_idx % len(ppo_samples)]

        # Step 1: Get a corruption of this sample
        corruption = get_corrupted_sample(sample, tables_dict)
        if corruption is None:
            continue
        wrong_sql, faulty_clause = corruption

        # Step 2: Reset environment — sets up gold SQL and schema
        state = env.reset(sample)

        # Step 3: Build rewrite prompt
        clauses      = split_into_clauses(sample['sql'])
        clause_names = [name for name, _ in clauses]
        prompt = build_rewrite_prompt(
            state['question'], state['schema'],
            wrong_sql, faulty_clause, clause_names,
        )

        # Step 4: Actor generates rewritten SQL
        query_tensor = tokenizer.encode(prompt, return_tensors='pt').squeeze(0)
        response_tensors = ppo_trainer.generate(
            [query_tensor],
            max_new_tokens=pcfg['max_new_tokens'],
            temperature=pcfg['temperature'],
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        # Decode only the newly generated tokens (response_tensors includes the query)
        generated_ids = response_tensors[0][len(query_tensor):]
        rewritten_sql = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Step 5: Terminal reward from environment
        terminal, _ = env.step(rewritten_sql)

        # Step 6: Dense reward from ClausePRM
        # Build cumulative prefix up to and including faulty clause position
        try:
            faulty_idx = clause_names.index(faulty_clause)
        except ValueError:
            faulty_idx = len(clause_names) - 1
        clause_names_up_to_faulty = clause_names[:faulty_idx + 1]

        prm_prompt = build_prm_prompt(
            state['question'], state['schema'], clause_names_up_to_faulty
        )
        prm_inputs = tokenizer(
            prm_prompt,
            return_tensors='pt',
            truncation=True,
            max_length=512,
        )
        prm_inputs = {k: v.to(device) for k, v in prm_inputs.items()}
        with torch.no_grad():
            prm_score = prm(
                prm_inputs['input_ids'],
                prm_inputs['attention_mask'],
            ).item()

        # Step 7: Combined reward
        reward = compute_reward(terminal, prm_score, pcfg['alpha'])

        # Step 8: PPO update
        ppo_trainer.step(
            [query_tensor],
            [response_tensors[0]],
            [torch.tensor(reward, dtype=torch.float32)],
        )

        # ── Logging ───────────────────────────────────────────────────────────
        if (ep_idx + 1) % tcfg['log_every'] == 0:
            entry = {
                'episode':      ep_idx + 1,
                'terminal':     terminal,
                'prm_score':    round(prm_score, 4),
                'reward':       round(reward, 4),
                'faulty_clause': faulty_clause,
            }
            log_entries.append(entry)
            with open(paths['log_file'], 'a') as f:
                f.write(json.dumps(entry) + '\n')
            print(
                f"[Ep {ep_idx+1:>4}/{num_episodes}] "
                f"terminal={terminal:+.1f}  "
                f"prm={prm_score:.3f}  "
                f"reward={reward:+.3f}  "
                f"clause={faulty_clause}"
            )

        # ── Checkpoint ────────────────────────────────────────────────────────
        if (ep_idx + 1) % tcfg['eval_every'] == 0:
            ckpt_path = os.path.join(paths['output_dir'], f'ep_{ep_idx + 1}')
            ppo_trainer.model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            print(f"Checkpoint saved → {ckpt_path}")

    print(f"\nPPO training complete. {num_episodes} episodes.")
    return log_entries
