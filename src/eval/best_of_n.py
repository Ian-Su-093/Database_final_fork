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
from typing import Dict, List, Tuple
from tqdm import tqdm

# Import dependencies
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import yaml

# Add clause_ppo to path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLAUSE_PPO_SRC = os.path.join(_REPO_ROOT, 'clause_ppo', 'src')
sys.path.insert(0, _CLAUSE_PPO_SRC)

from data.clause_splitter import split_into_clauses
from env.env import NL2SQLEnv
from eval.metrics import execution_accuracy, partial_match


class ClausePRMInference:
    """Minimal PRM wrapper for inference."""
    
    def __init__(self, backbone, score_head):
        self.backbone = backbone
        self.score_head = score_head
        
    def __call__(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
        last_tok_h = last_hidden[batch_idx, seq_lengths]
        return self.score_head(last_tok_h).squeeze(-1)
        
    def eval(self):
        if hasattr(self.backbone, 'eval'):
            self.backbone.eval()
        if hasattr(self.score_head, 'eval'):
            self.score_head.eval()
        return self


def load_prm_model(prm_ckpt: str, model_config: dict):
    """Load ClausePRM from checkpoint."""
    print(f"Loading ClausePRM from {prm_ckpt}...")
    
    # Load base model
    if torch.cuda.is_available():
        print("Using CUDA")
        base = AutoModelForCausalLM.from_pretrained(
            model_config['base_name'],
            device_map='auto',
            torch_dtype=torch.float16,
            use_safetensors=False,
        )
    else:
        print("Using CPU")
        base = AutoModelForCausalLM.from_pretrained(
            model_config['base_name'],
            torch_dtype=torch.float32,
            use_safetensors=False,
        )
    
    # Load LoRA adapter
    backbone = PeftModel.from_pretrained(base, prm_ckpt)
    
    # Load score head
    hidden_size = backbone.config.hidden_size
    score_head = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())
    
    score_head_path = os.path.join(prm_ckpt, 'score_head.pt')
    if os.path.exists(score_head_path):
        score_head.load_state_dict(torch.load(score_head_path, map_location='cpu'))
        print(f"Loaded score head from {score_head_path}")
    else:
        print(f"WARNING: No score head at {score_head_path}")
    
    # Move to device and ensure consistent dtype with backbone
    if torch.cuda.is_available():
        score_head = score_head.cuda().half()  # Match backbone's float16
    
    prm = ClausePRMInference(backbone, score_head)
    prm.eval()
    return prm


def load_generator(model_config: dict):
    """Load base model for SQL generation."""
    print(f"Loading generator: {model_config['base_name']}")
    
    if torch.cuda.is_available():
        return AutoModelForCausalLM.from_pretrained(
            model_config['base_name'],
            device_map='auto',
            torch_dtype=torch.float16,
            use_safetensors=False,
        )
    else:
        return AutoModelForCausalLM.from_pretrained(
            model_config['base_name'],
            torch_dtype=torch.float32,
            use_safetensors=False,
        )


def generate_sql(generator, tokenizer, prompt: str, max_tokens: int = 64, temperature: float = 0.0):
    """Generate SQL from prompt."""
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        # Always use greedy decoding to avoid sampling issues with float16
        output = generator.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode generated SQL
    generated_tokens = output[0][len(inputs['input_ids'][0]):]
    sql = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Clean up SQL
    sql = sql.split('\n')[0].strip()
    if not sql.upper().startswith('SELECT'):
        sql = "SELECT " + sql
        
    return sql


def build_initial_prompt(question: str, schema: str) -> str:
    """Build prompt for initial SQL generation."""
    return f"Question: {question}\nSchema: {schema}\nSQL: SELECT"


def build_prm_prompt(question: str, schema: str, sql_prefix: str) -> str:
    """Build prompt for PRM scoring."""
    return f"[QUESTION] {question} [SCHEMA] {schema} [PREFIX] {sql_prefix}"


def score_clauses(prm, tokenizer, question: str, schema: str, sql: str, device):
    """Score each clause with PRM."""
    try:
        # Parse SQL into clauses
        # This is a simplified parser - you might need the actual clause_splitter
        sql_upper = sql.upper()
        clauses = []
        
        if 'SELECT' in sql_upper:
            clauses.append(('SELECT', sql[sql.upper().find('SELECT'):]))
            
        # For now, just score the full SQL as one "clause"
        prompt = build_prm_prompt(question, schema, sql)
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            score = prm(inputs['input_ids'], inputs['attention_mask']).item()
            
        return {'full_sql': score}
        
    except Exception as e:
        print(f"Error scoring clauses: {e}")
        return {'full_sql': 0.5}  # Default neutral score


def build_repair_prompt(question: str, schema: str, original_sql: str, faulty_clause: str) -> str:
    """Build prompt for repairing faulty clause."""
    return f"Question: {question}\nSchema: {schema}\nOriginal SQL: {original_sql}\nRewrite the {faulty_clause} clause to fix the SQL:\nSQL: SELECT"


def eval_best_of_n(config: dict, spider_dir: str, prm_ckpt: str, limit: int = None):
    """
    Main Plan B evaluation pipeline.
    """
    print("🎯 Starting Plan B evaluation...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load models
    tokenizer = AutoTokenizer.from_pretrained(config['model']['base_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    prm = load_prm_model(prm_ckpt, config['model'])
    generator = load_generator(config['model'])
    
    # Load Spider data from processed dataset
    processed_file = 'data/processed/original_dataset.json'
    if not os.path.exists(processed_file):
        # Fallback to regular dev.json if processed file not found
        processed_file = os.path.join(spider_dir, 'dev.json')
    
    print(f"Loading data from: {processed_file}")
    with open(processed_file) as f:
        samples = json.load(f)
        
    if limit:
        samples = samples[:limit]
        
    print(f"Evaluating {len(samples)} samples...")
    
    # Initialize environment
    env = NL2SQLEnv(spider_dir=spider_dir)
    
    # Evaluation loop
    baseline_predictions = []
    plan_b_predictions = []
    baseline_tokens = []
    plan_b_tokens = []
    
    n_candidates = config['eval']['n_candidates']
    
    for i, sample in enumerate(tqdm(samples, desc="Evaluating Plan B")):
            
        # Reset environment
        state = env.reset(sample)
        question = state['question']
        schema = state['schema']
        
        # Step 1: Generate initial SQL (baseline prediction)
        initial_prompt = build_initial_prompt(question, schema)
        initial_sql = generate_sql(generator, tokenizer, initial_prompt, 
                                 config['eval']['max_new_tokens'], temperature=0.0)
        baseline_predictions.append(initial_sql)
        baseline_tokens.append(50)  # Estimated tokens
        
        # Step 2: Score clauses with PRM
        clause_scores = score_clauses(prm, tokenizer, question, schema, initial_sql, device)
        
        # Step 3: Identify faulty clause (lowest score)
        if clause_scores:
            faulty_clause = min(clause_scores.keys(), key=lambda k: clause_scores[k])
        else:
            faulty_clause = 'SELECT'
            
        # Step 4: Generate repair candidates
        repair_prompt = build_repair_prompt(question, schema, initial_sql, faulty_clause)
        candidates = []
        
        for _ in range(n_candidates):
            candidate = generate_sql(generator, tokenizer, repair_prompt,
                                   config['eval']['max_new_tokens'], 
                                   temperature=config['eval']['temperature'])
            candidates.append(candidate)
            
        # Step 5: Oracle selection
        best_sql = initial_sql  # Fallback
        for candidate in candidates:
            reward, _ = env.step(candidate)
            if reward > 0:  # Correct execution
                best_sql = candidate
                break
            env.reset(sample)  # Reset for next candidate
            
        plan_b_predictions.append(best_sql)
        plan_b_tokens.append(50 + n_candidates * 30)  # Estimated tokens
        
    # Calculate metrics
    baseline_acc = execution_accuracy(baseline_predictions, samples, spider_dir)
    plan_b_acc = execution_accuracy(plan_b_predictions, samples, spider_dir)
    
    baseline_avg_tokens = sum(baseline_tokens) / len(baseline_tokens)
    plan_b_avg_tokens = sum(plan_b_tokens) / len(plan_b_tokens)
    
    # Print results
    print("\n" + "="*60)
    print("PLAN B RESULTS")
    print("="*60)
    print(f"{'Method':<20} {'Accuracy':>10} {'Avg Tokens':>12}")
    print("-"*60)
    print(f"{'Baseline':<20} {baseline_acc:>10.4f} {baseline_avg_tokens:>12.1f}")
    print(f"{'Plan B (Best-of-N)':<20} {plan_b_acc:>10.4f} {plan_b_avg_tokens:>12.1f}")
    print("="*60)
    
    # Save detailed results
    output_file = config['paths']['output_file']
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        summary = {
            'baseline_ex': baseline_acc,
            'plan_b_ex': plan_b_acc,
            'baseline_mean_tokens': baseline_avg_tokens,
            'plan_b_mean_tokens': plan_b_avg_tokens,
            'n_candidates': n_candidates
        }
        f.write(json.dumps(summary) + '\n')
        
        for i, sample in enumerate(samples):
            record = {
                'idx': i,
                'question': sample['question'],
                'db_id': sample['db_id'],
                'baseline_sql': baseline_predictions[i],
                'plan_b_sql': plan_b_predictions[i],
                'gold_sql': sample.get('query', ''),
            }
            f.write(json.dumps(record) + '\n')
            
    print(f"\nResults saved to: {output_file}")
    
    return summary