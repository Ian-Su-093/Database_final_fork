# Design: PPO Training Loop for Clause-Level NL2SQL Repair

**Date:** 2026-05-24
**Branch:** `dev/ppo-training`

## Goal

Train CodeLlama-7B via PPO to repair wrong SQL queries at the clause level. Given a wrong SQL and a natural language question, the model learns to rewrite only the faulty clause to produce a correct query.

---

## Context

This is the Phase 2 training loop. Phase 1 (ClausePRM) must already be trained and a checkpoint saved at `results/prm_checkpoints/best_checkpoint` before running this.

**Inference pipeline this trains for:**
```
question + schema
    → CodeLlama baseline generates initial SQL   (Baseline 1)
    → ClausePRM scores each clause
    → get_faulty_clause() identifies weakest clause
    → PPO-trained model rewrites that clause
    → return repaired SQL                        (Ours)
```

**Baselines for evaluation:**
| System | Description |
|---|---|
| Baseline 1 | CodeLlama direct generation, no repair |
| Baseline 2 | CodeLlama generation + full query regeneration |
| Ours | CodeLlama generation + clause-level PPO repair |

---

## Files

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `clause_ppo/configs/ppo_config.yaml` | All PPO hyperparameters |
| Create | `clause_ppo/scripts/train_ppo.py` | CLI entry point |
| Create | `clause_ppo/src/training/ppo_loop.py` | Training loop logic |

No existing files are modified. Mirrors the `train_prm` file structure exactly.

---

## Data

**PPO split:** `spider/train_spider.json[4000:7000]` — 3000 samples never seen by the PRM.

**Wrong SQL source:** The corruption engine runs on-the-fly per episode, corrupting one clause of the gold SQL. This gives a guaranteed wrong SQL with a known faulty clause at every step, avoiding wasted episodes and matching the inference distribution (one or two wrong clauses in an otherwise reasonable query).

The corruption engine is already implemented in `clause_ppo/src/data/corruption.py`. It requires the Spider `tables.json` to look up column types.

---

## Episode Flow

One episode per training step:

```
sample = train_spider.json[4000:7000]   {question, gold_sql, db_id, sql}
│
├─ Corruption engine → wrong_sql        (one clause corrupted, verified ≠ gold)
│
├─ env.reset(sample) → state {question, schema, db_id}
│
├─ ClausePRM scores all clauses of wrong_sql
│   clause_scores = {c: prm.score(c, wrong_sql, context) for c in clauses}
│   faulty = env.get_faulty_clause(clause_scores)
│
├─ Build rewrite prompt (see Prompt Format)
│
├─ ppo_trainer.generate(prompt) → rewritten_sql   ← PPO trains this
│
├─ Compute reward
│   terminal, _ = env.step(rewritten_sql)          # +1.0 or -1.0
│   prm_score   = prm.score_clause(faulty, rewritten_sql, context)
│   reward      = terminal + alpha * prm_score     # alpha from config
│
└─ ppo_trainer.step([prompt], [rewritten_sql], [reward])
```

---

## Prompt Format

Consistent between training and inference. The model generates everything after `[SQL]`.

```
[QUESTION] {question}
[SCHEMA] {schema}
[WRONG_SQL] {wrong_sql}
[TASK] The {c1} clause is correct. The {c2} clause is correct.
       The {faulty} clause is wrong. Rewrite the full SQL fixing only the {faulty} clause.
[SQL]
```

- `{schema}` comes from `env.reset()` → `schema_to_string(db_id, tables)`
- Clause labels in `[TASK]` use the human-readable labels from `CLAUSE_LABELS` (e.g. `WHERE`, `GROUP BY`)
- The model outputs the full rewritten SQL (not just the clause text), which is passed directly to `env.step()`

---

## trl Integration

**Library:** `trl` PPOTrainer API (≤0.7)

**Three model instances:**

| Model | Role | VRAM |
|---|---|---|
| CodeLlama-7B 4-bit + LoRA (actor) | Generates SQL rewrites, updated by PPO | ~4 GB |
| Same weights, LoRA disabled (ref) | KL penalty baseline via `create_reference_model()` | ~0.05 GB extra |
| ClausePRM 4-bit + LoRA + head | Scores clauses for reward signal | ~4.1 GB |
| **Total** | | **~10 GB** — well within 24 GB RTX 4090 budget |

The reference model is created from the actor via PEFT's `create_reference_model()`, which disables the LoRA adapter on the shared 4-bit base weights rather than loading a full separate copy. This saves ~4 GB compared to a naive frozen copy.

**Training step:**
```python
query_tensors    = tokenizer(prompt, return_tensors='pt')
response_tensors = ppo_trainer.generate(query_tensors, **gen_kwargs)
rewritten_sql    = tokenizer.decode(response_tensors[0], skip_special_tokens=True)

terminal, _  = env.step(rewritten_sql)

# ClausePRM.forward() takes input_ids/attention_mask — build the scoring prompt,
# tokenize it, and call forward() directly. No separate score_clause wrapper exists.
prm_prompt   = build_prm_prompt(faulty, rewritten_sql, question, schema)
prm_inputs   = tokenizer(prm_prompt, return_tensors='pt', truncation=True, max_length=512)
with torch.no_grad():
    prm_score = prm(prm_inputs['input_ids'], prm_inputs['attention_mask']).item()

reward = terminal + config['ppo']['alpha'] * prm_score

ppo_trainer.step([query_tensors], [response_tensors], [torch.tensor(reward)])
```

`build_prm_prompt()` is a small helper in `ppo_loop.py` that formats the PRM input consistently with how `PRMDataset` formatted training examples (same `[SCHEMA]...[PREFIX]` template used during PRM training).

---

## Config — `ppo_config.yaml`

```yaml
model:
  actor_name: codellama/CodeLlama-7b-hf
  prm_checkpoint: results/prm_checkpoints/best_checkpoint
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: [q_proj, v_proj]
  quantization: 4bit

ppo:
  alpha: 0.5                   # weight on PRM score in combined reward
  kl_coef: 0.1                 # KL penalty coefficient
  learning_rate: 1.0e-5
  batch_size: 1                # PPO episode batch (VRAM constrained)
  mini_batch_size: 1
  gradient_accumulation_steps: 8
  ppo_epochs: 4                # inner PPO update epochs per batch
  max_new_tokens: 128          # max tokens for rewritten SQL
  temperature: 0.7

training:
  num_episodes: 3000           # one pass through PPO split
  ppo_split_start: 4000        # index into train_spider.json
  max_grad_norm: 1.0
  eval_every: 200              # episodes between evaluations on Spider dev
  log_every: 20

paths:
  spider_dir: data/spider
  output_dir: results/ppo_checkpoints
  log_file: results/ppo_training_log.jsonl
```

---

## Evaluation Metrics

Run on Spider dev set (1034 samples, never seen during training) after every `eval_every` episodes.

| Metric | Implementation |
|---|---|
| EX (Execution Accuracy) | `src/eval/metrics.py::execution_accuracy()` — already implemented |
| Partial Match | `src/eval/metrics.py::partial_match()` — already implemented |
| Tokens generated | `len(tokenizer(rewritten_sql).input_ids)` per prediction, mean over dev set |

Token efficiency is reported alongside accuracy to quantify the cost advantage of clause-level repair over full regeneration (Baseline 2).

---

## Entry Point

```bash
# Run from clause_ppo/
python scripts/train_ppo.py \
    --config      configs/ppo_config.yaml \
    --spider_dir  ../../spider \
    --prm_ckpt    results/prm_checkpoints/best_checkpoint
```

Mirrors `scripts/train_prm.py` exactly in structure: parse args → load YAML → call `ppo_loop.train_ppo(config, ...)`.

---

## Open Items (not in scope for this implementation)

- Multi-clause repair (fix more than one clause per episode)
- Iterative repair (run repair loop until `env.step()` returns +1.0 or N attempts exceeded)
- Exact value of `alpha` — to be tuned after initial training runs
