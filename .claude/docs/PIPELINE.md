# Pipeline Design

## Two Phases

```
Phase 1 — PRM Training (clause_ppo/, Henry)
  Spider train[0:4000] → corruption engine → PRMDataset → train ClausePRM
  Output: checkpoint at clause_ppo/results/prm_checkpoints/

Phase 2 — PPO Training (clause_ppo/src/training/ppo_loop.py, Henry)
  Spider train[4000:] → corruption engine → PPO episodes → train CodeLlama-7B
  Uses: NL2SQLEnv (Sam), ClausePRM checkpoint (Phase 1 output)
  Entry: clause_ppo/scripts/train_ppo.py

Evaluation (scripts/evaluate.py, Sam + Ian)
  Spider dev[all] → baseline + Plan B + RL → metrics comparison table
```

---

## Phase 2 Episode Flow (from ppo_loop.py)

Two separate feedback signals:
- **ClausePRM** (Henry): per-episode float ∈ [0,1], scores the corrupted-prefix prompt after generation
- **NL2SQLEnv** (Sam): terminal +1.0/-1.0, actual SQLite execution result

```
Spider train sample (question, sql, db_id)
        │
        ▼
get_corrupted_sample(sample, tables_dict)
  → (wrong_sql, faulty_clause)           ← corruption engine; faulty clause is KNOWN
        │
        ▼
env.reset(sample)                        ← Sam's env.py
  → state {question, schema, db_id}
        │
        ▼
build_rewrite_prompt(question, schema, wrong_sql, faulty_clause, clause_names)
  → prompt string ending with [SQL]
        │
        ▼
ppo_trainer.generate(prompt)
  → rewritten_sql                        ← CodeLlama-7B generates full fixed SQL
        │
        ├──────────────────────────────────────────────┐
        ▼                                              ▼
env.step(rewritten_sql)              build_prm_prompt → prm(inputs)
  → terminal (+1.0 / -1.0)            → prm_score ∈ [0,1]
        │                                              │
        └──────────────────┬───────────────────────────┘
                           ▼
             compute_reward(terminal, prm_score, alpha)
             = terminal + alpha * prm_score
                           │
                           ▼
             ppo_trainer.step(query, response, reward)
```

---

## Reward Formula

$$R = r_{\text{terminal}} + \alpha \cdot r_{\text{PRM}}$$

where:
- $r_{\text{terminal}} \in \{+1.0, -1.0\}$ — SQLite execution result (Sam's `env.step()`)
- $r_{\text{PRM}} \in [0, 1]$ — ClausePRM confidence score on the corrupted-prefix prompt
- $\alpha = 0.5$ — weight from `ppo_config.yaml`

---

## Baseline Pipeline (scripts/evaluate.py)

```
Spider dev sample
        │
        ▼
run_baseline(sample, generate_fn, max_retries=3, env=...)
  prompt = build_baseline_prompt(question, schema)   ← [QUESTION][SCHEMA][SQL]
  for attempt in range(max_retries):
    sql, n_in, n_out = generate_fn(prompt)            ← injected backend
    if env.step(sql) == +1: break
  → {predicted_sql, token_cost, attempts}
        │
        ▼
Sam: execution_accuracy + avg token cost
  → table: Full Regen vs Clause PPO
```

**Backend is injected.** `run_baseline` takes a `generate_fn` with contract
`prompt -> (sql_text, n_input_tokens, n_output_tokens)`, so it is decoupled
from any specific model. `make_hf_api_generate_fn()` adapts a
`huggingface_hub.InferenceClient` (chat-completions); token counts come from
the server's `completion.usage`.

**⚠ Backbone differs from the PPO actor (intentional).**
The baseline backbone is **Qwen2.5-Coder-1.5B via the HF Inference API**
(small, remote), while the PPO method is **CodeLlama-7B** trained locally.
The comparison table therefore contrasts a cheap API baseline against the
trained model — *not* two configs of the same backbone. Consequences:
- Accuracy differences mix model-size effects with method effects.
- "Avg Token Cost" is not strictly comparable across the two tokenizers.
This is a deliberate framing choice (see QUESTIONS.md), not an oversight.

---

## Evaluation Metric

**Fixed iteration (N=3 default, configurable):**

$$\text{Accuracy@N} = \frac{\text{correct predictions}}{\text{total samples}}$$

$$\text{Avg token cost} = \frac{\sum \text{(input + output tokens)}}{\text{total samples}}$$

Final comparison table:

| Method | Accuracy@3 | Avg Token Cost |
|---|---|---|
| Full regeneration (baseline) | ? | ? |
| Plan B — ClausePRM + Best-of-N | ? | ? |
| Clause PPO (ours) | ? | ? |

---

## Dataset Setup

Download Spider from https://yale-lily.github.io/spider  
Extract to `clause_ppo/data/spider/`:

```
clause_ppo/data/spider/
├── train_spider.json
├── dev.json
├── tables.json
└── database/<db_id>/<db_id>.sqlite
```

Do NOT commit (`clause_ppo/data/spider/` in `.gitignore`).

---

## Data Split

```
train_spider.json (7000 samples)
├── [0:4000]    → PRM training (Phase 1, Henry)
└── [4000:]     → PPO training (Phase 2, ppo_split_start in ppo_config.yaml)

dev.json (1034 samples) → evaluation only, never used in training
```
