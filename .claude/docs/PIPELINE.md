# Pipeline Design

## Data Flow

```
Spider sample (question, gold_sql, db_id)
        │
        ▼
┌───────────────┐
│  Ian: loader  │  load_spider("train")
└───────┬───────┘
        │ sample dict
        ▼
┌────────────────────────────────────────┐
│  Sam: env.reset()                      │
│  1. call Qwen → initial SQL            │
│  2. execute initial SQL                │
│  3. if correct → skip episode          │
│  4. call Henry: score_clauses()        │
│  5. pick faulty clause (argmin score)  │
│  6. build state dict                   │
└───────────────────┬────────────────────┘
                    │ state
                    ▼
        ┌───────────────────────┐
        │  Henry: PPO Actor     │
        │  Qwen rewrites clause │
        └───────────┬───────────┘
                    │ rewritten_clause (text)
                    ▼
┌────────────────────────────────────────┐
│  Sam: env.step()                       │
│  1. splice rewritten clause into SQL   │
│  2. execute on SQLite                  │
│  3. compare result with gold           │
│  4. return reward (+1 / -1)            │
└───────────────────┬────────────────────┘
                    │ reward
                    ▼
        ┌───────────────────────┐
        │  Henry: PPO update    │
        │  trl PPOTrainer.step  │
        └───────────────────────┘
```

---

## Dataset Setup

Download Spider from https://yale-lily.github.io/spider  
Extract to `spider/`:

```
spider/
├── train_spider.json
├── train_others.json
├── dev.json
├── tables.json
└── database/
    ├── concert_singer/
    │   └── concert_singer.sqlite
    └── ...
```

Do NOT commit the dataset to the repo (add `spider/` to `.gitignore`).

---

## Data Split

```
train_spider.json (7000 samples)
├── [0:4000]   → reward model training (Henry)
└── [4000:]    → PPO training (Henry + Sam env)

dev.json (1034 samples) → evaluation only, never used in training
```

Reason: reward model must not see PPO training data, or it overfits and gives biased scores.

---

## Baselines

| Name | Description | Who implements |
|------|-------------|----------------|
| Vanilla Qwen | Direct generation, no RL | Henry |
| Full regeneration | Wrong SQL → regenerate entire query | Sam + Henry |
| Ours (Clause PPO) | Wrong SQL → rewrite faulty clause only | All |

---

## Evaluation

Run on Spider dev set (1034 samples), never touched during training.

Metrics:
- **EX** (Execution Accuracy): does predicted SQL produce same result as gold?
- **Partial Match**: per-clause F1 (TBD pending Ian's survey)

Entry point: `python scripts/evaluate.py --split dev`
