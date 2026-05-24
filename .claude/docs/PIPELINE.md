# Pipeline Design

## Data Flow

Two separate feedback signals:
- **Reward model** (Henry): per-clause float ∈ [0,1], model confidence only, no SQL execution
- **Executor** (Sam): final +1/-1, based on actual SQLite execution result

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
│  1. load DB schema                     │
│  2. build prompt (question + schema)   │
│  3. return initial state               │
└───────────────────┬────────────────────┘
                    │ state
                    ▼
┌────────────────────────────────────────┐
│  Henry: PPO Actor (CodeLlama)          │
│  generates each clause one by one:     │
│    SELECT ... → reward model → r1      │
│    FROM ...   → reward model → r2      │
│    WHERE ...  → reward model → r3      │
│  faulty clause = argmin(r1, r2, r3)    │
│  CodeLlama rewrites faulty clause      │
│  reconstruct full SQL                  │
└───────────────────┬────────────────────┘
                    │ final reconstructed SQL
                    ▼
┌────────────────────────────────────────┐
│  Sam: env.step()                       │
│  1. execute full SQL on SQLite         │
│  2. compare result with gold           │
│  3. return final reward (+1 / -1)      │
└───────────────────┬────────────────────┘
                    │ final reward
                    ▼
        ┌───────────────────────┐
        │  Henry: PPO update    │
        │  uses both per-clause │
        │  scores + final reward│
        └───────────────────────┘
```

---

## Dataset Setup

Download Spider — see `spider/README.md` for instructions.

Extract to `clause_ppo/data/spider/`:
```
clause_ppo/data/spider/
├── train_spider.json
├── train_others.json
├── dev.json
├── tables.json
└── database/
    ├── concert_singer/
    │   └── concert_singer.sqlite
    └── ...
```

Do NOT commit the dataset (`clause_ppo/data/spider/` is in `.gitignore`).

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
| Vanilla CodeLlama | Direct generation, no RL | Henry |
| Full regeneration | Wrong SQL → regenerate entire query | Sam + Henry |
| Ours (Clause PPO) | Wrong SQL → rewrite faulty clause only | All |

---

## Evaluation

Run on Spider dev set (1034 samples), never touched during training.

Metrics:
- **EX** (Execution Accuracy): does predicted SQL produce same result as gold?
- **Partial Match**: per-clause F1 (TBD pending Ian's survey)

Entry point: `python scripts/evaluate.py --split dev`
