# CLAUDE.md — Database_final

NL2SQL clause-level repair via PPO.  
Given a wrong SQL query, identify the faulty clause and rewrite only that clause using reinforcement learning.

## Repo Structure

```
Database_final/
├── CLAUDE.md
├── .claude/docs/
│   ├── PIPELINE.md       ← full pipeline design & data flow
│   ├── INTERFACES.md     ← agreed function signatures between modules
│   └── QUESTIONS.md      ← open questions & decisions log
│
├── clause_ppo/           ← ALL RL code (Henry owns everything here)
│   ├── configs/
│   │   ├── ppo_config.yaml
│   │   └── prm_config.yaml
│   ├── data/processed/   ← built datasets (not committed)
│   ├── scripts/
│   │   ├── build_corruption_dataset.py
│   │   ├── train_prm.py
│   │   ├── train_ppo.py  ← PPO training entry point
│   │   ├── clause_rewards.py
│   │   └── score_clause.py
│   └── src/
│       ├── data/         ← clause_splitter, corruption, dataset
│       ├── models/       ← prm.py, prm_inference.py
│       ├── training/     ← ppo_loop.py, train_prm.py
│       └── utils/        ← execution.py, sql_utils.py
│
├── src/                  ← Sam's code (env + eval only)
│   ├── env/
│   │   └── env.py        ← NL2SQLEnv (DONE)
│   └── eval/
│       └── metrics.py    ← execution_accuracy, partial_match (DONE)
│
├── scripts/
│   └── evaluate.py       ← Sam: baseline vs RL comparison (TODO)
│
├── tests/                ← test suite (DONE)
├── conftest.py
├── validate_env.py
└── requirements.txt
```

## Team

| Name | Role |
|------|------|
| Sam | `src/env/env.py`, `src/eval/metrics.py`, `scripts/evaluate.py` |
| Henry | Everything under `clause_ppo/` |
| Ian | Baseline inference, `scripts/evaluate.py` helper, demo |

## Key Design Decisions

- **Dataset**: Spider (SQLite, no server needed)
- **Model**: CodeLlama-7B
- **RL framework**: `trl` PPOTrainer
- **Data split**: `train_spider[4000:]` → PPO, `dev.json` → eval only
- **Episode init**: corruption engine (`get_corrupted_sample`) produces wrong SQL, NOT Qwen's actual output
- **Reward**: `terminal + alpha * prm_score` where terminal = `env.step()` result (+1/-1)
- **Baseline**: full query regeneration, `max_retries` configurable (default 3)
- **Metric**: Accuracy@N + avg token cost (input + output tokens)

## Critical Integration Point

`ppo_loop.py` calls `env.py` directly:
```python
from env.env import NL2SQLEnv
env = NL2SQLEnv(spider_dir=spider_dir, tables=tables_dict)
state = env.reset(sample)          # sample from train_spider.json
terminal, _ = env.step(rewritten_sql)
```
`env.py` signatures must not change without updating `ppo_loop.py`.

## Environment Setup

```bash
pip install -r requirements.txt
# Spider dataset → clause_ppo/data/spider/ (see PIPELINE.md)
```

Training: Henry's Windows PC (WSL2, RTX 4090).  
Eval: any machine with Python 3.10+.

## Open Questions

See `.claude/docs/QUESTIONS.md`.