# CLAUDE.md — Database_final

NL2SQL clause-level repair via PPO.  
Given a wrong SQL query, identify the faulty clause and rewrite only that clause using reinforcement learning.

## Repo Structure

```
final/
├── CLAUDE.md
├── .claude/
│   └── docs/
|       ├── PIPELINE.md       ← full pipeline design & data flow
|       ├── INTERFACES.md     ← agreed function signatures between modules
|       └── QUESTIONS.md      ← open questions & decisions log
├── clause_ppo/           ← main training package (Henry, Phase 1+)
│   ├── configs/          ← YAML configs (prm_config.yaml)
│   ├── data/
│   │   ├── processed/    ← built datasets (corruption_dataset.json, etc.)
│   │   └── spider/       ← Spider dataset symlink (not committed)
│   ├── scripts/
│   │   ├── build_corruption_dataset.py
│   │   └── train_prm.py
│   └── src/
│       ├── data/         ← clause_splitter, corruption, dataset
│       ├── models/       ← prm.py
│       ├── training/     ← train_prm.py
│       └── utils/        ← execution.py, sql_utils.py
├── src/                  ← TBD (env, eval — Sam)
├── scripts/              ← TBD
├── tests/                ← test suite
├── requirements.txt
└── README.md
```

## Team

| Name | Role |
|------|------|
| Sam (you) | RL environment, pipeline, evaluation |
| Henry | Reward model, PPO training (trl) |
| Ian | Spider loader, clause parser, demo |

## Key Design Decisions

- **Dataset**: Spider (SQLite, no server needed)
- **Model**: CodeLlama-7B
- **RL framework**: `trl` PPO Trainer
- **Data split**: 4000 train → reward model, 3000 train → PPO, dev → eval only
- **Episode init**: Option B — use CodeLlama's actual wrong output (not corrupted gold)
- **Reward shaping**: +1/0 (correct/incorrect); partial credit under consideration (see QUESTIONS.md)
- **Baseline 1**: Vanilla CodeLlama, no RL
- **Baseline 2**: Full query regeneration (no clause-level repair)

## Environment Setup

```bash
# requires Python 3.10+
pip install -r requirements.txt

# Spider dataset — download manually
# see docs/PIPELINE.md for instructions
```

Training runs on Henry's Windows PC (WSL2, RTX 4090).  
Development and eval can run on any machine with Python 3.10+.

## Open Questions

See `docs/QUESTIONS.md`.
