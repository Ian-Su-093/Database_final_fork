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
├── spider/               ← Spider dataset (not committed, see docs/PIPELINE.md)
├── src/
│   ├── data/             ← Ian: Spider loader, clause parser
│   ├── env/              ← Sam: RL environment (reset, step, executor)
│   ├── reward/           ← Henry: reward model (clause scorer)
│   ├── ppo/              ← Henry: PPO training loop (trl)
│   └── eval/             ← Sam: metrics, evaluation harness
├── scripts/
│   ├── train.py          ← entry point for PPO training
│   └── evaluate.py       ← entry point for evaluation on Spider dev
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
- **Model**: Qwen2.5-Coder (size TBD with Henry)
- **RL framework**: `trl` PPO Trainer
- **Data split**: 4000 train → reward model, 3000 train → PPO, dev → eval only
- **Episode init**: Option B — use Qwen's actual wrong output (not corrupted gold)
- **Baseline 1**: Vanilla Qwen, no RL
- **Baseline 2**: Full query regeneration (no clause-level repair)

## Environment Setup

```bash
# requires Python 3.10+
pip install -r requirements.txt

# Spider dataset — download manually
# see docs/PIPELINE.md for instructions
```

Training runs on Henry's Windows PC (WSL2, RTX 5090).  
Development and eval can run on any machine with Python 3.10+.

## Open Questions

See `docs/QUESTIONS.md`.
