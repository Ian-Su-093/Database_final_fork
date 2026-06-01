# Database Final: NL2SQL Clause-Level Repair via PPO

NL2SQL clause-level repair via PPO. Given a wrong SQL query, identify the faulty clause and rewrite only that clause using reinforcement learning.

## Team

| Name | Role |
|---|---|
| Sam | RL environment, pipeline, evaluation |
| Henry | Reward model, PPO training |
| Ian | Spider loader, clause parser, demo |

## Repo Structure

```
final/
├── clause_ppo/           ← main training package
│   ├── configs/          ← training configs (ppo_config.yaml, prm_config.yaml)
│   ├── data/
│   │   ├── processed/    ← built datasets (corruption_dataset.json, etc.)
│   │   └── spider/       ← Spider dataset (not committed)
│   ├── scripts/          ← build_corruption_dataset.py, train_prm.py, train_ppo.py
│   └── src/              ← data, models, training, utils
├── scripts/
│   └── evaluate.py       ← baseline vs Plan B vs PPO comparison table
├── src/                  ← eval pipeline
│   ├── config.py         ← shared constants + .env loader
│   ├── env/              ← NL2SQLEnv (RL environment)
│   ├── eval/             ← execution_accuracy, partial_match, best_of_n (Plan B)
│   └── baseline/         ← full-regen baseline + plan_b_inference adapter
├── tests/                ← test suite
├── .env.example          ← copy to .env, set HF_TOKEN
└── requirements.txt
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# HF token for the baseline backbone (HF Inference API)
cp .env.example .env          # then edit .env and set HF_TOKEN=hf_...
```

## Spider Dataset

```bash
gdown "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J" -O spider.zip
unzip spider.zip
mv spider_data clause_ppo/data/spider
rm spider.zip
rm -rf __MACOSX/
```

---

## Evaluation

```bash
# Local backend (no API key needed; first run downloads ~3 GB)
python scripts/evaluate.py --split dev --backend local --max-samples 20

# With Plan B (ClausePRM + Best-of-N) — needs a trained PRM checkpoint
python scripts/evaluate.py --split dev --backend local --max-samples 20 \
    --plan-b-ckpt clause_ppo/results/prm_checkpoints/best_checkpoint

# API backend — set HF_TOKEN in .env first
python scripts/evaluate.py --split dev --max-samples 20
```

Output (Accuracy@N where N = `--max-retries`, default 3):

| Method        | Accuracy@3 | Avg Token Cost |
|---------------|-----------:|---------------:|
| Full regen    |          ? |              ? |
| Plan B (PRM)  |          ? |              ? |
| Clause PPO    |          ? |              ? |

Run the test suite (no GPU or API calls required — backends are faked):

```bash
python -m pytest tests/ -v
```

