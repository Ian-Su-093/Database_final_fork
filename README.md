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
python scripts/evaluate.py --split dev --max-samples 20

# Local backend — no API, no 504s. First run downloads ~3 GB weights.
python scripts/evaluate.py --split dev --backend local --max-samples 20
```

Output (Accuracy@N where N = `--max-retries`, default 3):

| Method        | Accuracy@3 | Avg Token Cost |
|---------------|-----------:|---------------:|
| Full regen    |          ? |              ? |
| Plan B (PRM)  |          ? |              ? |
| Clause PPO    |          ? |              ? |

- **Baseline backbone:** `Qwen/Qwen2.5-Coder-1.5B-Instruct`, available via two
  backends — `--backend api` (HF Inference API via Featherless AI) or
  `--backend local` (downloads weights, runs on-device). Precision and device
  for the local backend come from [`src/config.py`](src/config.py)
  (`LOCAL_DTYPE`, `LOCAL_DEVICE`), not CLI flags.
- **Token:** read from `HF_TOKEN` in `.env` (gitignored); needed for `api`,
  optional for `local`. Never a CLI flag.
- **Clause PPO column** needs `--ppo-ckpt PATH`, but currently raises
  `NotImplementedError` until the PPO actor exposes an inference entry point
  (see `.claude/docs/QUESTIONS.md`).

Flags: `--split {dev,train}`, `--backend {api,local}`, `--max-retries N`,
`--model ID`, `--max-tokens N`, `--max-samples N`, `--output preds.json`.

Run the test suite (no GPU or API calls required — backends are faked):

```bash
python -m pytest tests/ -v
```

