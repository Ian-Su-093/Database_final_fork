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
├── src/
│   └── not organized yet
├── scripts/
│   ├── explore_spider.py  ← script to explore Spider dataset
│   └── other scripts TBD
├── spider/         ← Spider dataset (not committed; see spider/README.md)
└── requirements.txt
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Spider Dataset
```bash
gdown "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J" -O spider.zip
unzip spider.zip
mv spider_data spider   # rename to match expected path
rm spider.zip
rm -rf __MACOSX/        # cleanup extraneous folder
```