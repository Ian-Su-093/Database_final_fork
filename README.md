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
│   ├── configs/          ← training configs (prm_config.yaml)
│   ├── data/
│   │   ├── processed/    ← built datasets (corruption_dataset.json, etc.)
│   │   └── spider/       ← Spider dataset (not committed)
│   ├── scripts/          ← build_corruption_dataset.py, train_prm.py
│   └── src/              ← data, models, training, utils
├── scripts/              ← entry points TBD (Sam)
├── src/                  ← env, eval TBD (Sam)
├── tests/                ← test suite
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
mv spider_data clause_ppo/data/spider
rm spider.zip
rm -rf __MACOSX/
```

---

## Dataset Documentation

### Source: Spider

[Spider](https://yale-nlp.github.io/spider/) is a large-scale cross-domain NL2SQL benchmark.

| Split | Examples | Use |
|-------|----------|-----|
| `train_spider.json` | ~7,000 | PRM training (corruption + PPO) |
| `dev.json` | ~1,034 | Evaluation only — never trained on |
| `tables.json` | 166 databases | Schema definitions |
| `database/` | 166 SQLite files | Execution oracle |

Each example contains: `question` (natural language), `query` (gold SQL), `db_id` (which database).

### Processed Datasets (built by `scripts/build_corruption_dataset.py`)

Stored in `clause_ppo/data/processed/` after running Step 1.

#### `original_dataset.json`

Filtered originals with pre-computed clause splits and prefix states. Each entry:

```json
{
  "db_id": "concert_singer",
  "question": "How many singers are there?",
  "query": "SELECT count(*) FROM singer",
  "prefix_states": [
    {
      "question": "How many singers are there?",
      "schema": "singer(singer_id, name, age, ...), concert(...)",
      "prefix_query_str": "SELECT count(*) FROM singer",
      "clause_name": "from",
      "position": 0
    }
  ]
}
```

Used as **positive examples** (label `1.0`) for PRM training.

#### `corruption_dataset.json`

Rule-based corruptions of the original queries, verified by the SQLite oracle (corrupted query must execute to a *different* result than gold). Each entry:

```json
{
  "db_id": "concert_singer",
  "question": "How many singers are there?",
  "original_query": "SELECT count(*) FROM singer",
  "corrupted_query": "SELECT count(*) FROM concert",
  "corrupted_clause": "from",
  "corrupted_position": 0,
  "strategy": "wrong_table"
}
```

Used as **negative examples** with cascade labeling:
- Prefix positions `0 .. j*-1` → label `1.0` (still correct before the fault)
- Prefix position `j*` and beyond → label `0.0` (fault introduced here)

#### `corruption_stats.json`

Per-clause pass rates for the corruption pipeline. Healthy range: `0.1–0.6`.

```json
{
  "select": {"attempts": 100, "verified": 88, "pass_rate": 0.88},
  "from":   {"attempts": 100, "verified": 96, "pass_rate": 0.96},
  ...
}
```

Near-zero means `reconstruct_sql` is broken; near-one means corruptions are trivially equivalent.

### PRM Input Format

The PRM receives one prefix state at a time, formatted as:

```
[QUESTION] {question} [SCHEMA] {schema} [PREFIX] {prefix_sql}
```

For a 4-clause query, the model is called **once per prefix** to produce per-clause scores:

| Call | Prefix passed to PRM | Score |
|------|----------------------|-------|
| 1 | `SELECT count(*)` | V_phi(s1) |
| 2 | `SELECT count(*) FROM singer` | V_phi(s2) |
| 3 | `SELECT count(*) FROM singer WHERE age > 25` | V_phi(s3) |
| 4 | `SELECT count(*) FROM singer WHERE age > 25 GROUP BY country` | V_phi(s4) |

A drop in score between consecutive steps localises the faulty clause.

### Data Split for Training

| Portion | Size | Purpose |
|---------|------|---------|
| ~4,000 train examples | corruption dataset | PRM training (Phase 1) |
| ~3,000 train examples | held out | PPO fine-tuning (Phase 2) |
| dev set | ~1,034 | Evaluation only |