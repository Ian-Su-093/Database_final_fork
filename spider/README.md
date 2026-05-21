# Spider Dataset

Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task (EMNLP 2018).

## Contents

| File / Folder | Description |
|---|---|
| `train_spider.json` | 7000 NL/SQL pairs across 140 databases (original Spider training set) |
| `train_others.json` | 1659 NL/SQL pairs across 6 databases (Restaurants, GeoQuery, Scholar, Academic, IMDB, Yelp) |
| `train_gold.sql` | Gold SQL for all training examples |
| `dev.json` | 1034 NL/SQL pairs across 20 databases (held-out evaluation) |
| `dev_gold.sql` | Gold SQL for dev examples |
| `test.json` | Test split (labels withheld) |
| `test_gold.sql` | Gold SQL for test examples |
| `tables.json` | Schema definitions for all 166 databases |
| `test_tables.json` | Schema definitions for test-split databases |
| `database/` | 166 SQLite database files (one subfolder per DB, each with `<db_name>.sqlite`) |
| `test_database/` | SQLite files for the test-split databases |

**Note:** the official full training set is `train_spider.json` + `train_others.json`.

## Usage in This Project

This dataset drives the NL2SQL clause-level repair pipeline (PPO).

**Data split used:**

| Split | Size | Purpose |
|---|---|---|
| `train_spider.json` (first 4000) | 4000 | Reward model training |
| `train_spider.json` (next 3000) | 3000 | PPO training |
| `dev.json` | 1034 | Evaluation only — never seen during training |

**Loading:** use `src/data/spider_loader.py` (Ian).

**Execution:** each `database/<db_name>/<db_name>.sqlite` is a self-contained SQLite file — no server required.

## Changelog (upstream)

- **2020-08-03**: corrected column name / column name original mismatch in `scholar` and `formula_1` in `tables.json`; reparsed SQL queries via `process_sql.py`.
- **2020-06-01**: corrected ~40 annotation errors/mismatches in `dev.json`.

## Citation

```bibtex
@inproceedings{yu2018spider,
  author    = {Tao Yu and Rui Zhang and Kai Yang and Michihiro Yasunaga and Dongxu Wang
               and Zifan Li and James Ma and Irene Li and Qingning Yao and Shanelle Roman
               and Zilin Zhang and Dragomir Radev},
  title     = {Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain
               Semantic Parsing and Text-to-SQL Task},
  booktitle = {EMNLP},
  year      = {2018}
}
```

`train_others.json` databases are sourced from prior work — see the original `README.txt` for their full citations.
