# Database_final

CLAUSE-PPO: a framework for NL2SQL query repair using a Process Reward Model and a clause-level repair policy.

---

## Dataset: Spider

This project uses the [Spider](https://yale-seas.github.io/spider/) benchmark (Yu et al., EMNLP 2018) — a large-scale, cross-domain text-to-SQL dataset.

### Download

1. Download the archive from Google Drive:
   ```
   pip install gdown
   gdown "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J" -O spider.zip
   ```
2. Unzip and place in this directory:
   ```
   unzip spider.zip
   mv spider_data spider   # rename to match expected path
   rm spider.zip
   ```

Expected layout after extraction:
```
spider/
├── train_spider.json     # 7,000 training examples
├── train_others.json     # additional training examples
├── dev.json              # 1,034 dev examples
├── test.json             # test examples (no gold SQL)
├── tables.json           # 166 database schemas
├── train_gold.sql
├── dev_gold.sql
└── database/
    └── <db_id>/
        └── <db_id>.sqlite
```

### File formats

**`train_spider.json` / `dev.json`** — list of examples, each with:

| Field | Description |
|---|---|
| `db_id` | Target database identifier |
| `question` | Natural-language question |
| `query` | Gold SQL string |
| `query_toks` | Tokenized SQL |
| `sql` | Pre-parsed structured representation (see below) |

The `sql` field contains these clause keys:

| Key | Type | Notes |
|---|---|---|
| `select` | dict | Always present |
| `from` | dict | Table units + join conditions |
| `where` | list | Filter conditions |
| `groupBy` | list | Group-by columns |
| `having` | list | Aggregate conditions |
| `orderBy` | list | `["asc"/"desc", [col_refs]]` or `[]` |
| `limit` | int\|null | LIMIT value |
| `intersect/union/except` | dict\|null | Nested SQL for set operations |

**`tables.json`** — list of database schemas, each with:

| Field | Description |
|---|---|
| `db_id` | Database identifier |
| `table_names_original` | Raw table names |
| `column_names_original` | `[table_idx, col_name]` pairs |
| `column_types` | Type per column |
| `foreign_keys` | Pairs of column indices |
| `primary_keys` | Column indices |

**`database/<db_id>/<db_id>.sqlite`** — SQLite file, queryable directly:

```python
import sqlite3
conn = sqlite3.connect("spider/database/department_management/department_management.sqlite")
rows = conn.execute("SELECT name FROM head WHERE age > 56").fetchall()
```

### Dataset statistics

| Property | Value |
|---|---|
| Train examples | 7,000 |
| Dev examples | 1,034 |
| Databases | 166 |
| Avg query token length | ~18.5 |
| Queries with WHERE | 48.7% |
| Queries with GROUP BY | 24.7% |
| Queries with ORDER BY | 22.8% |
| Queries with HAVING | 5.8% |
| Complex queries (JOIN / subquery / HAVING / GRPBY+ORDBY) | 51.1% |

---

## Dataset Exploration

Run the full exploration script to regenerate all statistics and artifacts:

```bash
python3 explore_spider.py

# If Spider is in a non-default location:
python3 explore_spider.py --spider-dir /path/to/spider
```

This produces:

| Output file | Description |
|---|---|
| `results/spider_dataset_report.md` | Full documentation with statistics, edge cases, and design rationale |
| `results/clause_stats.json` | Clause presence fractions for train and dev |
| `results/complexity_stats.json` | Counts of complex query categories |
| `results/sample_clause_splits.json` | 20 examples with their clause split sequences |

### Clause splitting

For CLAUSE-PPO, each SQL is split into clauses in **execution order** (matching the database engine's data-flow):

```
FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY
```

```python
from explore_spider import split_into_clauses
import json

with open("spider/train_spider.json") as f:
    train = json.load(f)

clauses = split_into_clauses(train[0]["sql"])
# [("from", {...}), ("where", [...]), ("select", {...})]
```

### Querying the execution oracle

```python
import sqlite3, json

with open("spider/train_spider.json") as f:
    ex = json.load(f)[0]

db_path = f"spider/database/{ex['db_id']}/{ex['db_id']}.sqlite"
conn = sqlite3.connect(db_path)
rows = conn.execute(ex["query"]).fetchall()
print(rows)
```
