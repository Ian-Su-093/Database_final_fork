"""
explore_spider.py
=================
Comprehensive exploration of the Spider NL2SQL dataset for CLAUSE-PPO.

Steps:
  1  Verify directory structure
  2  Explore train.json / dev.json
  3  Explore tables.json
  4  Test SQLite execution oracle
  5  Clause splitting logic (split_into_clauses)
  6  Identify complex queries relevant to CLAUSE-PPO
  7  Write full markdown documentation report
  8  Save JSON artifacts to results/

Usage:
  python explore_spider.py [--spider-dir PATH]

Default spider dir: ./spider
"""

import argparse
import collections
import json
import os
import sqlite3
import sys
import textwrap

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SPIDER_DIR = os.path.join(SCRIPT_DIR, "../spider")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


# ---------------------------------------------------------------------------
# Step 1 – Verify directory structure
# ---------------------------------------------------------------------------

EXPECTED_FILES = [
    "train_spider.json",
    "dev.json",
    "tables.json",
]
EXPECTED_DIRS = [
    "database",
]


def step1_verify_structure(spider_dir: str) -> bool:
    print("\n" + "=" * 60)
    print("STEP 1 — Verify Spider directory structure")
    print("=" * 60)

    if not os.path.isdir(spider_dir):
        print(f"\n[ERROR] Spider directory not found: {spider_dir}")
        print("\nTo download Spider:")
        print("  1. Visit: https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view")
        print("  2. Download and unzip the archive.")
        print(f"  3. Place the unzipped 'spider' folder at: {spider_dir}")
        print("     OR pass --spider-dir /path/to/spider\n")
        return False

    all_ok = True
    for fname in EXPECTED_FILES:
        fpath = os.path.join(spider_dir, fname)
        if os.path.isfile(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  [OK] {fname:30s}  ({size_kb:,.1f} KB)")
        else:
            print(f"  [MISSING] {fname}")
            all_ok = False

    db_dir = os.path.join(spider_dir, "database")
    if os.path.isdir(db_dir):
        db_count = sum(
            1 for d in os.scandir(db_dir) if d.is_dir()
        )
        print(f"  [OK] database/                   ({db_count} subdirectories)")
    else:
        print("  [MISSING] database/")
        all_ok = False

    if all_ok:
        print("\nAll expected files present. Proceeding.")
    else:
        print("\nSome files are missing. Exiting.")
    return all_ok


# ---------------------------------------------------------------------------
# Step 2 – Explore train.json / dev.json
# ---------------------------------------------------------------------------

MAIN_CLAUSES = ["select", "from", "where", "groupBy", "having", "orderBy"]


def is_clause_present(sql_dict: dict, clause: str) -> bool:
    """Return True if the clause has meaningful (non-empty) content."""
    val = sql_dict.get(clause)
    if val is None:
        return False
    if isinstance(val, list):
        return len(val) > 0
    if isinstance(val, dict):
        return bool(val)
    # select is always a dict {"aggregates": [...], "columns": [...]}
    return bool(val)


def clause_count(sql_dict: dict) -> int:
    return sum(1 for c in MAIN_CLAUSES if is_clause_present(sql_dict, c))


def step2_explore_split(data: list, split_name: str) -> dict:
    print(f"\n{'=' * 60}")
    print(f"STEP 2 — Explore {split_name}")
    print("=" * 60)

    n = len(data)
    dbs = {ex["db_id"] for ex in data}
    print(f"\n  Total examples       : {n:,}")
    print(f"  Unique databases     : {len(dbs):,}")

    # Clause presence
    clause_counts = {c: 0 for c in MAIN_CLAUSES}
    clause_combo_hist = collections.Counter()
    token_lengths = []

    for ex in data:
        sql = ex.get("sql", {}) or {}
        present = []
        for c in MAIN_CLAUSES:
            if is_clause_present(sql, c):
                clause_counts[c] += 1
                present.append(c)
        clause_combo_hist[len(present)] += 1
        token_lengths.append(len(ex.get("query_toks", [])))

    print("\n  Clause presence (fraction of queries):")
    print(f"  {'Clause':<12}  {'Count':>7}  {'Fraction':>9}")
    print(f"  {'-'*12}  {'-'*7}  {'-'*9}")
    for c in MAIN_CLAUSES:
        frac = clause_counts[c] / n
        print(f"  {c:<12}  {clause_counts[c]:>7,}  {frac:>9.3f}")

    print("\n  Distribution of clauses per query:")
    for k in sorted(clause_combo_hist):
        bar = "#" * int(clause_combo_hist[k] / n * 50)
        print(f"  {k} clauses: {clause_combo_hist[k]:>5,}  {bar}")

    avg_toks = sum(token_lengths) / n
    print(f"\n  Avg query token length: {avg_toks:.1f}")

    # Example: all 6 clauses
    all6 = [ex for ex in data if clause_count(ex.get("sql", {}) or {}) == 6]
    only2 = [ex for ex in data if clause_count(ex.get("sql", {}) or {}) == 2]

    print("\n  Example with ALL 6 main clauses:")
    if all6:
        ex = all6[0]
        print(f"    Q : {ex['question']}")
        print(f"    SQL: {ex['query']}")
    else:
        print("    (none found)")

    print("\n  Example with ONLY 2 main clauses:")
    if only2:
        ex = only2[0]
        print(f"    Q : {ex['question']}")
        print(f"    SQL: {ex['query']}")
    else:
        print("    (none found)")

    stats = {
        "split": split_name,
        "total": n,
        "unique_dbs": len(dbs),
        "clause_presence": {c: {"count": clause_counts[c], "fraction": round(clause_counts[c] / n, 4)} for c in MAIN_CLAUSES},
        "clause_count_histogram": {str(k): v for k, v in sorted(clause_combo_hist.items())},
        "avg_query_token_length": round(avg_toks, 2),
    }
    return stats


# ---------------------------------------------------------------------------
# Step 3 – Explore tables.json
# ---------------------------------------------------------------------------

def step3_explore_tables(tables: list) -> dict:
    print(f"\n{'=' * 60}")
    print("STEP 3 — Explore tables.json")
    print("=" * 60)

    n = len(tables)
    print(f"\n  Total databases: {n:,}")

    tables_per_db = []
    cols_per_db = []
    fks_per_db = []

    for db in tables:
        # table_names_original includes a "*" wildcard entry in some versions
        tnames = [t for t in db.get("table_names_original", []) if t != "*"]
        tables_per_db.append(len(tnames))

        # column_names_original: each is [table_idx, col_name]; skip wildcard (*) col
        cols = [c for c in db.get("column_names_original", []) if c[1] != "*"]
        cols_per_db.append(len(cols))

        fks_per_db.append(len(db.get("foreign_keys", [])))

    def _stats(lst):
        return {"min": min(lst), "max": max(lst), "mean": round(sum(lst) / len(lst), 2)}

    t_stats = _stats(tables_per_db)
    c_stats = _stats(cols_per_db)
    f_stats = _stats(fks_per_db)

    print(f"\n  Tables per DB   — min={t_stats['min']}  max={t_stats['max']}  mean={t_stats['mean']}")
    print(f"  Columns per DB  — min={c_stats['min']}  max={c_stats['max']}  mean={c_stats['mean']}")
    print(f"  FKs per DB      — min={f_stats['min']}  max={f_stats['max']}  mean={f_stats['mean']}")

    # Pretty-print 3 example schemas
    print("\n  Sample schemas (3 databases):")
    for db in tables[:3]:
        db_id = db["db_id"]
        tnames = db.get("table_names_original", [])
        col_names = db.get("column_names_original", [])
        col_types = db.get("column_types", [])

        # Group columns by table
        table_cols = collections.defaultdict(list)
        for (tidx, cname), ctype in zip(col_names, col_types):
            if tidx >= 0 and cname != "*":  # skip wildcard
                table_cols[tidx].append(f"{cname} ({ctype})")

        print(f"\n  DB: {db_id}")
        for i, tname in enumerate(tnames):
            cols_str = ", ".join(table_cols.get(i, []))
            print(f"    {tname}: {cols_str}")

    return {
        "total_databases": n,
        "tables_per_db": t_stats,
        "columns_per_db": c_stats,
        "foreign_keys_per_db": f_stats,
    }


# ---------------------------------------------------------------------------
# Step 4 – Test execution oracle
# ---------------------------------------------------------------------------

def step4_test_oracle(train_data: list, spider_dir: str):
    print(f"\n{'=' * 60}")
    print("STEP 4 — Test SQLite execution oracle (5 examples)")
    print("=" * 60)

    tested = 0
    idx = 0
    while tested < 5 and idx < len(train_data):
        ex = train_data[idx]
        idx += 1
        db_id = ex["db_id"]
        query = ex["query"]
        db_path = os.path.join(spider_dir, "database", db_id, f"{db_id}.sqlite")

        if not os.path.isfile(db_path):
            continue  # skip if db file absent

        print(f"\n  [{tested + 1}] DB: {db_id}")
        print(f"      Q  : {ex['question']}")
        print(f"      SQL: {query}")
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(query)
            rows = cur.fetchmany(3)
            conn.close()
            if rows:
                col_names = [desc[0] for desc in cur.description]
                print(f"      Cols: {col_names}")
                for r in rows:
                    print(f"      Row : {list(r)}")
            else:
                print("      Result: (empty)")
        except sqlite3.Error as e:
            print(f"      [ORACLE ERROR] {e}")

        tested += 1


# ---------------------------------------------------------------------------
# Step 5 – Clause splitting logic
# ---------------------------------------------------------------------------

# SQL logical execution order for CLAUSE-PPO prefix scoring
CLAUSE_ORDER = ["from", "where", "groupBy", "having", "select", "orderBy"]


def split_into_clauses(sql_dict: dict) -> list:
    """
    Takes the parsed 'sql' field from a Spider entry and returns an ordered
    list of (clause_name, clause_content) tuples in SQL execution order:
        FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY

    Only non-empty clauses are included. This order reflects how the database
    engine processes the query: it filters rows first (FROM/WHERE), aggregates
    (GROUP BY / HAVING), projects (SELECT), then sorts (ORDER BY).

    This ordering is critical for CLAUSE-PPO: the Process Reward Model scores
    prefixes in execution order, so the argmin naturally points to the first
    clause where a base model went wrong.
    """
    result = []
    for clause in CLAUSE_ORDER:
        val = sql_dict.get(clause)
        if val is None:
            continue
        if isinstance(val, list) and len(val) == 0:
            continue
        if isinstance(val, dict) and not val:
            continue
        result.append((clause, val))
    return result


def step5_clause_splits(train_data: list, n: int = 10) -> list:
    print(f"\n{'=' * 60}")
    print(f"STEP 5 — Clause splitting logic (first {n} train examples)")
    print("=" * 60)

    samples = []
    for i, ex in enumerate(train_data[:n]):
        sql = ex.get("sql", {}) or {}
        clauses = split_into_clauses(sql)
        clause_names = [c[0] for c in clauses]
        print(f"\n  [{i}] {ex['question'][:80]}")
        print(f"      SQL   : {ex['query'][:100]}")
        print(f"      Clauses: {clause_names}")
        samples.append({
            "db_id": ex["db_id"],
            "question": ex["question"],
            "query": ex["query"],
            "clause_sequence": clause_names,
        })
    return samples


# ---------------------------------------------------------------------------
# Step 6 – Identify complex queries
# ---------------------------------------------------------------------------

def has_joins(sql_dict: dict) -> bool:
    """More than one table_unit in the FROM clause."""
    from_clause = sql_dict.get("from", {}) or {}
    table_units = from_clause.get("table_units", [])
    return len(table_units) > 1


def has_subquery(sql_dict: dict) -> bool:
    """Any condition in WHERE whose value is itself a SQL dict (nested query)."""
    where = sql_dict.get("where", []) or []
    for item in where:
        if isinstance(item, list) and len(item) >= 3:
            # condition format: [not_flag, op, val_unit, val1, val2]
            # val1/val2 can be a nested sql dict
            for v in item[3:]:
                if isinstance(v, dict) and "select" in v:
                    return True
    return False


def has_having(sql_dict: dict) -> bool:
    having = sql_dict.get("having", []) or []
    return len(having) > 0


def has_groupby_and_orderby(sql_dict: dict) -> bool:
    gb = sql_dict.get("groupBy", []) or []
    ob = sql_dict.get("orderBy", []) or []
    # orderBy is ["asc"/"desc", [...]] or empty list
    ob_nonempty = isinstance(ob, list) and len(ob) == 2 and len(ob[1]) > 0
    return len(gb) > 0 and ob_nonempty


def step6_complexity(train_data: list) -> dict:
    print(f"\n{'=' * 60}")
    print("STEP 6 — Complex queries relevant to CLAUSE-PPO")
    print("=" * 60)

    categories = {
        "joins": 0,
        "subqueries": 0,
        "having": 0,
        "groupby_and_orderby": 0,
    }
    at_least_one = 0

    for ex in train_data:
        sql = ex.get("sql", {}) or {}
        flags = {
            "joins": has_joins(sql),
            "subqueries": has_subquery(sql),
            "having": has_having(sql),
            "groupby_and_orderby": has_groupby_and_orderby(sql),
        }
        if any(flags.values()):
            at_least_one += 1
        for k, v in flags.items():
            if v:
                categories[k] += 1

    total = len(train_data)
    print(f"\n  Total train examples: {total:,}")
    print(f"\n  {'Category':<25}  {'Count':>6}  {'Fraction':>9}")
    print(f"  {'-'*25}  {'-'*6}  {'-'*9}")
    for cat, cnt in categories.items():
        print(f"  {cat:<25}  {cnt:>6,}  {cnt/total:>9.3f}")
    print(f"\n  At least one category: {at_least_one:,}  ({at_least_one/total:.3f})")

    return {
        **{k: {"count": v, "fraction": round(v / total, 4)} for k, v in categories.items()},
        "at_least_one": {"count": at_least_one, "fraction": round(at_least_one / total, 4)},
        "total_train": total,
    }


# ---------------------------------------------------------------------------
# Step 7 – Write markdown documentation report
# ---------------------------------------------------------------------------

def step7_write_report(
    train_stats: dict,
    dev_stats: dict,
    table_stats: dict,
    complexity_stats: dict,
    sample_clauses: list,
    out_path: str,
):
    print(f"\n{'=' * 60}")
    print("STEP 7 — Writing markdown documentation report")
    print("=" * 60)

    def frac_table(split_stats):
        lines = []
        lines.append(f"| Clause    | Count  | Fraction |")
        lines.append(f"|-----------|-------:|:--------:|")
        for clause, info in split_stats["clause_presence"].items():
            lines.append(f"| {clause:<9} | {info['count']:>6,} | {info['fraction']:.3f}    |")
        return "\n".join(lines)

    def hist_table(split_stats):
        lines = ["| # Clauses | Count  |", "|-----------|-------:|"]
        for k, v in split_stats["clause_count_histogram"].items():
            lines.append(f"| {k:<9} | {v:>6,} |")
        return "\n".join(lines)

    report = ""
    report += textwrap.dedent(f"""\
# Spider Dataset — Exploration Report
*Generated by explore_spider.py for the CLAUSE-PPO project.*

---

## 1. Dataset Overview

Spider (Yu et al., EMNLP 2018) is a large-scale, cross-domain text-to-SQL benchmark.

| Property              | Value                          |
|-----------------------|-------------------------------|
| Train examples        | {train_stats['total']:,}       |
| Dev examples          | {dev_stats['total']:,}         |
| Train unique DBs      | {train_stats['unique_dbs']:,}  |
| Dev unique DBs        | {dev_stats['unique_dbs']:,}    |
| Total databases       | {table_stats['total_databases']:,} |
| Task                  | Cross-domain NL → SQL          |

Spider covers 138+ domains ranging from academic databases to airline
reservation systems. Queries span 6 SQL clause types and include JOINs,
aggregations, subqueries, and set operations.

---

## 2. File Format Specification

### 2.1 train.json / dev.json

Each entry is a JSON object with the following fields:

| Field           | Type          | Description                                   |
|-----------------|---------------|-----------------------------------------------|
| `db_id`         | string        | Identifier of the target SQLite database       |
| `question`      | string        | Natural-language question                      |
| `question_toks` | list[str]     | Tokenised question                             |
| `query`         | string        | Gold SQL string                                |
| `query_toks`    | list[str]     | Tokenised SQL                                  |
| `query_toks_no_value` | list[str] | SQL tokens with literals replaced by placeholders |
| `sql`           | dict          | Pre-parsed structured representation (see below) |

#### `sql` sub-fields

| Field      | Type          | Non-empty means…                              |
|------------|---------------|-----------------------------------------------|
| `select`   | dict          | Always present — columns + aggregation funcs  |
| `from`     | dict          | Table units + join conditions                 |
| `where`    | list          | Conditions (each is a 5-element list)         |
| `groupBy`  | list          | Column references                             |
| `having`   | list          | Aggregate conditions (same format as where)   |
| `orderBy`  | list          | `["asc"/"desc", [col_refs]]`                  |
| `limit`    | int or null   | LIMIT value                                   |
| `intersect`| dict or null  | Nested SQL for INTERSECT                      |
| `union`    | dict or null  | Nested SQL for UNION                          |
| `except`   | dict or null  | Nested SQL for EXCEPT                         |

### 2.2 tables.json

Each entry describes one database schema:

| Field                   | Type           | Description                      |
|-------------------------|----------------|----------------------------------|
| `db_id`                 | string         | Database identifier               |
| `table_names_original`  | list[str]      | Raw table names                  |
| `column_names_original` | list[[int,str]]| `[table_idx, col_name]` pairs    |
| `column_types`          | list[str]      | Type for each column             |
| `foreign_keys`          | list[[int,int]]| Pairs of column indices          |
| `primary_keys`          | list[int]      | Column indices                   |

### 2.3 database/

One subdirectory per `db_id`, each containing a `<db_id>.sqlite` file
that can be queried directly with Python's `sqlite3` module.

---

## 3. Clause Distribution Statistics

### 3.1 Training set (n = {train_stats['total']:,})

Average query token length: **{train_stats['avg_query_token_length']}**

#### Clause presence

{frac_table(train_stats)}

#### Clause count histogram

{hist_table(train_stats)}

### 3.2 Dev set (n = {dev_stats['total']:,})

Average query token length: **{dev_stats['avg_query_token_length']}**

#### Clause presence

{frac_table(dev_stats)}

---

## 4. Key Observations for CLAUSE-PPO

### 4.1 Rarest clauses → highest repair value
- **HAVING** appears in only {train_stats['clause_presence']['having']['fraction']:.1%} of train queries.
    Base NL2SQL models fail here most often, making it a prime repair target.
- **GROUP BY** appears in {train_stats['clause_presence']['groupBy']['fraction']:.1%} of queries.
    Errors here propagate to downstream HAVING/ORDER BY clauses.
- **ORDER BY** appears in {train_stats['clause_presence']['orderBy']['fraction']:.1%} of queries.

### 4.2 Complexity profile (training set)

| Category              | Count  | Fraction |
|-----------------------|-------:|:--------:|
| JOINs (multi-table)   | {complexity_stats['joins']['count']:>6,} | {complexity_stats['joins']['fraction']:.3f}    |
| Subqueries            | {complexity_stats['subqueries']['count']:>6,} | {complexity_stats['subqueries']['fraction']:.3f}    |
| HAVING clause         | {complexity_stats['having']['count']:>6,} | {complexity_stats['having']['fraction']:.3f}    |
| GROUP BY + ORDER BY   | {complexity_stats['groupby_and_orderby']['count']:>6,} | {complexity_stats['groupby_and_orderby']['fraction']:.3f}    |
| **At least one above**| **{complexity_stats['at_least_one']['count']:>5,}** | **{complexity_stats['at_least_one']['fraction']:.3f}**   |

**{complexity_stats['at_least_one']['fraction']:.1%} of training examples** are complex enough to
be useful CLAUSE-PPO training candidates (queries a weak base model is likely to
generate incorrectly, enabling reward signal differentiation).

### 4.3 Schema complexity
- Databases range from {table_stats['tables_per_db']['min']} to {table_stats['tables_per_db']['max']} tables (mean {table_stats['tables_per_db']['mean']}).
- Queries routinely JOIN multiple tables, so errors in the FROM clause are
    especially damaging — they invalidate all downstream clause reasoning.

### 4.4 Execution order vs. syntactic order
Spider's parsed `sql` field separates clauses structurally. CLAUSE-PPO
scores prefixes in **execution order** (FROM→WHERE→GROUP BY→HAVING→SELECT→ORDER BY),
which matches the database engine's data-flow rather than SQL's written order.
This means a faulty WHERE clause can be detected before SELECT is even scored.

---

## 5. Clause Splitting Function

```python
CLAUSE_ORDER = ["from", "where", "groupBy", "having", "select", "orderBy"]

def split_into_clauses(sql_dict: dict) -> list:
    \"\"\"
    Returns [(clause_name, clause_content), ...] in execution order,
    omitting empty / null clauses.
    \"\"\"
    result = []
    for clause in CLAUSE_ORDER:
        val = sql_dict.get(clause)
        if val is None:
            continue
        if isinstance(val, list) and len(val) == 0:
            continue
        if isinstance(val, dict) and not val:
            continue
        result.append((clause, val))
    return result
```

**Design rationale:**
- Execution order (not syntactic order) lets the PRM score from the
    most-upstream clause inward; the argmin identifies the *first* wrong clause.
- `select` is always present but listed last among projection/aggregation
    because it logically applies after filtering/grouping.
- `limit`, `intersect`, `union`, `except` are deliberately excluded from the
    main 6-clause split and treated as query-level metadata (see edge cases).

---

## 6. Edge Cases and Gotchas

| Issue | Detail |
|-------|--------|
| INTERSECT / UNION / EXCEPT | `sql['intersect']` etc. are nested sql dicts. CLAUSE-PPO should treat the outer query and each set-operation branch as separate repair units. |
| LIMIT | Stored as an integer (e.g. `1`) or `null`. Not part of the 6-clause split but may matter for correctness. |
| Wildcard column | `column_names_original` always includes `(-1, "*")`; skip when computing column counts. |
| Empty WHERE list | Stored as `[]`, not `null`. The emptiness check `len(val) == 0` is required. |
| ORDER BY format | `["asc", [[agg_id, col_ref]]]` or `[]` — the inner list can itself be empty for an `orderBy` key that exists. |
| Implicit JOIN | Some `from.table_units` entries are `["sql", nested_sql]` rather than `["table_unit", table_ref]` — a subquery in FROM. |
| Logical connectors | WHERE / HAVING lists interleave conditions with `"and"` / `"or"` string tokens. Strip these when iterating conditions. |
| No train_others.json in all distributions | Some Spider releases include `train_others.json`; this report covers only `train.json`. |

---

## 7. Sample Clause Splits (first 10 train examples)

    """)

    for s in sample_clauses:
        report += f"**Q:** {s['question']}\n\n"
        report += f"**SQL:** `{s['query']}`\n\n"
        report += f"**Clauses:** {' → '.join(s['clause_sequence'])}\n\n---\n\n"

    report += textwrap.dedent("""\
## 8. Recommended Next Steps for CLAUSE-PPO

1. **Base model predictions**: Run a pretrained NL2SQL model (e.g. PICARD,
    RASAT, or T5-based) over the train/dev sets and collect failed predictions.
    Only failed queries become CLAUSE-PPO training examples.

2. **Clause-level reward**: Use the Spider execution oracle (sqlite3) to
    compute execution accuracy per clause prefix, providing the PRM training
    signal.

3. **Tokenizer alignment**: Ensure the repair policy tokenizer can faithfully
    reconstruct individual clauses from the `sql` dict so that `split_into_clauses`
    maps cleanly to model token spans.

4. **Set-operation handling**: Decide whether to repair INTERSECT/UNION/EXCEPT
    queries holistically or to split them into independent repair sub-problems.

5. **LIMIT / subquery coverage**: Consider whether the repair policy should
    handle LIMIT values and subquery rewrites, or filter these out of training
    to keep the action space manageable.
    """)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Wrote {out_path}")


# ---------------------------------------------------------------------------
# Step 8 – Save JSON artifacts
# ---------------------------------------------------------------------------

def step8_save_artifacts(
    clause_stats: dict,
    complexity_stats: dict,
    sample_clauses: list,
    results_dir: str,
):
    print(f"\n{'=' * 60}")
    print("STEP 8 — Save JSON artifacts to results/")
    print("=" * 60)

    os.makedirs(results_dir, exist_ok=True)

    paths = {
        "clause_stats.json": clause_stats,
        "complexity_stats.json": complexity_stats,
        "sample_clause_splits.json": sample_clauses,
    }
    for fname, obj in paths.items():
        fpath = os.path.join(results_dir, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        print(f"  Wrote {fpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Explore the Spider NL2SQL dataset")
    parser.add_argument("--spider-dir", default=DEFAULT_SPIDER_DIR, help="Path to the unzipped Spider directory")
    args = parser.parse_args()

    spider_dir = args.spider_dir

    # Step 1
    ok = step1_verify_structure(spider_dir)
    if not ok:
        sys.exit(1)

    # Load JSON files
    print("\nLoading JSON files...")
    with open(os.path.join(spider_dir, "train_spider.json"), encoding="utf-8") as f:
        train_data = json.load(f)
    with open(os.path.join(spider_dir, "dev.json"), encoding="utf-8") as f:
        dev_data = json.load(f)
    with open(os.path.join(spider_dir, "tables.json"), encoding="utf-8") as f:
        tables_data = json.load(f)
    print(f"  train.json: {len(train_data):,} entries")
    print(f"  dev.json  : {len(dev_data):,} entries")
    print(f"  tables.json: {len(tables_data):,} entries")

    # Step 2
    train_stats = step2_explore_split(train_data, "train.json")
    dev_stats = step2_explore_split(dev_data, "dev.json")

    # Step 3
    table_stats = step3_explore_tables(tables_data)

    # Step 4
    step4_test_oracle(train_data, spider_dir)

    # Step 5
    sample_clauses_10 = step5_clause_splits(train_data, n=10)

    # Extend to 20 for artifact
    sample_clauses_20 = []
    for ex in train_data[:20]:
        sql = ex.get("sql", {}) or {}
        clauses = split_into_clauses(sql)
        sample_clauses_20.append({
            "db_id": ex["db_id"],
            "question": ex["question"],
            "query": ex["query"],
            "clause_sequence": [c[0] for c in clauses],
        })

    # Step 6
    complexity_stats = step6_complexity(train_data)

    # Step 7
    report_path = os.path.join(RESULTS_DIR, "spider_dataset_report.md")
    step7_write_report(
        train_stats=train_stats,
        dev_stats=dev_stats,
        table_stats=table_stats,
        complexity_stats=complexity_stats,
        sample_clauses=sample_clauses_10,
        out_path=report_path,
    )

    # Step 8
    clause_stats_combined = {"train": train_stats, "dev": dev_stats}
    step8_save_artifacts(
        clause_stats=clause_stats_combined,
        complexity_stats=complexity_stats,
        sample_clauses=sample_clauses_20,
        results_dir=RESULTS_DIR,
    )

    print(f"\n{'=' * 60}")
    print("All steps complete.")
    print(f"  Results directory: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
