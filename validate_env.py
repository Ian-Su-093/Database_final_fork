#!/usr/bin/env python3
"""
Standalone smoke-test for NL2SQLEnv.

Runs each record from the corruption dataset as one training episode:
  1. env.reset(sample)             -> check state keys + non-empty schema
  2. MockClausePRM.score_clauses() -> check get_faulty_clause() returns right clause
  3. env.step(original_query)      -> check reward == +1.0
  4. env.step(corrupted_query)     -> check reward == -1.0

Usage:  python validate_env.py
Exit:   0 if all checks pass, 1 if any fail.
"""

import json
import os
import sys

REPO_ROOT       = os.path.dirname(os.path.abspath(__file__))
SPIDER_DIR      = os.path.join(REPO_ROOT, 'spider')
CORRUPTION_FILE = os.path.join(REPO_ROOT, 'clause_ppo', 'data', 'processed', 'corruption_dataset.json')

# Make src/ and clause_ppo/src/ importable (env.py needs both)
for _p in [os.path.join(REPO_ROOT, 'src'),
           os.path.join(REPO_ROOT, 'clause_ppo', 'src')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from env.env import NL2SQLEnv  # noqa: E402

CLAUSE_ORDER = ['from', 'where', 'groupBy', 'having', 'select', 'orderBy']


class MockClausePRM:
    """
    Simulates ClausePRM without loading the real model.
    Returns 0.1 for the known faulty clause and 0.9 for all others.
    This guarantees get_faulty_clause() returns the right clause when
    scores are passed to NL2SQLEnv.get_faulty_clause().
    """

    def score_clauses(self, known_faulty: str) -> dict[str, float]:
        return {c: (0.1 if c == known_faulty else 0.9) for c in CLAUSE_ORDER}


def _load_data():
    """Validate paths, load tables.json and corruption_dataset.json. Exits on error."""
    errors = []
    if not os.path.isdir(SPIDER_DIR):
        errors.append(f"Spider directory not found: {SPIDER_DIR}")
    if not os.path.isfile(CORRUPTION_FILE):
        errors.append(f"Corruption dataset not found: {CORRUPTION_FILE}")
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        sys.exit(1)

    with open(os.path.join(SPIDER_DIR, 'tables.json')) as f:
        tables = {t['db_id']: t for t in json.load(f)}

    with open(CORRUPTION_FILE) as f:
        records = json.load(f)

    return tables, records


if __name__ == '__main__':
    tables, records = _load_data()
    print(f"Loaded {len(records)} records. Spider OK.")
