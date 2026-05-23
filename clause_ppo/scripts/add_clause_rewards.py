#!/usr/bin/env python3
"""
add_clause_rewards.py
=====================
Compute per-clause reward vectors and write them into corruption_dataset.json.

Usage:
  python scripts/add_clause_rewards.py \\
      --processed_dir data/processed \\
      --spider_dir    data/spider \\
      --limit         5
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.clause_rewards import (
    build_orig_clause_texts,
    compute_clause_rewards,
    validate_rewards,
)

MANUAL_OVERRIDES: dict[int, list[float]] = {}


def parse_args():
    p = argparse.ArgumentParser(description='Add clause-level reward vectors to corruption dataset.')
    p.add_argument('--processed_dir', default='data/processed')
    p.add_argument('--spider_dir', default='data/spider')
    p.add_argument('--limit', type=int, default=5,
                   help='Number of records to annotate (-1 for all)')
    p.add_argument('--output', default=None,
                   help='Output path (default: overwrite corruption_dataset.json)')
    return p.parse_args()


def main():
    args = parse_args()

    corr_path = os.path.join(args.processed_dir, 'corruption_dataset.json')
    orig_path = os.path.join(args.processed_dir, 'original_dataset.json')
    out_path = args.output or corr_path

    with open(corr_path) as f:
        corruptions = json.load(f)
    with open(orig_path) as f:
        originals = json.load(f)

    lookup = {(ex['db_id'], ex['question']): ex for ex in originals}

    n = len(corruptions) if args.limit < 0 else min(args.limit, len(corruptions))
    print(f'Annotating {n} / {len(corruptions)} corruption records ...\n')

    warnings_total = 0
    for i in range(n):
        record = corruptions[i]
        key = (record['db_id'], record['question'])
        orig_ex = lookup.get(key)
        if orig_ex is None:
            print(f'  [{i}] SKIP — no original_dataset match for {key!r}')
            continue

        clause_names = orig_ex['clause_names']
        db_path = os.path.join(
            args.spider_dir, 'database', record['db_id'], f"{record['db_id']}.sqlite"
        )

        orig_texts = build_orig_clause_texts(record['original_query'], clause_names)

        if i in MANUAL_OVERRIDES:
            rewards = MANUAL_OVERRIDES[i]
        else:
            if not os.path.isfile(db_path):
                print(f'  [{i}] NOTE — database not found, downstream sim=0: {db_path}')
            rewards = compute_clause_rewards(
                record, clause_names, orig_texts, db_path
            )

        for w in validate_rewards(rewards, clause_names, record['corrupted_clause']):
            print(f'  [{i}] WARN: {w}')
            warnings_total += 1

        record['reward'] = rewards
        print(
            f"  [{i}] {record['corrupted_clause']:<10}  "
            f"clauses={clause_names}  reward={rewards}"
        )

    with open(out_path, 'w') as f:
        json.dump(corruptions, f, indent=2)

    print(f'\nWrote {out_path}  ({warnings_total} warnings)')


if __name__ == '__main__':
    main()
