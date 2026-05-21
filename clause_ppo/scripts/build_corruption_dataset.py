#!/usr/bin/env python3
"""
build_corruption_dataset.py
============================
Runs the corruption pipeline over Spider train examples and saves:
  - data/processed/corruption_dataset.json   — verified corruption records
  - data/processed/original_dataset.json     — filtered original examples with clause splits
  - data/processed/corruption_stats.json     — diagnostic statistics

Usage:
  python scripts/build_corruption_dataset.py \
      --spider_dir  data/spider \
      --output_dir  data/processed \
      --split       train \
      --max_examples -1 \
      --min_clauses  2 \
      --log_every    100
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

from tqdm import tqdm

# Ensure src/ is on the path when run from clause_ppo/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.clause_splitter import split_into_clauses, clauses_to_prefix_states, schema_to_string
from data.corruption import generate_corruptions


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--spider_dir',    default='data/spider')
    p.add_argument('--output_dir',    default='data/processed')
    p.add_argument('--split',         default='train',
                   choices=['train', 'dev'])
    p.add_argument('--max_examples',  type=int, default=-1,
                   help='-1 means process all examples')
    p.add_argument('--min_clauses',   type=int, default=2,
                   help='Skip queries with fewer than this many clauses')
    p.add_argument('--log_every',     type=int, default=100)
    return p.parse_args()


def _make_serializable(obj):
    """Recursively convert tuples to lists for JSON serialization."""
    if isinstance(obj, tuple):
        return [_make_serializable(item) for item in obj]
    if isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    return obj


def main():
    args = parse_args()

    # ── Load data ────────────────────────────────────────────────────────────
    split_file = ('train_spider.json' if args.split == 'train' else 'dev.json')
    train_path  = os.path.join(args.spider_dir, split_file)
    tables_path = os.path.join(args.spider_dir, 'tables.json')

    print(f"Loading {train_path} ...")
    with open(train_path) as f:
        examples = json.load(f)

    print(f"Loading {tables_path} ...")
    with open(tables_path) as f:
        tables_dict = {t['db_id']: t for t in json.load(f)}

    if args.max_examples > 0:
        examples = examples[: args.max_examples]

    print(f"\nProcessing {len(examples):,} examples "
          f"(min_clauses={args.min_clauses}) ...")

    # ── Stats tracking ───────────────────────────────────────────────────────
    stats = {
        'total_processed':       0,
        'skipped_few_clauses':   0,
        'skipped_no_corruption': 0,
        'total_corruptions':     0,
        'per_clause': defaultdict(lambda: {'attempts': 0, 'verified': 0}),
    }

    corruption_records: list[dict] = []
    original_records:   list[dict] = []

    t0 = time.time()
    for i, ex in enumerate(tqdm(examples, desc='Corrupting')):
        stats['total_processed'] += 1

        clauses = split_into_clauses(ex['sql'])
        if len(clauses) < args.min_clauses:
            stats['skipped_few_clauses'] += 1
            continue

        corruptions = generate_corruptions(ex, tables_dict)

        # Track per-clause attempts (any clause that appears in this query counts as an attempt)
        attempted_clauses = {c[0] for c in clauses}
        for clause_name in attempted_clauses:
            stats['per_clause'][clause_name]['attempts'] += 1

        for c in corruptions:
            stats['per_clause'][c['corrupted_clause']]['verified'] += 1

        if not corruptions:
            stats['skipped_no_corruption'] += 1
            continue

        corruption_records.extend(corruptions)
        stats['total_corruptions'] += len(corruptions)

        # Save original with pre-computed clause splits and prefix states
        schema = schema_to_string(ex['db_id'], tables_dict)
        prefix_states = clauses_to_prefix_states(
            ex['question'], schema, clauses, ex['query']
        )
        original_records.append({
            'db_id':         ex['db_id'],
            'question':      ex['question'],
            'query':         ex['query'],
            'clause_names':  [name for name, _ in clauses],
            'schema':        schema,
            'prefix_states': _make_serializable(prefix_states),
        })

        if (i + 1) % args.log_every == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1:>5}/{len(examples)}]  "
                  f"corruptions so far: {stats['total_corruptions']:,}  "
                  f"({elapsed:.0f}s elapsed)")

    # ── Finalise stats ───────────────────────────────────────────────────────
    final_stats = {
        'total_processed':            stats['total_processed'],
        'skipped_few_clauses':        stats['skipped_few_clauses'],
        'skipped_no_corruption':      stats['skipped_no_corruption'],
        'total_corruptions':          stats['total_corruptions'],
        'examples_with_corruption':   len(original_records),
        'per_clause_stats': {
            clause: {
                'attempts':  v['attempts'],
                'verified':  v['verified'],
                'pass_rate': round(v['verified'] / v['attempts'], 4) if v['attempts'] else 0,
            }
            for clause, v in stats['per_clause'].items()
        },
    }

    # ── Save outputs ─────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    def _save(obj, fname):
        path = os.path.join(args.output_dir, fname)
        with open(path, 'w') as f:
            json.dump(obj, f, indent=2)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  Saved {path}  ({size_mb:.1f} MB, {len(obj):,} records)")

    print('\nSaving outputs ...')
    _save(corruption_records, 'corruption_dataset.json')
    _save(original_records,   'original_dataset.json')
    with open(os.path.join(args.output_dir, 'corruption_stats.json'), 'w') as f:
        json.dump(final_stats, f, indent=2)
    print(f"  Saved {os.path.join(args.output_dir, 'corruption_stats.json')}")

    # ── Print summary ────────────────────────────────────────────────────────
    print('\n── Corruption Statistics ──────────────────────────────')
    print(f"  Examples processed      : {final_stats['total_processed']:,}")
    print(f"  Skipped (< {args.min_clauses} clauses)   : {final_stats['skipped_few_clauses']:,}")
    print(f"  Skipped (no corruption) : {final_stats['skipped_no_corruption']:,}")
    print(f"  Examples with >=1 corr. : {final_stats['examples_with_corruption']:,}")
    print(f"  Total corruption records: {final_stats['total_corruptions']:,}")
    print(f"\n  Per-clause verification pass rates:")
    for clause, cstats in final_stats['per_clause_stats'].items():
        print(f"    {clause:<12}  attempts={cstats['attempts']:>5}  "
              f"verified={cstats['verified']:>5}  "
              f"pass_rate={cstats['pass_rate']:.3f}")
    print()


if __name__ == '__main__':
    main()
