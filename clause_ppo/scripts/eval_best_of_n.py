#!/usr/bin/env python3
"""
Entry point for Best-of-N clause repair evaluation (Baseline 3 / Plan B).

Usage:
  python scripts/eval_best_of_n.py \
      --config    configs/eval_config.yaml \
      --spider_dir ../../spider \
      --prm_ckpt  results/prm_checkpoints/best_checkpoint
"""

import argparse
import os
import sys

import yaml

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPTS_DIR, '..', 'src'))        # clause_ppo/src
sys.path.insert(0, os.path.join(_SCRIPTS_DIR, '..', '..', 'src'))  # src/ (env, eval)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',     default='configs/eval_config.yaml')
    p.add_argument('--spider_dir', default='../../spider')
    p.add_argument('--prm_ckpt',   default='results/prm_checkpoints/best_checkpoint')
    p.add_argument('--limit',      type=int, default=None, help='Limit number of samples for testing')
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("Configuration loaded:")
    print(yaml.dump(config, default_flow_style=False))

    from eval.best_of_n import eval_best_of_n
    eval_best_of_n(
        config=config,
        spider_dir=args.spider_dir,
        prm_ckpt=args.prm_ckpt,
        limit=args.limit,
    )


if __name__ == '__main__':
    main()