#!/usr/bin/env python3
"""
Entry point for PRM training.

Usage:
  python scripts/train_prm.py \
      --config       configs/prm_config.yaml \
      --spider_dir   data/spider \
      --processed_dir data/processed
"""

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',        default='configs/prm_config.yaml')
    p.add_argument('--spider_dir',    default='data/spider')
    p.add_argument('--processed_dir', default='data/processed')
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("Configuration loaded:")
    print(yaml.dump(config, default_flow_style=False))

    from training.train_prm import train_prm
    train_prm(
        config=config,
        spider_dir=args.spider_dir,
        processed_dir=args.processed_dir,
    )


if __name__ == '__main__':
    main()
