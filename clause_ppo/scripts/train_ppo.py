#!/usr/bin/env python3
"""
Entry point for PPO training (Phase 2).

Usage:
  python scripts/train_ppo.py \
      --config     configs/ppo_config.yaml \
      --spider_dir ../../spider \
      --prm_ckpt   results/prm_checkpoints/best_checkpoint
"""

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',      default='configs/ppo_config.yaml')
    p.add_argument('--spider_dir',  default='../../spider')
    p.add_argument('--prm_ckpt',    default='results/prm_checkpoints/best_checkpoint')
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("Configuration loaded:")
    print(yaml.dump(config, default_flow_style=False))

    from training.ppo_loop import train_ppo
    train_ppo(
        config=config,
        spider_dir=args.spider_dir,
        prm_ckpt=args.prm_ckpt,
    )


if __name__ == '__main__':
    main()
