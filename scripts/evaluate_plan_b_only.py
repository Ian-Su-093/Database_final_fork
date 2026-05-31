#!/usr/bin/env python3
"""
Evaluate Plan B (ClausePRM + Best-of-N) only, without baseline comparison.

This script runs Plan B inference without requiring HuggingFace Hub dependencies.

Usage:
    python scripts/evaluate_plan_b_only.py --split dev --max-samples 10
    python scripts/evaluate_plan_b_only.py --split dev --prm-ckpt clause_ppo/results/prm_checkpoints/best_checkpoint
"""

import argparse
import json
import os
import sys

# ── Make src/ and clause_ppo/src/ importable ──────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (
    os.path.join(_REPO_ROOT, 'src'),
    os.path.join(_REPO_ROOT, 'clause_ppo', 'src'),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from env.env import NL2SQLEnv
from eval.metrics import execution_accuracy
from baseline.plan_b_inference import run_plan_b_inference
from config import SPIDER_DIR


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--split',       default='dev', choices=['dev', 'train'])
    p.add_argument('--spider-dir',  default=SPIDER_DIR)
    p.add_argument('--prm-ckpt',    default='clause_ppo/results/prm_checkpoints/best_checkpoint',
                   help='ClausePRM checkpoint for Plan B inference.')
    p.add_argument('--max-samples', type=int, default=None,
                   help='Truncate the split for a quick test run.')
    p.add_argument('--output',      default=None,
                   help='Optional path to dump per-sample predictions as JSON.')
    return p.parse_args()


def load_spider(split: str, spider_dir: str) -> list[dict]:
    """Return the raw list of samples from train_spider.json / dev.json."""
    fname = 'dev.json' if split == 'dev' else 'train_spider.json'
    with open(os.path.join(spider_dir, fname)) as f:
        return json.load(f)


def print_plan_b_results(accuracy: float, avg_tokens: float, n_samples: int):
    """Print Plan B results in a nice table format."""
    print("\n" + "=" * 50)
    print("PLAN B RESULTS ON SPIDER DEV")
    print("=" * 50)
    print(f"{'Method':<25} {'Accuracy':>10} {'Avg Tokens':>12}")
    print("-" * 50)
    print(f"{'Plan B (ClausePRM + BoN)':<25} {accuracy:>10.4f} {avg_tokens:>12.1f}")
    print("=" * 50)
    print(f"Samples evaluated: {n_samples}")
    print(f"ClausePRM approach: Pure reward model (no RL training)")


def dump_plan_b_predictions(
    output_path: str,
    samples: list[dict],
    preds: list[str],
    tokens: list[int],
    attempts: list[int],
    accuracy: float,
    args: argparse.Namespace,
):
    """Write Plan B predictions as JSON for offline inspection."""
    payload = {
        'method': 'Plan B (ClausePRM + Best-of-N)',
        'split': args.split,
        'prm_checkpoint': args.prm_ckpt,
        'accuracy': accuracy,
        'avg_tokens': sum(tokens) / len(tokens) if tokens else 0.0,
        'n_samples': len(samples),
        'samples': [
            {
                'idx': i,
                'db_id': s['db_id'],
                'question': s['question'],
                'gold_sql': s.get('gold_sql') or s.get('query'),
                'predicted_sql': p,
                'token_cost': t,
                'attempts': a,
            }
            for i, (s, p, t, a) in enumerate(zip(samples, preds, tokens, attempts))
        ],
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")


def main():
    args = parse_args()

    print(f"🎯 Plan B Evaluation: ClausePRM + Best-of-N")
    print(f"Loading Spider {args.split} from {args.spider_dir}")
    
    samples = load_spider(args.split, args.spider_dir)
    if args.max_samples is not None:
        samples = samples[:args.max_samples]
    print(f"  {len(samples)} samples")

    print(f"\nPlan B configuration:")
    print(f"  ClausePRM checkpoint: {args.prm_ckpt}")
    print(f"  Approach: Pure reward model (no RL training)")
    print(f"  Method: Clause scoring + Best-of-N repair + Oracle selection")

    # Run Plan B inference
    print(f"\n🚀 Running Plan B inference...")
    predictions, token_costs, attempts = run_plan_b_inference(
        samples=samples,
        prm_ckpt=args.prm_ckpt,
        max_retries=3,  # Not used by Plan B
        limit=args.max_samples
    )

    # Calculate execution accuracy
    print(f"\n📊 Evaluating predictions...")
    accuracy = execution_accuracy(predictions, samples, spider_dir=args.spider_dir)
    avg_tokens = sum(token_costs) / len(token_costs) if token_costs else 0.0

    # Display results
    print_plan_b_results(accuracy, avg_tokens, len(samples))

    # Save detailed results if requested
    if args.output:
        dump_plan_b_predictions(
            args.output, samples, predictions, token_costs, attempts, accuracy, args
        )

    # Summary
    print(f"\n✅ Plan B evaluation complete!")
    print(f"🎯 Execution Accuracy: {accuracy:.4f}")
    print(f"🔧 Token Efficiency: {avg_tokens:.1f} tokens/query")
    
    if accuracy > 0.3:
        print("🎉 Plan B shows promising results!")
    elif accuracy > 0.1:
        print("💡 Plan B shows some improvement potential.")
    else:
        print("🔧 Plan B may need debugging or better ClausePRM training.")


if __name__ == '__main__':
    main()