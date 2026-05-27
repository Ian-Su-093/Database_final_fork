#!/usr/bin/env python3
"""
Evaluate baseline (full regeneration) and PPO (clause-level repair) on Spider.

Outputs a markdown comparison table:

    | Method     | Accuracy@N | Avg Token Cost |
    | Full regen |    ?       |      ?         |
    | Clause PPO |    ?       |      ?         |

PPO inference is a stub today — see run_clause_ppo() and QUESTIONS.md.

Usage:
    python scripts/evaluate.py --split dev
    python scripts/evaluate.py --split dev --max-retries 3 --max-samples 20
    python scripts/evaluate.py --split dev --ppo-ckpt clause_ppo/results/ppo_checkpoints/ep_3000
"""

import argparse
import json
import os
import sys

# ── Make src/ and clause_ppo/src/ importable when run as a script ──────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (
    os.path.join(_REPO_ROOT, 'src'),
    os.path.join(_REPO_ROOT, 'clause_ppo', 'src'),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from env.env             import NL2SQLEnv
from eval.metrics        import execution_accuracy
from baseline.full_regen import make_hf_api_generate_fn, run_baseline


# ── Defaults ───────────────────────────────────────────────────────────────

DEFAULT_SPIDER_DIR = os.path.join(_REPO_ROOT, 'clause_ppo', 'data', 'spider')
DEFAULT_MODEL_NAME = 'qwen/qwen2.5-coder-1.5b'
DEFAULT_PROVIDER   = 'hf-inference'
DEFAULT_MAX_TOKENS = 500


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--split',       default='dev', choices=['dev', 'train'])
    p.add_argument('--spider-dir',  default=DEFAULT_SPIDER_DIR)
    p.add_argument('--max-retries', type=int, default=3)
    p.add_argument('--model',       default=DEFAULT_MODEL_NAME,
                   help='HF Inference API model id for the baseline backbone.')
    p.add_argument('--provider',    default=DEFAULT_PROVIDER,
                   help='HF InferenceClient provider (e.g. hf-inference).')
    p.add_argument('--hf-token',    default=os.environ.get('HF_TOKEN'),
                   help='HF API token. Falls back to the HF_TOKEN env var.')
    p.add_argument('--max-tokens',  type=int, default=DEFAULT_MAX_TOKENS,
                   help='Max generated tokens per API call.')
    p.add_argument('--ppo-ckpt',    default=None,
                   help='PPO actor checkpoint. PPO path is a stub today (see run_clause_ppo).')
    p.add_argument('--max-samples', type=int, default=None,
                   help='Truncate the split for a quick smoke run.')
    p.add_argument('--output',      default=None,
                   help='Optional path to dump per-sample predictions as JSON.')
    return p.parse_args()


# ── Data + model loading ───────────────────────────────────────────────────

def load_spider(split: str, spider_dir: str) -> list[dict]:
    """Return the raw list of samples from train_spider.json / dev.json."""
    fname = 'dev.json' if split == 'dev' else 'train_spider.json'
    with open(os.path.join(spider_dir, fname)) as f:
        return json.load(f)


def build_inference_client(provider: str, token: str):
    """
    Build a huggingface_hub InferenceClient for the baseline backbone.

    Lazy import — keeps the CLI importable on machines without huggingface_hub.
    """
    from huggingface_hub import InferenceClient

    if not token:
        raise SystemExit(
            "No HF token. Pass --hf-token or set the HF_TOKEN env var "
            "(the baseline calls the HF Inference API)."
        )
    return InferenceClient(provider=provider, token=token)


# ── Per-method runners ─────────────────────────────────────────────────────

def run_full_regen(
    samples:     list[dict],
    generate_fn,
    env:         NL2SQLEnv,
    max_retries: int,
    log_every:   int = 50,
) -> tuple[list[str], list[int], list[int]]:
    """Run the full-regen baseline across all samples."""
    predictions:    list[str] = []
    token_costs:    list[int] = []
    attempt_counts: list[int] = []

    for i, sample in enumerate(samples):
        result = run_baseline(
            sample, generate_fn,
            max_retries=max_retries, env=env,
        )
        predictions.append(result['predicted_sql'])
        token_costs.append(result['token_cost'])
        attempt_counts.append(result['attempts'])

        if (i + 1) % log_every == 0:
            print(f"  [{i+1}/{len(samples)}] baseline running...")

    return predictions, token_costs, attempt_counts


def run_clause_ppo(
    samples:     list[dict],
    ppo_ckpt:    str,
    max_retries: int,
) -> tuple[list[str], list[int], list[int]]:
    """
    Run the PPO actor on each sample. Not implemented yet.

    Henry's ppo_loop.py provides train_ppo() and saves actor checkpoints,
    but exposes no run_ppo_inference() / actor loader. Wire it in here once
    that exists. See .claude/docs/QUESTIONS.md.
    """
    raise NotImplementedError(
        "PPO inference is not implemented. ppo_loop.py has train_ppo() but no "
        "actor-loading / generation entry point. Ask Henry to add a "
        "run_ppo_inference(sample, model, tokenizer) -> dict before passing --ppo-ckpt."
    )


# ── Output ─────────────────────────────────────────────────────────────────

def print_table(rows: list[dict], n: int):
    """Print the comparison table to stdout."""
    header = f"| {'Method':<12} | {f'Accuracy@{n}':>11} | {'Avg Token Cost':>14} |"
    sep    = "|" + "-" * (len(header) - 2) + "|"
    print()
    print(header)
    print(sep)
    for r in rows:
        print(
            f"| {r['method']:<12} | "
            f"{r['accuracy']:>11.3f} | "
            f"{r['avg_tokens']:>14.1f} |"
        )


def dump_predictions(
    output_path: str,
    samples:     list[dict],
    preds:       list[str],
    tokens:      list[int],
    attempts:    list[int],
    args:        argparse.Namespace,
):
    """Write per-sample predictions as JSON for offline inspection."""
    payload = {
        'split':       args.split,
        'max_retries': args.max_retries,
        'samples': [
            {
                'db_id':         s['db_id'],
                'question':      s['question'],
                'gold_sql':      s.get('gold_sql') or s.get('query'),
                'predicted_sql': p,
                'token_cost':    t,
                'attempts':      a,
            }
            for s, p, t, a in zip(samples, preds, tokens, attempts)
        ],
    }
    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote per-sample predictions to {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"Loading Spider {args.split} from {args.spider_dir}")
    samples = load_spider(args.split, args.spider_dir)
    if args.max_samples is not None:
        samples = samples[:args.max_samples]
    print(f"  {len(samples)} samples")

    env = NL2SQLEnv(spider_dir=args.spider_dir)

    print(f"\nBaseline backbone (HF Inference API): {args.model}")
    client      = build_inference_client(args.provider, args.hf_token)
    generate_fn = make_hf_api_generate_fn(client, args.model, max_tokens=args.max_tokens)

    print(f"\nRunning full-regen baseline (max_retries={args.max_retries})")
    preds, tokens, attempts = run_full_regen(
        samples, generate_fn, env, args.max_retries,
    )

    acc        = execution_accuracy(preds, samples, spider_dir=args.spider_dir)
    avg_tokens = sum(tokens) / len(tokens) if tokens else 0.0
    rows       = [{'method': 'Full regen', 'accuracy': acc, 'avg_tokens': avg_tokens}]

    if args.ppo_ckpt is not None:
        print(f"\nRunning Clause PPO (--ppo-ckpt {args.ppo_ckpt})")
        try:
            ppo_preds, ppo_tokens, _ = run_clause_ppo(
                samples, args.ppo_ckpt, args.max_retries,
            )
            ppo_acc = execution_accuracy(ppo_preds, samples, spider_dir=args.spider_dir)
            ppo_avg = sum(ppo_tokens) / len(ppo_tokens) if ppo_tokens else 0.0
            rows.append({
                'method':     'Clause PPO',
                'accuracy':   ppo_acc,
                'avg_tokens': ppo_avg,
            })
        except NotImplementedError as e:
            print(f"  Skipped — {e}")

    print_table(rows, n=args.max_retries)

    if args.output:
        dump_predictions(args.output, samples, preds, tokens, attempts, args)


if __name__ == '__main__':
    main()
