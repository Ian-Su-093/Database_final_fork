"""
CLI for scoring a clause prefix with a trained PRM.

Example:
    python scripts/score_clause.py \
        --checkpoint results/prm_checkpoints/best_checkpoint \
        --base_model /home/henrylin0822/models/qwen \
        --question "How many singers are there?" \
        --schema "singer(singer_id, name, age)" \
        --prefix "SELECT count(*) FROM singer"
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def main():
    parser = argparse.ArgumentParser(description='Score a SQL clause prefix with the PRM.')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to checkpoint dir (e.g. results/prm_checkpoints/best_checkpoint)')
    parser.add_argument('--base_model', default=None,
                        help='Base model path or HF name. Auto-detected from checkpoint if omitted.')
    parser.add_argument('--question', required=True, help='Natural language question')
    parser.add_argument('--schema', required=True, help='Database schema string')
    parser.add_argument('--prefix', required=True, help='SQL prefix to score')
    parser.add_argument('--max_length', type=int, default=512)
    args = parser.parse_args()

    from models.prm_inference import PRMScorer

    scorer = PRMScorer(
        checkpoint_dir=args.checkpoint,
        base_model=args.base_model,
        max_length=args.max_length,
    )

    score = scorer.score(args.question, args.schema, args.prefix)
    print(f"\nScore: {score:.4f}  ({'likely correct' if score >= 0.5 else 'likely corrupted'})")


if __name__ == '__main__':
    main()
