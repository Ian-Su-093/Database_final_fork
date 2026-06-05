"""
Plan B inference: ClausePRM + Best-of-N clause repair.

Thin adapter between evaluate.py and best_of_n.run_plan_b_for_evaluate().
"""

import os
import sys
from typing import List, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in [
    os.path.join(_REPO_ROOT, 'src'),
    os.path.join(_REPO_ROOT, 'src', 'eval'),
    os.path.join(_REPO_ROOT, 'clause_ppo', 'src'),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def run_plan_b_inference(
    samples:     List[dict],
    prm_ckpt:    str,
    max_retries: int = 3,
    limit:       int = None,
) -> Tuple[List[str], List[int], List[int], List[bool]]:
    """
    Run Plan B inference (ClausePRM + Best-of-N) on a list of Spider samples.

    Args:
        samples:     Spider samples with question, db_id, and gold_sql / query.
        prm_ckpt:    Path to a trained ClausePRM checkpoint directory.
        max_retries: Passed through but unused — Plan B uses oracle selection.
        limit:       Truncate samples for smoke tests.

    Returns:
        (predictions, token_costs, attempt_counts, successes) matching the evaluate.py interface.
    """
    if not os.path.exists(prm_ckpt):
        raise FileNotFoundError(
            f"ClausePRM checkpoint not found: {prm_ckpt}\n"
            "Train the PRM first (see clause_ppo/PHASE1_README.md)."
        )

    if limit is not None:
        samples = samples[:limit]

    from best_of_n import run_plan_b_for_evaluate
    return run_plan_b_for_evaluate(samples, prm_ckpt, max_retries)
