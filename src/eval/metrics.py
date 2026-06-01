"""
Evaluation metrics for clause-PPO.

execution_accuracy — Spider EX: fraction of predicted SQLs whose execution
                     result matches the gold's.
partial_match      — per-clause token F1. Surfaces which clauses are wrong
                     even when EX is 0.
"""

import os
import re
import sys
from collections import Counter
from typing import Sequence

# ── Make src/ and clause_ppo/src importable (config + Henry's oracle) ───────
_REPO_ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC            = os.path.join(_REPO_ROOT, 'src')
_CLAUSE_PPO_SRC = os.path.join(_REPO_ROOT, 'clause_ppo', 'src')
for _p in (_SRC, _CLAUSE_PPO_SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import SPIDER_DIR, TIMEOUT_SECS, CLAUSE_KEYWORDS   # noqa: E402
from utils.execution import queries_produce_same_result        # noqa: E402


# ── Module-level constants ─────────────────────────────────────────────────

# Multi-word keywords first so the regex prefers GROUP BY over GROUP.
_CLAUSE_PATTERN = re.compile(
    r'\b(GROUP\s+BY|ORDER\s+BY|SELECT|FROM|WHERE|HAVING|LIMIT)\b',
    re.IGNORECASE,
)


# ── Public metrics ─────────────────────────────────────────────────────────

def execution_accuracy(
    predictions: Sequence[str],
    samples: Sequence[dict],
    spider_dir: str = SPIDER_DIR,
    timeout_secs: float = TIMEOUT_SECS,
) -> float:
    """
    Standard Spider EX metric.
    Args:
        predictions: predicted SQL strings.
        samples:     Spider samples; must include `db_id` and either `gold_sql` (from load_spider) or raw `query`.
        spider_dir:  path to a Spider snapshot (used to locate `*.sqlite`).
        timeout_secs: hard timeout per execution.
    Returns:
        fraction of predictions whose result matches the gold's.
    """
    if len(predictions) != len(samples):
        raise ValueError(
            f"len(predictions)={len(predictions)} != len(samples)={len(samples)}"
        )
    if not predictions:
        return 0.0

    correct = 0
    for pred, sample in zip(predictions, samples):
        gold = sample.get('gold_sql') or sample.get('query')
        if gold is None:
            continue
        db_path = os.path.join(
            spider_dir, 'database', sample['db_id'], f"{sample['db_id']}.sqlite"
        )
        if queries_produce_same_result(pred, gold, db_path,
                                       timeout_secs=timeout_secs):
            correct += 1
    return correct / len(predictions)


def partial_match(
    predictions: Sequence[str],
    samples: Sequence[dict],
) -> dict[str, float]:
    """
    Per-clause token F1 between predicted and gold SQL strings.
    A clause contributes to its average only when at least one side has non-empty text for it;
    otherwise empty/empty would inflate F1 to 1.0 on clauses neither side uses.
    Args:
        predictions: predicted SQL strings.
        samples:     Spider samples — needs `gold_sql` or raw `query`.
    Returns:
        {clause_keyword: mean F1}. All CLAUSE_KEYWORDS are present; clauses with no qualifying samples get 0.0.
    """
    if len(predictions) != len(samples):
        raise ValueError(
            f"len(predictions)={len(predictions)} != len(samples)={len(samples)}"
        )

    per_clause: dict[str, list[float]] = {c: [] for c in CLAUSE_KEYWORDS}

    for pred, sample in zip(predictions, samples):
        gold = sample.get('gold_sql') or sample.get('query')
        if gold is None:
            continue
        pred_c = _split_sql_clauses(pred)
        gold_c = _split_sql_clauses(gold)
        for c in CLAUSE_KEYWORDS:
            if not pred_c[c] and not gold_c[c]:
                continue
            per_clause[c].append(_token_f1(pred_c[c], gold_c[c]))

    return {
        c: (sum(vals) / len(vals) if vals else 0.0)
        for c, vals in per_clause.items()
    }


# ── Helpers ────────────────────────────────────────────────────────────────

def _split_sql_clauses(sql: str) -> dict[str, str]:
    """
    Flat top-level split of *sql* into {clause_keyword: text}.
    Does not descend into subqueries — intended for diagnostic per-clause F1, not AST equivalence.
    If a keyword appears multiple times (e.g. nested SELECT), the outer (first) occurrence wins.
    """
    result = {c: '' for c in CLAUSE_KEYWORDS}
    matches = list(_CLAUSE_PATTERN.finditer(sql))
    if not matches:
        return result

    for i, m in enumerate(matches):
        kw   = re.sub(r'\s+', ' ', m.group(1).upper())
        start = m.end()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(sql)
        text  = sql[start:end].strip().rstrip(';').strip()
        if not result[kw]:
            result[kw] = text
    return result


def split_sql_prefixes(sql: str) -> list[tuple[str, str]]:
    """
    Split a raw SQL string into (clause_label, cumulative_prefix) pairs.

    Each prefix includes all clauses up to and including that keyword, suitable
    for prefix-based PRM scoring where the faulty clause is argmin over scores.

    Example:
        "SELECT a FROM t WHERE x = 1"
        → [('SELECT', 'SELECT a'),
           ('FROM',   'SELECT a FROM t'),
           ('WHERE',  'SELECT a FROM t WHERE x = 1')]
    """
    matches = list(_CLAUSE_PATTERN.finditer(sql))
    if not matches:
        stripped = sql.strip()
        return [('SELECT', stripped)] if stripped else []

    select_body = sql[:matches[0].start()].strip()
    prefixes = [('SELECT', select_body)] if select_body else []

    for i, m in enumerate(matches):
        kw = re.sub(r'\s+', ' ', m.group(1).upper())
        end = matches[i + 1].start() if i + 1 < len(matches) else len(sql)
        prefix_sql = sql[:end].strip().rstrip(';')
        prefixes.append((kw, prefix_sql))
    
    return prefixes


def _token_f1(pred_text: str, gold_text: str) -> float:
    """Token-level multiset F1, lowercased and whitespace-split."""
    pred_toks = pred_text.lower().split()
    gold_toks = gold_text.lower().split()
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    common = sum((Counter(pred_toks) & Counter(gold_toks)).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_toks)
    recall    = common / len(gold_toks)
    return 2 * precision * recall / (precision + recall)
