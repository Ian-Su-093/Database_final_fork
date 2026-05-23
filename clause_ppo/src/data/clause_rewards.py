"""
Per-clause reward scoring for corruption_dataset.json.

Rewards are in SQL execution order (FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY).
Clauses before the corrupted clause receive 1.0; the corrupted clause and all clauses
after it receive scores in [0, 0.5].
"""

from __future__ import annotations

import os
import re
from typing import Optional

from data.clause_splitter import CLAUSE_ORDER
from utils.execution import fetch_results, query_result_similarity

# Surface-order keywords in reconstruct_sql output
_SURFACE_KEYWORDS = [
    ('SELECT', 'select'),
    ('FROM', 'from'),
    ('WHERE', 'where'),
    ('GROUP BY', 'groupBy'),
    ('HAVING', 'having'),
    ('ORDER BY', 'orderBy'),
    ('LIMIT', 'limit'),
]

_SYNTAX_NOT_OP = re.compile(r'\bNOT\s*[<>!=]', re.IGNORECASE)
_SYNTAX_AGG_STAR = re.compile(
    r'\b(MAX|MIN|SUM|AVG)\s*\(\s*\*\s*\)', re.IGNORECASE
)

_STRATEGY_BASE = {
    'corrupt_from': 0.15,
    'corrupt_where': 0.12,
    'corrupt_groupBy': 0.18,
    'corrupt_having': 0.15,
    'corrupt_select': 0.20,
    'corrupt_orderBy': 0.30,
}


def _normalize_clause_text(text: str) -> str:
    """Collapse whitespace and uppercase SQL keywords for comparison."""
    s = ' '.join(text.split())
    for kw in ('SELECT', 'FROM', 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY', 'LIMIT',
               'AND', 'OR', 'NOT', 'ASC', 'DESC', 'DISTINCT', 'IN', 'IS', 'NULL',
               'BETWEEN', 'LIKE', 'EXISTS'):
        s = re.sub(rf'\b{kw}\b', kw, s, flags=re.IGNORECASE)
    return s


def extract_clause_texts(sql: str) -> dict[str, str]:
    """
    Parse a SQL string into clause_name → clause body (including keyword prefix).

    Handles reconstruct_sql-style ordering: SELECT ... FROM ... WHERE ...
    """
    sql = sql.strip()
    if not sql:
        return {}

    # Find keyword positions (longest match first within the string scan)
    positions: list[tuple[int, str, str]] = []
    for surface, internal in _SURFACE_KEYWORDS:
        pattern = re.compile(r'\b' + surface.replace(' ', r'\s+') + r'\b', re.IGNORECASE)
        for m in pattern.finditer(sql):
            positions.append((m.start(), internal, surface))

    positions.sort(key=lambda x: x[0])
    if not positions:
        return {}

    # Deduplicate: keep first occurrence per internal name in surface order
    seen: set[str] = set()
    unique: list[tuple[int, str, str]] = []
    for pos, internal, surface in positions:
        if internal in seen:
            continue
        seen.add(internal)
        unique.append((pos, internal, surface))

    fragments: dict[str, str] = {}
    for idx, (pos, internal, surface) in enumerate(unique):
        end = unique[idx + 1][0] if idx + 1 < len(unique) else len(sql)
        fragments[internal] = sql[pos:end].strip()

    return fragments


def extract_clauses_ordered(sql: str) -> list[tuple[str, str]]:
    """Return [(clause_name, clause_text), ...] in execution order."""
    fragments = extract_clause_texts(sql)
    return [(name, fragments[name]) for name in CLAUSE_ORDER if name in fragments]


def is_syntax_invalid_clause(clause_name: str, clause_text: str) -> bool:
    """Return True if this clause fragment is invalid SQL."""
    if _SYNTAX_NOT_OP.search(clause_text):
        return True
    if clause_name == 'select' and _SYNTAX_AGG_STAR.search(clause_text):
        return True
    return False


def _semantic_score(
    clause_name: str,
    orig_text: str,
    corr_text: str,
    strategy: str,
) -> float:
    """Score a changed clause in [0.05, 0.5] using strategy rubric and diff heuristics."""
    base = _STRATEGY_BASE.get(strategy, 0.20)

    o_norm = _normalize_clause_text(orig_text)
    c_norm = _normalize_clause_text(corr_text)

    if clause_name == 'from':
        # Join drop: fewer commas / table names
        o_tables = len(re.findall(r'\bFROM\b|\,', o_norm, re.I))
        c_tables = len(re.findall(r'\bFROM\b|\,', c_norm, re.I))
        if c_tables < o_tables:
            base = 0.10

    elif clause_name == 'select':
        o_cols = o_norm.split('SELECT', 1)[-1].count(',') + 1 if 'SELECT' in o_norm else 1
        c_cols = c_norm.split('SELECT', 1)[-1].count(',') + 1 if 'SELECT' in c_norm else 1
        if c_cols < o_cols:
            base = 0.25
        elif re.search(r'\b(COUNT|MAX|MIN|SUM|AVG)\b', o_norm, re.I) and re.search(
            r'\b(COUNT|MAX|MIN|SUM|AVG)\b', c_norm, re.I
        ):
            o_agg = re.findall(r'\b(COUNT|MAX|MIN|SUM|AVG)\b', o_norm, re.I)
            c_agg = re.findall(r'\b(COUNT|MAX|MIN|SUM|AVG)\b', c_norm, re.I)
            if o_agg != c_agg:
                base = 0.15

    elif clause_name == 'where' and 'NOT' in c_norm.upper() and 'NOT' not in o_norm.upper():
        base = 0.12

    elif clause_name == 'orderBy':
        if ('ASC' in o_norm and 'DESC' in c_norm) or ('DESC' in o_norm and 'ASC' in c_norm):
            base = 0.30

    return min(0.5, max(0.05, base))


def score_clause_pair(
    clause_name: str,
    orig_text: str,
    corr_text: str,
    *,
    strategy: str = '',
) -> float:
    """
    Score one clause comparison in [0.0, 0.5].

    Returns 0.5 if unchanged, 0.0 if syntax-invalid, else semantic tier.
    """
    if _normalize_clause_text(orig_text) == _normalize_clause_text(corr_text):
        return 0.5

    if is_syntax_invalid_clause(clause_name, corr_text):
        return 0.0

    return _semantic_score(clause_name, orig_text, corr_text, strategy)


def compute_clause_rewards(
    record: dict,
    clause_names: list[str],
    orig_clause_texts: dict[str, str],
    db_path: str,
) -> list[float]:
    """
    Build the per-clause reward list for one corruption record.

    Fault index j is derived from corrupted_clause (not corrupted_position).
    """
    corrupted_clause = record['corrupted_clause']
    strategy = record.get('corruption_strategy', f'corrupt_{corrupted_clause}')
    j = clause_names.index(corrupted_clause)

    corr_fragments = extract_clause_texts(record['corrupted_query'])
    rewards: list[float] = []

    db_available = bool(db_path) and os.path.isfile(db_path)

    if db_available:
        ok_gold, gold_rows = fetch_results(record['original_query'], db_path)
        ok_corr, corr_rows = fetch_results(record['corrupted_query'], db_path)
        query_executable = ok_gold and ok_corr
        sim = (
            query_result_similarity(
                record['original_query'], record['corrupted_query'], db_path
            )
            if query_executable
            else 0.0
        )
    else:
        ok_gold, gold_rows = False, None
        ok_corr, corr_rows = False, None
        query_executable = False
        sim = 0.0

    def _maybe_empty_nudge(score: float) -> float:
        if ok_gold and gold_rows and ok_corr and corr_rows is not None and len(corr_rows) == 0:
            return min(score, 0.05)
        return score

    for i, clause_name in enumerate(clause_names):
        orig_text = orig_clause_texts.get(clause_name, '')
        corr_text = corr_fragments.get(clause_name, '')

        if i < j:
            rewards.append(1.0)
            continue

        if is_syntax_invalid_clause(clause_name, corr_text):
            rewards.append(0.0)
            continue

        if i == j:
            s = score_clause_pair(
                clause_name, orig_text, corr_text, strategy=strategy
            )
            rewards.append(round(_maybe_empty_nudge(s), 4))
            continue

        # i > j
        if _normalize_clause_text(orig_text) == _normalize_clause_text(corr_text):
            if query_executable:
                rewards.append(round(0.5 * sim, 4))
            else:
                rewards.append(0.0)
        else:
            s = score_clause_pair(
                clause_name, orig_text, corr_text, strategy=strategy
            )
            rewards.append(round(_maybe_empty_nudge(s), 4))

    return rewards


def build_orig_clause_texts(original_query: str, clause_names: list[str]) -> dict[str, str]:
    """Extract clause texts from the gold query for the given clause name list."""
    fragments = extract_clause_texts(original_query)
    return {name: fragments[name] for name in clause_names if name in fragments}


def validate_rewards(
    rewards: list[float],
    clause_names: list[str],
    corrupted_clause: str,
) -> list[str]:
    """Return a list of sanity-check warning messages (empty if all pass)."""
    warnings: list[str] = []
    j = clause_names.index(corrupted_clause)

    if len(rewards) != len(clause_names):
        warnings.append(
            f'len(reward)={len(rewards)} != len(clause_names)={len(clause_names)}'
        )
        return warnings

    for i, r in enumerate(rewards):
        if r < 0 or r > 1.0:
            warnings.append(f'reward[{i}]={r} out of [0,1]')
        if i < j and r != 1.0:
            warnings.append(f'reward[{i}]={r} expected 1.0 (before fault at j={j})')
        if i >= j and r > 0.5:
            warnings.append(f'reward[{i}]={r} expected <= 0.5 (at/after fault)')

    return warnings
