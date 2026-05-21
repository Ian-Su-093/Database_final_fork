"""
Clause splitting utilities for CLAUSE-PPO Phase 1.

split_into_clauses       — Spider sql dict → ordered [(clause_name, content)] tuples
clauses_to_prefix_states — build prefix state dicts for PRM scoring
schema_to_string         — tables.json entry → human-readable schema string

Execution order (FROM→WHERE→GROUPBY→HAVING→SELECT→ORDERBY) matches how the
database engine processes the query, so the PRM's argmin over prefix scores
identifies the first clause where the base model went wrong.
"""

from __future__ import annotations
from typing import Optional


# SQL execution order for CLAUSE-PPO prefix decomposition
CLAUSE_ORDER = ['from', 'where', 'groupBy', 'having', 'select', 'orderBy']

# Human-readable labels for display / logging
CLAUSE_LABELS = {
    'from':    'FROM',
    'where':   'WHERE',
    'groupBy': 'GROUP BY',
    'having':  'HAVING',
    'select':  'SELECT',
    'orderBy': 'ORDER BY',
}


def _is_nonempty(val) -> bool:
    """Return True if a clause value is meaningfully populated."""
    if val is None:
        return False
    if isinstance(val, list):
        if len(val) == 0:
            return False
        # orderBy = ["asc"/"desc", [val_units...]] — inner list may be empty
        if (len(val) == 2 and isinstance(val[0], str)
                and isinstance(val[1], list)):
            return len(val[1]) > 0
        return True
    if isinstance(val, dict):
        return bool(val)
    return True


def split_into_clauses(sql_dict: dict) -> list[tuple[str, object]]:
    """
    Takes the parsed `sql` field from a Spider entry.

    Returns an ordered list of (clause_name, clause_content) tuples in
    SQL execution order:
        FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY

    Only non-empty/non-null clauses are included.
    """
    result: list[tuple[str, object]] = []
    for clause in CLAUSE_ORDER:
        val = sql_dict.get(clause)
        if _is_nonempty(val):
            result.append((clause, val))
    return result


def clauses_to_prefix_states(
    question: str,
    schema_str: str,
    clauses: list[tuple[str, object]],
    query: str,
) -> list[dict]:
    """
    Build one prefix state dict per clause position j (0-indexed).

    Each state represents the input to V_phi at that position:
      - prefix_clauses:    clauses[0..j] inclusive
      - prefix_query_str:  space-joined CLAUSE LABEL tokens up to j
      - clause_position:   j
      - total_clauses:     total number of clauses m
    """
    m = len(clauses)
    states: list[dict] = []
    for j in range(m):
        prefix = clauses[: j + 1]
        parts: list[str] = []
        for name, content in prefix:
            label = CLAUSE_LABELS.get(name, name.upper())
            parts.append(label)
        prefix_str = ' '.join(parts)

        states.append({
            'question':         question,
            'schema':           schema_str,
            'prefix_clauses':   prefix,
            'prefix_query_str': prefix_str,
            'clause_position':  j,
            'total_clauses':    m,
            'full_query':       query,
        })
    return states


def schema_to_string(db_id: str, tables_dict: dict) -> str:
    """
    Convert a tables.json entry into a readable one-line-per-table schema string.

    Format:
        Table: city | Columns: ID (number), Name (text) | PK: ID | FK: CountryCode -> country.Code

    Used as the [SCHEMA] segment in the PRM model input.
    """
    import collections

    db = tables_dict.get(db_id)
    if db is None:
        return f"[schema unavailable for {db_id}]"

    table_names  = db['table_names_original']
    col_names    = db['column_names_original']   # [[table_idx, col_name], ...]
    col_types    = db['column_types']
    foreign_keys = db.get('foreign_keys', [])    # [[col_idx_a, col_idx_b], ...]
    primary_keys = set(db.get('primary_keys', []))

    # Build per-table column lists (skip col_id=0 which is the wildcard *)
    table_cols: dict[int, list] = {i: [] for i in range(len(table_names))}
    for col_id, (t_idx, c_name) in enumerate(col_names):
        if t_idx < 0:   # wildcard entry
            continue
        c_type = col_types[col_id]
        table_cols[t_idx].append((col_id, c_name, c_type))

    # Build FK lookup: col_idx → "target_table.target_col"
    fk_map: dict[int, str] = {}
    for src_idx, tgt_idx in foreign_keys:
        tgt_t_idx, tgt_c_name = col_names[tgt_idx]
        tgt_table = table_names[tgt_t_idx] if tgt_t_idx >= 0 else '?'
        fk_map[src_idx] = f"{tgt_table}.{tgt_c_name}"

    lines: list[str] = []
    for t_idx, t_name in enumerate(table_names):
        cols = table_cols.get(t_idx, [])
        col_strs = [f"{c_name} ({c_type})" for _, c_name, c_type in cols]
        col_part = ', '.join(col_strs) if col_strs else '(no columns)'

        pk_names = [c_name for col_id, c_name, _ in cols if col_id in primary_keys]
        pk_part  = f" | PK: {', '.join(pk_names)}" if pk_names else ''

        fk_strs = [f"{c_name} -> {fk_map[col_id]}"
                   for col_id, c_name, _ in cols if col_id in fk_map]
        fk_part = f" | FK: {', '.join(fk_strs)}" if fk_strs else ''

        lines.append(f"Table: {t_name} | Columns: {col_part}{pk_part}{fk_part}")

    return '\n'.join(lines)
