"""
Rule-based SQL corruption engine for CLAUSE-PPO PRM training data generation.

Each corrupt_* function takes a Spider parsed sql_dict and the tables entry for
that db_id, and returns a DEEP COPY of the dict with exactly one clause mutated,
or None if no valid mutation is possible for that clause.

generate_corruptions orchestrates all six corruptions for one example, verifying
via the execution oracle that each mutation actually changes the query result.

Design:
  - Never mutate the input sql_dict (always deep-copy first).
  - Return None rather than a trivially equivalent mutation.
  - Keep strategies simple and deterministic (pick first valid option).
"""

import copy
import os
from typing import Optional

from utils.sql_utils import reconstruct_sql
from utils.execution import queries_produce_same_result
from data.clause_splitter import split_into_clauses


# ── Aggregate operator constants ──────────────────────────────────────────────
AGG_OPS      = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
REAL_AGG_IDS = [1, 2, 3, 4, 5]   # non-zero agg_ids

# WHERE operator ids for flipping comparisons
_OP_GT, _OP_LT, _OP_GTE, _OP_LTE = 3, 4, 5, 6
_OP_EQ, _OP_NEQ = 2, 7
_FLIP_OP = {_OP_GT: _OP_LT, _OP_LT: _OP_GT,
            _OP_GTE: _OP_LTE, _OP_LTE: _OP_GTE,
            _OP_EQ: _OP_NEQ, _OP_NEQ: _OP_EQ}


# ── Column lookup helpers ─────────────────────────────────────────────────────

def _cols_for_table(table_idx: int, tables: dict) -> list:
    """Return [(col_id, col_name, col_type), ...] for a given table_idx (skip wildcard)."""
    result = []
    for col_id, (t_idx, c_name) in enumerate(tables['column_names_original']):
        if t_idx == table_idx and c_name != '*':
            result.append((col_id, c_name, tables['column_types'][col_id]))
    return result


def _col_type(col_id: int, tables: dict) -> str:
    return tables['column_types'][col_id]


def _table_idx_of(col_id: int, tables: dict) -> int:
    return tables['column_names_original'][col_id][0]


# ── Per-clause corruption functions ──────────────────────────────────────────

def corrupt_select(sql_dict: dict, tables: dict) -> Optional[dict]:
    """
    Corrupt the SELECT clause via one of three strategies (tried in order):
      1. Swap an aggregate function for a different one.
      2. Drop one column if there are multiple.
      3. Replace a column with a different column of the same type from the same table.
    Returns a deep-copied, mutated sql_dict, or None if no mutation is found.
    """
    sql = copy.deepcopy(sql_dict)
    select = sql.get('select')
    if not select or len(select) < 2:
        return None
    agg_items = select[1]   # list of [agg_id, val_unit]
    if not agg_items:
        return None

    # Strategy 1: swap an aggregate at the top-level agg_id
    for i, item in enumerate(agg_items):
        agg_id = item[0]
        if agg_id > 0:
            alternatives = [a for a in REAL_AGG_IDS if a != agg_id]
            if alternatives:
                sql['select'][1][i][0] = alternatives[0]
                return sql
        # Also check nested col_unit agg_id inside val_unit
        val_unit = item[1]   # [unit_op, col_unit, col_unit2]
        col_unit = val_unit[1]   # [agg_id, col_id, is_distinct]
        if col_unit and col_unit[0] > 0:
            alternatives = [a for a in REAL_AGG_IDS if a != col_unit[0]]
            if alternatives:
                sql['select'][1][i][1][1][0] = alternatives[0]
                return sql

    # Strategy 2: drop one column from multi-column select
    if len(agg_items) > 1:
        sql['select'][1] = agg_items[1:]
        return sql

    # Strategy 3: replace single column with same-type sibling
    item = agg_items[0]
    val_unit = item[1]
    col_unit = val_unit[1]   # [agg_id, col_id, is_distinct]
    if col_unit:
        col_id = col_unit[1]
        if col_id == 0:    # wildcard — skip
            return None
        t_idx    = _table_idx_of(col_id, tables)
        col_type = _col_type(col_id, tables)
        siblings = [c for c in _cols_for_table(t_idx, tables)
                    if c[0] != col_id and c[2] == col_type]
        if siblings:
            sql['select'][1][0][1][1][1] = siblings[0][0]
            return sql

    return None


def corrupt_from(sql_dict: dict, tables: dict) -> Optional[dict]:
    """
    Corrupt the FROM clause:
      1. Drop one non-first table_unit if there are multiple (breaks JOIN).
      2. If only one table, swap it with a different table in the same database.
    """
    sql = copy.deepcopy(sql_dict)
    from_clause = sql.get('from') or {}
    table_units = from_clause.get('table_units', [])

    real_units = [u for u in table_units if u[0] == 'table_unit']
    if len(real_units) > 1:
        # Strategy 1: drop last table_unit and its join conditions
        dropped_t_idx = real_units[-1][1]
        new_units = [u for u in table_units if not (u[0] == 'table_unit' and u[1] == dropped_t_idx)]

        # Remove join conditions that reference the dropped table
        old_conds = from_clause.get('conds', [])
        new_conds = []
        skip_next_connector = False
        for item in old_conds:
            if isinstance(item, str):
                if not skip_next_connector:
                    new_conds.append(item)
                skip_next_connector = False
            else:
                _, _, val_unit, val1, _ = item
                col_id_lhs = val_unit[1][1]  # val_unit[1] = col_unit, [1] = col_id
                col_id_rhs = (val1[1] if isinstance(val1, list)
                              and len(val1) == 3 and isinstance(val1[1], int)
                              else -1)
                lhs_t = _table_idx_of(col_id_lhs, tables) if col_id_lhs > 0 else -1
                rhs_t = _table_idx_of(col_id_rhs, tables) if col_id_rhs > 0 else -1
                if lhs_t == dropped_t_idx or rhs_t == dropped_t_idx:
                    skip_next_connector = True
                else:
                    new_conds.append(item)

        # Strip any leading/trailing connectors left by the pruning pass.
        # This happens when the dropped condition was last (trailing 'and'/'or')
        # or first (leading 'and'/'or') in the list.
        while new_conds and isinstance(new_conds[-1], str):
            new_conds.pop()
        while new_conds and isinstance(new_conds[0], str):
            new_conds.pop(0)

        sql['from']['table_units'] = new_units
        sql['from']['conds']       = new_conds
        return sql

    if len(real_units) == 1:
        # Strategy 2: swap single table with another table in the same db
        current_idx = real_units[0][1]
        n_tables = len(tables['table_names_original'])
        alternatives = [i for i in range(n_tables) if i != current_idx]
        if alternatives:
            sql['from']['table_units'] = [['table_unit', alternatives[0]]]
            sql['from']['conds'] = []
            return sql

    return None


def corrupt_where(sql_dict: dict, tables: dict) -> Optional[dict]:
    """
    Corrupt the WHERE clause:
      1. Negate one condition (flip not_flag).
      2. Replace a column reference with a same-type sibling column.
      3. Flip AND ↔ OR between conditions.
    """
    sql   = copy.deepcopy(sql_dict)
    where = sql.get('where') or []
    conds = [item for item in where if not isinstance(item, str)]
    if not conds:
        return None

    # Strategy 1: negate first condition (flip not_flag)
    for i, item in enumerate(sql['where']):
        if not isinstance(item, str):
            sql['where'][i][0] = not item[0]
            return sql

    # Strategies 2 and 3 below are dead code in the current implementation:
    # Strategy 1 always returns when there is at least one condition, because
    # Spider WHERE lists always contain at least one non-string item when conds
    # is non-empty, and the for-loop above fires on the very first one.
    # They are preserved here for spec completeness and future refactoring.

    # Strategy 2: replace column reference in first condition
    cond = sql['where'][0]
    val_unit = cond[2]      # [unit_op, col_unit, col_unit2]
    col_unit = val_unit[1]  # [agg_id, col_id, is_distinct]
    col_id   = col_unit[1]
    if col_id > 0:
        t_idx    = _table_idx_of(col_id, tables)
        col_type = _col_type(col_id, tables)
        siblings = [c for c in _cols_for_table(t_idx, tables)
                    if c[0] != col_id and c[2] == col_type]
        if siblings:
            sql['where'][0][2][1][1] = siblings[0][0]
            return sql

    # Strategy 3: flip AND ↔ OR
    connectors = [i for i, item in enumerate(sql['where']) if isinstance(item, str)]
    if connectors:
        i = connectors[0]
        sql['where'][i] = 'or' if sql['where'][i] == 'and' else 'and'
        return sql

    return None


def corrupt_group_by(sql_dict: dict, tables: dict) -> Optional[dict]:
    """
    Corrupt the GROUP BY clause:
      1. Replace the first GROUP BY column with a different column from the same table.
      2. Remove one column if multiple GROUP BY columns exist.
    """
    sql      = copy.deepcopy(sql_dict)
    group_by = sql.get('groupBy') or []
    if not group_by:
        return None

    col_unit = group_by[0]   # [agg_id, col_id, is_distinct]
    col_id   = col_unit[1]

    # Strategy 1: replace first GROUP BY column
    if col_id > 0:
        t_idx    = _table_idx_of(col_id, tables)
        all_cols = _cols_for_table(t_idx, tables)
        others   = [c for c in all_cols if c[0] != col_id]
        if others:
            sql['groupBy'][0][1] = others[0][0]
            return sql

    # Strategy 2: remove one column if multiple
    if len(group_by) > 1:
        sql['groupBy'] = group_by[1:]
        return sql

    return None


def corrupt_having(sql_dict: dict, tables: dict) -> Optional[dict]:
    """
    Corrupt the HAVING clause:
      1. Flip the comparison operator (> ↔ <, >= ↔ <=, = ↔ !=).
      2. Replace the aggregate function in the condition.
    """
    sql    = copy.deepcopy(sql_dict)
    having = sql.get('having') or []
    if not having:
        return None

    for i, item in enumerate(sql['having']):
        if isinstance(item, str):
            continue

        # Strategy 1: flip comparison operator
        op_id = item[1]
        if op_id in _FLIP_OP:
            sql['having'][i][1] = _FLIP_OP[op_id]
            return sql

        # Strategy 2: replace aggregate in val_unit col_unit
        val_unit = item[2]   # [unit_op, col_unit, col_unit2]
        col_unit = val_unit[1]
        if col_unit and col_unit[0] > 0:
            alternatives = [a for a in REAL_AGG_IDS if a != col_unit[0]]
            if alternatives:
                sql['having'][i][2][1][0] = alternatives[0]
                return sql

    return None


def corrupt_order_by(sql_dict: dict, tables: dict) -> Optional[dict]:
    """
    Corrupt the ORDER BY clause:
      1. Flip ASC ↔ DESC.
    """
    sql      = copy.deepcopy(sql_dict)
    order_by = sql.get('orderBy') or []
    if not order_by or len(order_by) != 2:
        return None
    direction, val_units = order_by
    if not val_units:
        return None

    # Strategy 1: flip direction
    sql['orderBy'][0] = 'asc' if direction == 'desc' else 'desc'
    return sql


# ── Orchestration ─────────────────────────────────────────────────────────────

_CORRUPT_FNS = {
    'from':    corrupt_from,
    'where':   corrupt_where,
    'groupBy': corrupt_group_by,
    'having':  corrupt_having,
    'select':  corrupt_select,
    'orderBy': corrupt_order_by,
}


def generate_corruptions(example: dict, tables_dict: dict) -> list[dict]:
    """
    Generate all verified corruption records for one Spider train example.

    For each clause position j:
      - Apply corrupt_* function
      - Reconstruct corrupted SQL
      - Verify via execution oracle that result changed
      - If verified, record the corruption

    Returns list of verified corruption dicts. May be empty.
    """
    db_id    = example['db_id']
    question = example['question']
    sql_dict = example['sql']
    tables   = tables_dict[db_id]

    # Locate the sqlite database file
    # Search relative to this file's location
    this_dir   = os.path.dirname(os.path.abspath(__file__))
    spider_dir = os.path.normpath(os.path.join(this_dir, '..', '..', 'data', 'spider'))
    db_path    = os.path.join(spider_dir, 'database', db_id, f'{db_id}.sqlite')

    orig_reconstructed = reconstruct_sql(sql_dict, tables)

    clauses = split_into_clauses(sql_dict)
    results: list[dict] = []

    for j, (clause_name, _) in enumerate(clauses):
        corrupt_fn = _CORRUPT_FNS.get(clause_name)
        if corrupt_fn is None:
            continue

        corrupted_dict = corrupt_fn(sql_dict, tables)
        if corrupted_dict is None:
            continue

        try:
            corrupted_sql = reconstruct_sql(corrupted_dict, tables)
        except Exception:
            continue

        try:
            changed = not queries_produce_same_result(
                orig_reconstructed, corrupted_sql, db_path
            )
        except Exception:
            changed = False

        if not changed:
            continue

        results.append({
            'db_id':              db_id,
            'question':           question,
            'original_query':     orig_reconstructed,
            'corrupted_query':    corrupted_sql,
            'corrupted_clause':   clause_name,
            'corrupted_position': j,
            'corruption_strategy': f"corrupt_{clause_name}",
        })

    return results
