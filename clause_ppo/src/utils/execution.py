"""
SQLite execution oracle for CLAUSE-PPO.

execute_query      — run a SQL string against a .sqlite file; optionally treat empty results as failure.
queries_produce_same_result — compare two queries' result sets (order-insensitive).

Timeout is implemented with a daemon thread so it works on any OS (signal.alarm
is UNIX-only and requires the main thread).
"""

import sqlite3
import threading
from typing import Optional


class _QueryTimeoutError(Exception):
    pass


def _run_with_timeout(fn, timeout_secs: float):
    """Run fn() in a daemon thread; raise _QueryTimeoutError if it takes too long."""
    result: list = [None]
    exc: list = [None]

    def _target():
        try:
            result[0] = fn()
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout_secs)
    if t.is_alive():
        # NOTE: On timeout, the daemon thread continues running until the SQLite
        # operation completes or the process exits. The connection is not forcibly
        # closed because SQLite has no cancellation API. This is acceptable for
        # our use case (short-lived process, rare timeouts).
        raise _QueryTimeoutError(f"Query exceeded {timeout_secs}s timeout")
    if exc[0] is not None:
        raise exc[0]
    return result[0]


def execute_query(query: str, db_path: str,
                  timeout_secs: float = 5.0,
                  allow_empty: bool = False) -> tuple[bool, Optional[list]]:
    """
    Execute *query* against the SQLite database at *db_path*.

    Returns:
        (True,  list[list])  — query ran successfully (rows may be empty if
                               *allow_empty* is True).
        (False, None)        — query raised an exception or timed out.
        (False, [])          — query ran but returned 0 rows when *allow_empty*
                               is False (empty results provide no training signal).
    """
    def _run():
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            cur.execute(query)
            rows = cur.fetchall()
            return [list(r) for r in rows]
        finally:
            conn.close()

    try:
        rows = _run_with_timeout(_run, timeout_secs)
    except _QueryTimeoutError:
        return False, None
    except Exception:
        return False, None

    if not rows:
        return (True, []) if allow_empty else (False, [])
    return True, rows


def _normalize_row(row):
    """Normalize a result row for order-insensitive comparison.
    Converts all numeric types to float to avoid int/float mismatches
    (e.g. COUNT(*) returns int, AVG() returns float for the same value).
    """
    return tuple(float(v) if isinstance(v, (int, float)) else v for v in row)


def queries_produce_same_result(q1: str, q2: str, db_path: str,
                                timeout_secs: float = 5.0) -> bool:
    """
    Return True iff *q1* and *q2* produce identical result sets (order-insensitive)
    against the database at *db_path*.

    Used in the corruption pipeline: a corruption is only kept when this returns
    False — meaning the mutation actually changed the query's answer.
    """
    ok1, r1 = execute_query(q1, db_path, timeout_secs)
    ok2, r2 = execute_query(q2, db_path, timeout_secs)

    if not ok1 or not ok2:
        return False

    return sorted(_normalize_row(r) for r in r1) == sorted(_normalize_row(r) for r in r2)


def result_set_similarity(rows_a: list, rows_b: list) -> float:
    """
    Multiset Jaccard similarity between two result sets (0..1).
    Uses normalized rows for numeric comparison.
    """
    if not rows_a and not rows_b:
        return 1.0
    if not rows_a or not rows_b:
        return 0.0

    from collections import Counter

    ca = Counter(_normalize_row(r) for r in rows_a)
    cb = Counter(_normalize_row(r) for r in rows_b)
    intersection = sum((ca & cb).values())
    union = sum((ca | cb).values())
    return intersection / union if union else 1.0


def query_result_similarity(q_gold: str, q_corr: str, db_path: str,
                            timeout_secs: float = 5.0) -> float:
    """Jaccard similarity between gold and corrupted query result sets."""
    ok_g, rg = execute_query(q_gold, db_path, timeout_secs, allow_empty=True)
    ok_c, rc = execute_query(q_corr, db_path, timeout_secs, allow_empty=True)
    if not ok_g or not ok_c:
        return 0.0
    return result_set_similarity(rg, rc)
