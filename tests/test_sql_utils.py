"""Tests for SQL reconstruction from Spider parsed sql dicts."""
import os, sys, json
import pytest
from utils.sql_utils import reconstruct_sql
from utils.execution import execute_query

SPIDER_DIR = os.path.join(os.path.dirname(__file__), '..', 'spider')


def _load_db(db_id):
    return os.path.join(SPIDER_DIR, 'database', db_id, f'{db_id}.sqlite')


def _load_tables():
    with open(os.path.join(SPIDER_DIR, 'tables.json')) as f:
        return {t['db_id']: t for t in json.load(f)}


def _load_train():
    with open(os.path.join(SPIDER_DIR, 'train_spider.json')) as f:
        return json.load(f)


TABLES = _load_tables()
TRAIN  = _load_train()


def test_simple_select_count():
    # ex[0]: "SELECT count(*) FROM head WHERE age > 56" -> should return [[5]]
    ex = TRAIN[0]
    sql_str = reconstruct_sql(ex['sql'], TABLES[ex['db_id']])
    ok, rows = execute_query(sql_str, _load_db(ex['db_id']))
    assert ok is True
    assert rows == [[5]]


def test_group_by_having():
    # ex[7]: "SELECT born_state FROM head GROUP BY born_state HAVING count(*) >= 3"
    ex = TRAIN[7]
    sql_str = reconstruct_sql(ex['sql'], TABLES[ex['db_id']])
    ok, rows = execute_query(sql_str, _load_db(ex['db_id']))
    assert ok is True


def test_reconstructed_executes_first_20():
    """Reconstructed SQL must be executable for the first 20 train examples."""
    failures = []
    for ex in TRAIN[:20]:
        sql_str = reconstruct_sql(ex['sql'], TABLES[ex['db_id']])
        ok, _ = execute_query(sql_str, _load_db(ex['db_id']))
        if not ok:
            failures.append((ex['db_id'], ex['query'], sql_str))
    assert len(failures) == 0, f"Reconstruction failed for: {failures[:3]}"
