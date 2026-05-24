"""Tests for the rule-based SQL corruption engine."""
import os, sys, json, copy
import pytest
from data.corruption import (
    corrupt_select, corrupt_from, corrupt_where,
    corrupt_group_by, corrupt_having, corrupt_order_by,
    generate_corruptions,
)
from utils.execution import queries_produce_same_result
from utils.sql_utils import reconstruct_sql

SPIDER_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'clause_ppo', 'data', 'spider'
)

def _load():
    with open(os.path.join(SPIDER_DIR, 'tables.json')) as f:
        tables = {t['db_id']: t for t in json.load(f)}
    with open(os.path.join(SPIDER_DIR, 'train_spider.json')) as f:
        train = json.load(f)
    return tables, train


TABLES, TRAIN = _load()


def _db_path(db_id):
    return os.path.join(SPIDER_DIR, 'database', db_id, f'{db_id}.sqlite')


def test_corrupt_does_not_modify_original():
    """Corruption must return a NEW dict; original sql_dict must be unchanged."""
    ex = TRAIN[0]
    original = copy.deepcopy(ex['sql'])
    corrupt_select(ex['sql'], TABLES[ex['db_id']])
    assert ex['sql'] == original


def test_corrupt_select_returns_dict_or_none():
    for ex in TRAIN[:20]:
        result = corrupt_select(ex['sql'], TABLES[ex['db_id']])
        assert result is None or isinstance(result, dict)


def test_corrupt_where_on_query_without_where_returns_none():
    # ex[2] has no WHERE clause
    ex = TRAIN[2]
    assert ex['sql']['where'] == []
    result = corrupt_where(ex['sql'], TABLES[ex['db_id']])
    assert result is None


def test_generate_corruptions_verified():
    """All returned corruptions must actually change query results."""
    for ex in TRAIN[:30]:
        db_id = ex['db_id']
        db_path = _db_path(db_id)
        corruptions = generate_corruptions(ex, TABLES)
        for c in corruptions:
            same = queries_produce_same_result(
                c['original_query'], c['corrupted_query'], db_path
            )
            assert same is False, (
                f"Corruption did not change results:\n"
                f"  orig: {c['original_query']}\n"
                f"  corr: {c['corrupted_query']}"
            )


def test_generate_corruptions_fields():
    ex = TRAIN[7]   # has FROM+GROUPBY+HAVING+SELECT
    corruptions = generate_corruptions(ex, TABLES)
    for c in corruptions:
        for field in ('db_id', 'question', 'original_query', 'corrupted_query',
                      'corrupted_clause', 'corrupted_position', 'corruption_strategy'):
            assert field in c, f"Missing field '{field}' in corruption record"
