"""Tests for clause_splitter.py"""
import os, sys, json
import pytest
from data.clause_splitter import split_into_clauses, clauses_to_prefix_states, schema_to_string

SPIDER_DIR = os.path.join(os.path.dirname(__file__), '..', 'spider')


def _tables():
    with open(os.path.join(SPIDER_DIR, 'tables.json')) as f:
        return {t['db_id']: t for t in json.load(f)}


def _train():
    with open(os.path.join(SPIDER_DIR, 'train_spider.json')) as f:
        return json.load(f)


TABLES = _tables()
TRAIN  = _train()


def test_split_execution_order():
    """Clauses must come out in execution order: FROM before WHERE before SELECT."""
    # ex[0]: SELECT count(*) FROM head WHERE age > 56
    clauses = split_into_clauses(TRAIN[0]['sql'])
    names = [c[0] for c in clauses]
    assert names == ['from', 'where', 'select']


def test_split_group_having():
    # ex[7]: FROM head GROUP BY born_state HAVING count(*) >= 3
    clauses = split_into_clauses(TRAIN[7]['sql'])
    names = [c[0] for c in clauses]
    assert 'groupBy' in names
    assert 'having' in names
    assert names.index('groupBy') < names.index('having')
    assert names.index('having') < names.index('select')


def test_split_omits_empty():
    """Empty clauses must not appear in output."""
    for ex in TRAIN[:50]:
        names = [c[0] for c in split_into_clauses(ex['sql'])]
        sql = ex['sql']
        if not sql.get('where'):
            assert 'where' not in names
        if not sql.get('groupBy'):
            assert 'groupBy' not in names


def test_prefix_states_count():
    ex = TRAIN[0]
    clauses = split_into_clauses(ex['sql'])
    schema  = schema_to_string(ex['db_id'], TABLES)
    states  = clauses_to_prefix_states(ex['question'], schema, clauses, ex['query'])
    assert len(states) == len(clauses)


def test_prefix_states_fields():
    ex = TRAIN[7]
    clauses = split_into_clauses(ex['sql'])
    schema  = schema_to_string(ex['db_id'], TABLES)
    states  = clauses_to_prefix_states(ex['question'], schema, clauses, ex['query'])
    for j, s in enumerate(states):
        assert s['clause_position'] == j
        assert s['total_clauses']   == len(clauses)
        assert s['question'] == ex['question']
        assert len(s['prefix_clauses']) == j + 1


def test_schema_to_string_contains_tables():
    s = schema_to_string('department_management', TABLES)
    assert 'department' in s.lower()
    assert 'head' in s.lower()
    assert 'management' in s.lower()
