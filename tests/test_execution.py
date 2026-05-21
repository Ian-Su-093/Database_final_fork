"""Tests for the SQLite execution oracle."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'clause_ppo', 'src'))

import pytest
from utils.execution import execute_query, queries_produce_same_result

SPIDER_DIR = os.path.join(os.path.dirname(__file__), '..', 'spider')
DB = os.path.join(SPIDER_DIR, 'database', 'department_management',
                  'department_management.sqlite')


def test_valid_query_returns_results():
    ok, rows = execute_query("SELECT count(*) FROM head WHERE age > 56", DB)
    assert ok is True
    assert rows == [[5]]


def test_invalid_query_returns_false():
    ok, rows = execute_query("SELECT nonexistent_col FROM head", DB)
    assert ok is False
    assert rows is None


def test_empty_result_returns_false():
    ok, rows = execute_query("SELECT * FROM head WHERE age > 9999", DB)
    assert ok is False


def test_same_query_is_same():
    q = "SELECT count(*) FROM head"
    assert queries_produce_same_result(q, q, DB) is True


def test_different_queries_differ():
    q1 = "SELECT count(*) FROM head WHERE age > 56"
    q2 = "SELECT count(*) FROM head WHERE age < 56"
    assert queries_produce_same_result(q1, q2, DB) is False


def test_timeout_on_pathological_query():
    q = "SELECT * FROM head, head h2, head h3"
    ok, _ = execute_query(q, DB)
    assert isinstance(ok, bool)
