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
    assert rows == []


def test_same_query_is_same():
    q = "SELECT count(*) FROM head"
    assert queries_produce_same_result(q, q, DB) is True


def test_different_queries_differ():
    q1 = "SELECT count(*) FROM head WHERE age > 56"
    q2 = "SELECT count(*) FROM head WHERE age < 56"
    assert queries_produce_same_result(q1, q2, DB) is False


def test_timeout_fires():
    # WITH RECURSIVE runs forever — will hit the timeout
    q = ("WITH RECURSIVE r(x) AS "
         "(SELECT 1 UNION ALL SELECT x+1 FROM r) "
         "SELECT x FROM r LIMIT 1000000000")
    ok, rows = execute_query(q, DB, timeout_secs=0.1)
    assert ok is False
    assert rows is None


def test_timeout_does_not_fire_on_normal_query():
    ok, _ = execute_query("SELECT count(*) FROM head", DB, timeout_secs=5.0)
    assert ok is True


def test_comparison_returns_false_if_one_query_fails():
    good_q = "SELECT count(*) FROM head"
    bad_q  = "SELECT nonexistent FROM head"
    assert queries_produce_same_result(good_q, bad_q, DB) is False
