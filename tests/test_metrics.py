"""Tests for evaluation metrics."""
import json
import os
import pytest

from eval.metrics import (
    execution_accuracy,
    partial_match,
    _split_sql_clauses,
    _token_f1,
    CLAUSE_KEYWORDS,
)

SPIDER_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'clause_ppo', 'data', 'spider'
)


def _train():
    with open(os.path.join(SPIDER_DIR, 'train_spider.json')) as f:
        return json.load(f)


_HAS_SPIDER = os.path.exists(os.path.join(SPIDER_DIR, 'tables.json'))
TRAIN = _train() if _HAS_SPIDER else []


# ── _split_sql_clauses (no Spider data required) ───────────────────────────

def test_split_simple_select():
    parts = _split_sql_clauses("SELECT count(*) FROM head WHERE age > 56")
    assert parts['SELECT'] == 'count(*)'
    assert parts['FROM']   == 'head'
    assert parts['WHERE']  == 'age > 56'
    assert parts['GROUP BY'] == ''


def test_split_group_having():
    parts = _split_sql_clauses(
        "SELECT born_state FROM head GROUP BY born_state HAVING count(*) >= 3"
    )
    assert parts['SELECT']   == 'born_state'
    assert parts['FROM']     == 'head'
    assert parts['GROUP BY'] == 'born_state'
    assert parts['HAVING']   == 'count(*) >= 3'


def test_split_order_by_limit():
    parts = _split_sql_clauses("SELECT name FROM head ORDER BY age DESC LIMIT 3")
    assert parts['ORDER BY'] == 'age DESC'
    assert parts['LIMIT']    == '3'


def test_split_case_insensitive():
    parts = _split_sql_clauses("select name from head where id = 1")
    assert parts['SELECT'] == 'name'
    assert parts['FROM']   == 'head'
    assert parts['WHERE']  == 'id = 1'


def test_split_missing_clauses_are_empty():
    parts = _split_sql_clauses("SELECT * FROM t")
    assert parts['WHERE']    == ''
    assert parts['GROUP BY'] == ''
    assert parts['HAVING']   == ''


def test_split_keyword_set_complete():
    parts = _split_sql_clauses("SELECT * FROM t")
    assert set(parts.keys()) == set(CLAUSE_KEYWORDS)


# ── _token_f1 ──────────────────────────────────────────────────────────────

def test_token_f1_identical():
    assert _token_f1("count ( * )", "count ( * )") == 1.0


def test_token_f1_disjoint():
    assert _token_f1("a b c", "d e f") == 0.0


def test_token_f1_partial():
    # pred = {a, b}, gold = {a, c} → common=1, P=R=0.5, F1=0.5
    assert _token_f1("a b", "a c") == pytest.approx(0.5)


def test_token_f1_both_empty():
    assert _token_f1("", "") == 1.0


def test_token_f1_one_empty():
    assert _token_f1("a b", "") == 0.0
    assert _token_f1("", "a b") == 0.0


def test_token_f1_case_insensitive():
    assert _token_f1("AGE > 56", "age > 56") == 1.0


# ── partial_match (no execution needed) ────────────────────────────────────

def test_partial_match_identical_pairs():
    preds = [
        "SELECT count(*) FROM head WHERE age > 56",
        "SELECT name FROM singer ORDER BY age DESC",
    ]
    samples = [
        {'db_id': 'x', 'gold_sql': preds[0]},
        {'db_id': 'x', 'gold_sql': preds[1]},
    ]
    f1 = partial_match(preds, samples)
    assert f1['SELECT'] == 1.0
    assert f1['FROM']   == 1.0
    # WHERE only appears in pred[0]; ORDER BY only in pred[1] — still 1.0 each
    assert f1['WHERE']    == 1.0
    assert f1['ORDER BY'] == 1.0


def test_partial_match_only_where_differs():
    preds   = ["SELECT * FROM t WHERE a = 1"]
    samples = [{'db_id': 'x', 'gold_sql': "SELECT * FROM t WHERE a = 2"}]
    f1 = partial_match(preds, samples)
    assert f1['SELECT'] == 1.0
    assert f1['FROM']   == 1.0
    # "a = 1" vs "a = 2" — share {a, =}, differ on {1, 2}
    assert 0.0 < f1['WHERE'] < 1.0


def test_partial_match_returns_all_clauses():
    f1 = partial_match(["SELECT * FROM t"], [{'db_id': 'x', 'gold_sql': "SELECT * FROM t"}])
    assert set(f1.keys()) == set(CLAUSE_KEYWORDS)


def test_partial_match_length_mismatch_raises():
    with pytest.raises(ValueError):
        partial_match(["a"], [])


# ── execution_accuracy (requires Spider data) ──────────────────────────────

pytestmark = pytest.mark.skipif(
    not _HAS_SPIDER,
    reason="Spider dataset not found at clause_ppo/data/spider/"
)


def test_execution_accuracy_all_correct():
    samples = TRAIN[:3]
    preds   = [s['query'] for s in samples]
    acc = execution_accuracy(preds, samples, spider_dir=SPIDER_DIR)
    assert acc == 1.0


def test_execution_accuracy_all_wrong():
    # ex[0]: flip the predicate → different result
    sample = TRAIN[0]
    wrong  = "SELECT count(*) FROM head WHERE age < 0"
    acc = execution_accuracy([wrong], [sample], spider_dir=SPIDER_DIR)
    assert acc == 0.0


def test_execution_accuracy_mixed():
    samples = [TRAIN[0], TRAIN[0]]
    preds   = [TRAIN[0]['query'], "SELECT count(*) FROM head WHERE age < 0"]
    acc = execution_accuracy(preds, samples, spider_dir=SPIDER_DIR)
    assert acc == 0.5


def test_execution_accuracy_empty():
    assert execution_accuracy([], [], spider_dir=SPIDER_DIR) == 0.0


def test_execution_accuracy_length_mismatch_raises():
    with pytest.raises(ValueError):
        execution_accuracy(["a"], [], spider_dir=SPIDER_DIR)


def test_execution_accuracy_accepts_gold_sql_field():
    # Ian's loader emits `gold_sql`; metric must accept either.
    sample = dict(TRAIN[0])
    sample['gold_sql'] = sample.pop('query')
    acc = execution_accuracy([sample['gold_sql']], [sample], spider_dir=SPIDER_DIR)
    assert acc == 1.0
