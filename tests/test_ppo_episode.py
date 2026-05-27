"""Tests for get_corrupted_sample — requires Spider data at spider/."""
import json
import os
import sys

import pytest

_REPO      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPIDER_DIR = os.path.join(_REPO, 'spider')

for _p in [
    os.path.join(_REPO, 'clause_ppo', 'src'),
    os.path.join(_REPO, 'src'),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(SPIDER_DIR, 'tables.json')),
    reason="Spider dataset not found at spider/",
)


@pytest.fixture(scope="module")
def spider_data():
    with open(os.path.join(SPIDER_DIR, 'tables.json')) as f:
        tables_dict = {t['db_id']: t for t in json.load(f)}
    with open(os.path.join(SPIDER_DIR, 'train_spider.json')) as f:
        samples = json.load(f)
    return tables_dict, samples


def test_get_corrupted_sample_returns_tuple(spider_data):
    from training.ppo_loop import get_corrupted_sample
    tables_dict, samples = spider_data
    # Try up to 20 samples to find one that can be corrupted
    result = None
    for s in samples[:20]:
        result = get_corrupted_sample(s, tables_dict)
        if result is not None:
            break
    assert result is not None, "No corruption found in first 20 samples"
    wrong_sql, faulty_clause = result
    assert isinstance(wrong_sql, str) and len(wrong_sql) > 0
    assert isinstance(faulty_clause, str) and len(faulty_clause) > 0


def test_get_corrupted_sample_faulty_clause_is_valid_key(spider_data):
    from training.ppo_loop import get_corrupted_sample
    from data.clause_splitter import CLAUSE_LABELS
    tables_dict, samples = spider_data
    for s in samples[:20]:
        result = get_corrupted_sample(s, tables_dict)
        if result is not None:
            _, faulty_clause = result
            assert faulty_clause in CLAUSE_LABELS, \
                f"Unexpected clause key: {faulty_clause}"
            return
    pytest.skip("No corruptible sample found in first 20")


def test_get_corrupted_sample_returns_none_for_invalid_sample(spider_data):
    from training.ppo_loop import get_corrupted_sample
    tables_dict, _ = spider_data
    bad_sample = {'db_id': 'nonexistent_db', 'question': 'Q', 'sql': {}}
    result = get_corrupted_sample(bad_sample, tables_dict)
    assert result is None


def test_get_corrupted_sample_wrong_sql_differs_from_gold(spider_data):
    from training.ppo_loop import get_corrupted_sample
    from utils.sql_utils import reconstruct_sql
    tables_dict, samples = spider_data
    for s in samples[:20]:
        result = get_corrupted_sample(s, tables_dict)
        if result is not None:
            wrong_sql, _ = result
            gold_sql = reconstruct_sql(s['sql'], tables_dict[s['db_id']])
            assert wrong_sql != gold_sql, \
                "Corrupted SQL should differ from gold SQL"
            return
    pytest.skip("No corruptible sample found in first 20")
