"""Tests for clause_ppo/scripts/clause_rewards.py"""

import json
import os
import sys

import pytest

_CLAUSE_PPO = os.path.join(os.path.dirname(__file__), '..', 'clause_ppo')
sys.path.insert(0, os.path.join(_CLAUSE_PPO, 'src'))
sys.path.insert(0, os.path.join(_CLAUSE_PPO, 'scripts'))

from clause_rewards import (
    build_orig_clause_texts,
    compute_clause_rewards,
    extract_clause_texts,
    extract_clauses_ordered,
    is_syntax_invalid_clause,
    score_clause_pair,
    validate_rewards,
)

PROCESSED_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'clause_ppo', 'data', 'processed'
)
SPIDER_DB = os.path.join(
    os.path.dirname(__file__), '..', 'clause_ppo', 'data', 'spider',
    'database', 'department_management', 'department_management.sqlite'
)


def _load_corruptions():
    with open(os.path.join(PROCESSED_DIR, 'corruption_dataset.json')) as f:
        return json.load(f)


def _load_clause_lookup():
    with open(os.path.join(PROCESSED_DIR, 'original_dataset.json')) as f:
        originals = json.load(f)
    return {(ex['db_id'], ex['question']): ex['clause_names'] for ex in originals}


@pytest.fixture
def spider_db():
    if not os.path.isfile(SPIDER_DB):
        pytest.skip('Spider SQLite database not available')
    return SPIDER_DB


class TestExtractClauseTexts:
    def test_three_clause_query_execution_order(self):
        sql = 'SELECT COUNT(*) FROM head WHERE age > 56.0'
        ordered = extract_clauses_ordered(sql)
        names = [n for n, _ in ordered]
        assert names == ['from', 'where', 'select']

    def test_order_by_query(self):
        sql = 'SELECT name, born_state, age FROM head ORDER BY age ASC'
        ordered = extract_clauses_ordered(sql)
        names = [n for n, _ in ordered]
        assert names == ['from', 'select', 'orderBy']


class TestScoreClausePair:
    def test_unchanged_clause(self):
        text = 'FROM head'
        assert score_clause_pair('from', text, text) == 0.5

    def test_syntax_invalid_not_gt(self):
        assert is_syntax_invalid_clause(
            'where', 'WHERE age NOT > 56.0'
        )
        assert score_clause_pair(
            'where', 'WHERE age > 56.0', 'WHERE age NOT > 56.0'
        ) == 0.0

    def test_syntax_invalid_max_star(self):
        assert is_syntax_invalid_clause(
            'select', 'SELECT MAX(*) FROM head'
        )
        assert score_clause_pair(
            'select', 'SELECT COUNT(*)', 'SELECT MAX(*)'
        ) == 0.0

    def test_wrong_table(self):
        s = score_clause_pair(
            'from',
            'FROM head',
            'FROM department',
            strategy='corrupt_from',
        )
        assert 0.05 <= s <= 0.5

    def test_dropped_column(self):
        s = score_clause_pair(
            'select',
            'SELECT name, born_state, age',
            'SELECT born_state, age',
            strategy='corrupt_select',
        )
        assert s == 0.25


class TestComputeRewardsNoDb:
    """Integration tests that do not require Spider SQLite files."""

    def test_records_1_and_3_without_db(self):
        corruptions = _load_corruptions()
        lookup = _load_clause_lookup()
        for idx in (1, 2):
            record = corruptions[idx]
            clause_names = lookup[(record['db_id'], record['question'])]
            orig_texts = build_orig_clause_texts(record['original_query'], clause_names)
            rewards = compute_clause_rewards(
                record, clause_names, orig_texts, '/nonexistent/db.sqlite'
            )
            if idx == 1:
                assert rewards == [1.0, 0.0, 0.0]
            else:
                assert rewards == [1.0, 1.0, 0.0]


class TestComputeRewardsFirstFive:
    @pytest.fixture
    def setup(self):
        corruptions = _load_corruptions()
        lookup = _load_clause_lookup()
        return corruptions[:5], lookup

    @pytest.mark.parametrize('idx,check', [
        (0, lambda r: r[0] <= 0.5 and all(x <= 0.5 for x in r)),
        (1, lambda r: r[0] == 1.0 and r[1] == 0.0),
        (2, lambda r: r[0] == 1.0 and r[1] == 1.0 and r[2] == 0.0),
        (3, lambda r: r[0] <= 0.5 and all(x <= 0.5 for x in r)),
        (4, lambda r: r[0] == 1.0 and r[1] <= 0.5 and r[1] > 0),
    ])
    def test_first_five_invariants(self, setup, spider_db, idx, check):
        corruptions, lookup = setup
        record = corruptions[idx]
        clause_names = lookup[(record['db_id'], record['question'])]
        orig_texts = build_orig_clause_texts(record['original_query'], clause_names)
        rewards = compute_clause_rewards(record, clause_names, orig_texts, spider_db)
        assert len(rewards) == len(clause_names)
        assert check(rewards)
        assert not validate_rewards(rewards, clause_names, record['corrupted_clause'])

    def test_record1_exact_prefix(self, setup, spider_db):
        corruptions, lookup = setup
        record = corruptions[1]
        clause_names = lookup[(record['db_id'], record['question'])]
        orig_texts = build_orig_clause_texts(record['original_query'], clause_names)
        rewards = compute_clause_rewards(record, clause_names, orig_texts, spider_db)
        assert rewards[0] == 1.0
        assert rewards[1] == 0.0

    def test_record3_exact_syntax(self, setup, spider_db):
        corruptions, lookup = setup
        record = corruptions[2]
        clause_names = lookup[(record['db_id'], record['question'])]
        orig_texts = build_orig_clause_texts(record['original_query'], clause_names)
        rewards = compute_clause_rewards(record, clause_names, orig_texts, spider_db)
        assert rewards == [1.0, 1.0, 0.0]
