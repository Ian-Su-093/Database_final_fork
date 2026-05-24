"""Tests for NL2SQLEnv."""
import json
import os
import pytest

from env.env import NL2SQLEnv

SPIDER_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'clause_ppo', 'data', 'spider'
)

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(SPIDER_DIR, 'tables.json')),
    reason="Spider dataset not found at clause_ppo/data/spider/"
)


def _tables():
    with open(os.path.join(SPIDER_DIR, 'tables.json')) as f:
        return {t['db_id']: t for t in json.load(f)}


def _train():
    with open(os.path.join(SPIDER_DIR, 'train_spider.json')) as f:
        return json.load(f)


TABLES = _tables()
TRAIN  = _train()


@pytest.fixture
def env():
    return NL2SQLEnv(spider_dir=SPIDER_DIR, tables=TABLES)


def test_reset_returns_required_keys(env):
    state = env.reset(TRAIN[0])
    assert set(state.keys()) == {'question', 'schema', 'db_id'}
    assert state['question'] == TRAIN[0]['question']
    assert state['db_id']    == TRAIN[0]['db_id']


def test_reset_schema_contains_tables(env):
    # ex[0] uses department_management → tables: department, head, management
    state = env.reset(TRAIN[0])
    assert 'head'       in state['schema'].lower()
    assert 'department' in state['schema'].lower()


def test_step_correct_sql_returns_positive(env):
    # ex[0]: gold = "SELECT count(*) FROM head WHERE age > 56"
    env.reset(TRAIN[0])
    reward, done = env.step(TRAIN[0]['query'])
    assert reward == 1.0
    assert done is True


def test_step_wrong_sql_returns_negative(env):
    env.reset(TRAIN[0])
    # Flip the predicate → different result
    reward, done = env.step("SELECT count(*) FROM head WHERE age < 56")
    assert reward == -1.0
    assert done is True


def test_step_malformed_sql_returns_negative(env):
    env.reset(TRAIN[0])
    reward, done = env.step("SELECT nonexistent FROM nowhere")
    assert reward == -1.0
    assert done is True


def test_step_before_reset_raises():
    env = NL2SQLEnv(spider_dir=SPIDER_DIR, tables=TABLES)
    with pytest.raises(RuntimeError):
        env.step("SELECT 1")


def test_step_gold_sql_field_supported(env):
    # Ian's loader returns `gold_sql` not raw `query` — env must accept both.
    sample = dict(TRAIN[0])
    sample['gold_sql'] = sample.pop('query')
    env.reset(sample)
    reward, done = env.step(sample['gold_sql'])
    assert reward == 1.0
    assert done is True


def test_get_faulty_clause_argmin(env):
    scores = {'SELECT': 0.91, 'FROM': 0.88, 'WHERE': 0.21, 'GROUP BY': 0.55}
    assert env.get_faulty_clause(scores) == 'WHERE'


def test_get_faulty_clause_single(env):
    assert env.get_faulty_clause({'SELECT': 0.5}) == 'SELECT'


def test_get_faulty_clause_empty_raises(env):
    with pytest.raises(ValueError):
        env.get_faulty_clause({})


def test_episode_is_independent(env):
    # Resetting must clear the previous sample so step() uses the new gold.
    env.reset(TRAIN[0])  # department_management
    env.reset(TRAIN[7])  # different db
    reward, done = env.step(TRAIN[7]['query'])
    assert reward == 1.0
    assert done is True
