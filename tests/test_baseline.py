"""Tests for the full-regen baseline scaffold (src/baseline/full_regen.py).

Uses lightweight fakes for the HF model/tokenizer and the env, so the tests
run without torch/transformers and without the Spider DB. Heavier integration
testing happens in scripts/evaluate.py against the real backbone.
"""
import pytest

from baseline.full_regen import build_baseline_prompt, run_baseline


# ── Fakes ──────────────────────────────────────────────────────────────────

class FakeIds:
    """
    Minimal stand-in for a torch tensor of token IDs.

    Tracks length only — that's all _generate_sql() inspects (.shape[-1],
    .to(device), integer-index and slicing into a 1-D segment).
    """
    def __init__(self, length: int, is_batched: bool = True):
        self._length     = length
        self._is_batched = is_batched

    @property
    def shape(self):
        return (1, self._length) if self._is_batched else (self._length,)

    def to(self, device):
        return self

    def __getitem__(self, idx):
        if isinstance(idx, int):                      # output[0] strips batch
            return FakeIds(self._length, is_batched=False)
        if isinstance(idx, slice):                    # output[0][input_len:]
            start = idx.start or 0
            stop  = idx.stop if idx.stop is not None else self._length
            return FakeIds(max(0, stop - start), is_batched=self._is_batched)
        raise TypeError(f"Unsupported index: {idx!r}")


class FakeBackbone:
    """
    Serves as both model and tokenizer (Python is duck-typed, so we can
    pass the same object as both arguments to run_baseline).

    sql_outputs:   one decoded string per attempt, consumed in order.
    input_tokens:  pretended input prompt length.
    output_tokens: pretended generated-tokens length.
    """
    def __init__(
        self,
        sql_outputs:   list[str],
        input_tokens:  int = 10,
        output_tokens: int = 8,
    ):
        self._sql_outputs   = list(sql_outputs)
        self._call_idx      = 0
        self._next_sql      = ''
        self._input_tokens  = input_tokens
        self._output_tokens = output_tokens
        self.pad_token_id   = 0
        self.eos_token_id   = 0
        self.device         = None

    # ── tokenizer interface ──
    def encode(self, text, return_tensors='pt'):
        return FakeIds(self._input_tokens)

    def decode(self, ids, skip_special_tokens=False):
        return self._next_sql

    # ── model interface ──
    def generate(self, input_ids, **kwargs):
        self._next_sql = self._sql_outputs[self._call_idx]
        self._call_idx += 1
        return FakeIds(self._input_tokens + self._output_tokens)


class FakeEnv:
    """Reward = +1 when predicted SQL matches the configured gold, else -1."""
    def __init__(self, gold_sql: str = 'SELECT 1'):
        self._gold = gold_sql

    def reset(self, sample):
        return {'question': 'Q', 'schema': 'S', 'db_id': 'D'}

    def step(self, sql):
        return (1.0 if sql.strip() == self._gold.strip() else -1.0, True)


# ── build_baseline_prompt (no fakes needed) ────────────────────────────────

def test_build_baseline_prompt_contains_markers():
    prompt = build_baseline_prompt("how many heads?", "head(name, age)")
    assert '[QUESTION] how many heads?' in prompt
    assert '[SCHEMA] head(name, age)'   in prompt
    assert prompt.endswith('[SQL]')


def test_build_baseline_prompt_mirrors_rewrite_marker_order():
    # QUESTION before SCHEMA before SQL — matches build_rewrite_prompt() so
    # input-token counts are comparable between baseline and PPO actor.
    prompt = build_baseline_prompt("q", "s")
    assert prompt.index('[QUESTION]') < prompt.index('[SCHEMA]') < prompt.index('[SQL]')


# ── run_baseline ───────────────────────────────────────────────────────────

def test_run_baseline_returns_expected_dict_shape():
    fake = FakeBackbone(['SELECT 1'])
    env  = FakeEnv(gold_sql='SELECT 1')
    out  = run_baseline({}, fake, fake, max_retries=3, env=env)
    assert set(out.keys()) == {'predicted_sql', 'token_cost', 'attempts'}
    assert isinstance(out['predicted_sql'], str)
    assert isinstance(out['token_cost'],    int)
    assert isinstance(out['attempts'],      int)


def test_run_baseline_stops_on_first_correct_attempt():
    fake = FakeBackbone(['SELECT 1', 'SELECT 2', 'SELECT 3'])
    env  = FakeEnv(gold_sql='SELECT 1')
    out  = run_baseline({}, fake, fake, max_retries=3, env=env)
    assert out['attempts']      == 1
    assert out['predicted_sql'] == 'SELECT 1'


def test_run_baseline_retries_up_to_max_when_always_wrong():
    fake = FakeBackbone(['WRONG_A', 'WRONG_B', 'WRONG_C'])
    env  = FakeEnv(gold_sql='SELECT 1')
    out  = run_baseline({}, fake, fake, max_retries=3, env=env)
    assert out['attempts']      == 3
    # predicted_sql is the LAST attempt, not the first
    assert out['predicted_sql'] == 'WRONG_C'


def test_run_baseline_succeeds_on_last_attempt():
    fake = FakeBackbone(['WRONG_A', 'WRONG_B', 'SELECT 1'])
    env  = FakeEnv(gold_sql='SELECT 1')
    out  = run_baseline({}, fake, fake, max_retries=3, env=env)
    assert out['attempts']      == 3
    assert out['predicted_sql'] == 'SELECT 1'


def test_run_baseline_max_retries_one():
    fake = FakeBackbone(['WRONG_ONLY'])
    env  = FakeEnv(gold_sql='SELECT 1')
    out  = run_baseline({}, fake, fake, max_retries=1, env=env)
    assert out['attempts']      == 1
    assert out['predicted_sql'] == 'WRONG_ONLY'


def test_run_baseline_token_cost_accumulates_across_attempts():
    fake = FakeBackbone(
        ['WRONG_A', 'WRONG_B', 'SELECT 1'],
        input_tokens=10, output_tokens=8,
    )
    env = FakeEnv(gold_sql='SELECT 1')
    out = run_baseline({}, fake, fake, max_retries=3, env=env)
    # 3 attempts × (10 input + 8 output) = 54
    assert out['attempts']   == 3
    assert out['token_cost'] == 54


def test_run_baseline_token_cost_single_attempt():
    fake = FakeBackbone(['SELECT 1'], input_tokens=12, output_tokens=5)
    env  = FakeEnv(gold_sql='SELECT 1')
    out  = run_baseline({}, fake, fake, max_retries=3, env=env)
    assert out['attempts']   == 1
    assert out['token_cost'] == 17
