"""Tests for the full-regen baseline (src/baseline/full_regen.py).

Generation is injected as a generate_fn, so these run without torch /
transformers / huggingface_hub and without the Spider DB (env is faked too).
"""
import pytest

from baseline.full_regen import (
    build_baseline_prompt,
    make_hf_api_generate_fn,
    run_baseline,
    _extract_sql,
    _usage_tokens,
)


# ── Fakes ──────────────────────────────────────────────────────────────────

def make_fake_generate_fn(outputs: list[tuple[str, int, int]]):
    """Return a generate_fn that yields (sql, n_in, n_out) tuples in order."""
    state = {'i': 0}

    def generate_fn(prompt: str):
        out = outputs[state['i']]
        state['i'] += 1
        return out

    return generate_fn


class FakeEnv:
    """Reward = +1 when predicted SQL matches the configured gold, else -1."""
    def __init__(self, gold_sql: str = 'SELECT 1'):
        self._gold = gold_sql

    def reset(self, sample):
        return {'question': 'Q', 'schema': 'S', 'db_id': 'D'}

    def step(self, sql):
        return (1.0 if sql.strip() == self._gold.strip() else -1.0, True)


# ── InferenceClient fakes (for make_hf_api_generate_fn) ────────────────────

class _Usage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens     = prompt_tokens
        self.completion_tokens = completion_tokens


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Completion:
    def __init__(self, content, usage=None):
        self.choices = [_Choice(content)]
        self.usage   = usage


class _Completions:
    def __init__(self, completions: list):
        self._completions = completions
        self._i           = 0
        self.calls        = []

    def create(self, model, messages, max_tokens):
        self.calls.append({'model': model, 'messages': messages, 'max_tokens': max_tokens})
        c = self._completions[self._i]
        self._i += 1
        return c


class FakeClient:
    def __init__(self, completions: list):
        self.chat = type('Chat', (), {'completions': _Completions(completions)})()


class WordTokenizer:
    """Fallback tokenizer stand-in: token count == word count."""
    def encode(self, text):
        return text.split()


# ── build_baseline_prompt ──────────────────────────────────────────────────

def test_build_baseline_prompt_contains_markers():
    prompt = build_baseline_prompt("how many heads?", "head(name, age)")
    assert '[QUESTION] how many heads?' in prompt
    assert '[SCHEMA] head(name, age)'   in prompt
    assert prompt.endswith('[SQL]')


def test_build_baseline_prompt_marker_order():
    prompt = build_baseline_prompt("q", "s")
    assert prompt.index('[QUESTION]') < prompt.index('[SCHEMA]') < prompt.index('[SQL]')


# ── run_baseline ───────────────────────────────────────────────────────────

def test_run_baseline_returns_expected_dict_shape():
    gen = make_fake_generate_fn([('SELECT 1', 10, 8)])
    out = run_baseline({}, gen, max_retries=3, env=FakeEnv('SELECT 1'))
    assert set(out.keys()) == {'predicted_sql', 'token_cost', 'attempts'}
    assert isinstance(out['predicted_sql'], str)
    assert isinstance(out['token_cost'],    int)
    assert isinstance(out['attempts'],      int)


def test_run_baseline_stops_on_first_correct():
    gen = make_fake_generate_fn([('SELECT 1', 10, 8), ('SELECT 2', 10, 8)])
    out = run_baseline({}, gen, max_retries=3, env=FakeEnv('SELECT 1'))
    assert out['attempts']      == 1
    assert out['predicted_sql'] == 'SELECT 1'
    assert out['token_cost']    == 18


def test_run_baseline_retries_up_to_max_when_always_wrong():
    gen = make_fake_generate_fn([('A', 10, 8), ('B', 10, 8), ('C', 10, 8)])
    out = run_baseline({}, gen, max_retries=3, env=FakeEnv('SELECT 1'))
    assert out['attempts']      == 3
    assert out['predicted_sql'] == 'C'        # last attempt, not first


def test_run_baseline_succeeds_on_last_attempt():
    gen = make_fake_generate_fn([('A', 10, 8), ('B', 10, 8), ('SELECT 1', 10, 8)])
    out = run_baseline({}, gen, max_retries=3, env=FakeEnv('SELECT 1'))
    assert out['attempts']      == 3
    assert out['predicted_sql'] == 'SELECT 1'


def test_run_baseline_max_retries_one():
    gen = make_fake_generate_fn([('WRONG', 10, 8)])
    out = run_baseline({}, gen, max_retries=1, env=FakeEnv('SELECT 1'))
    assert out['attempts']      == 1
    assert out['predicted_sql'] == 'WRONG'


def test_run_baseline_token_cost_accumulates():
    gen = make_fake_generate_fn([('A', 10, 8), ('B', 11, 9), ('SELECT 1', 12, 10)])
    out = run_baseline({}, gen, max_retries=3, env=FakeEnv('SELECT 1'))
    # (10+8) + (11+9) + (12+10) = 60
    assert out['attempts']   == 3
    assert out['token_cost'] == 60


# ── make_hf_api_generate_fn ─────────────────────────────────────────────────

def test_api_adapter_uses_server_usage_stats():
    client = FakeClient([_Completion('SELECT 1', usage=_Usage(40, 7))])
    gen    = make_hf_api_generate_fn(client, model='qwen/qwen2.5-coder-1.5b')
    sql, n_in, n_out = gen("[QUESTION] q [SCHEMA] s [SQL]")
    assert sql   == 'SELECT 1'
    assert n_in  == 40
    assert n_out == 7


def test_api_adapter_strips_sql_code_fence():
    content = "Here you go:\n```sql\nSELECT count(*) FROM t\n```"
    client  = FakeClient([_Completion(content, usage=_Usage(20, 12))])
    gen     = make_hf_api_generate_fn(client, model='m')
    sql, _, _ = gen("prompt")
    assert sql == 'SELECT count(*) FROM t'


def test_api_adapter_passes_model_and_wraps_prompt():
    completions = _Completions([_Completion('SELECT 1', usage=_Usage(1, 1))])
    client = FakeClient([])
    client.chat.completions = completions
    gen = make_hf_api_generate_fn(client, model='qwen/qwen2.5-coder-1.5b', max_tokens=321)
    gen("MY_PROMPT")
    call = completions.calls[0]
    assert call['model']      == 'qwen/qwen2.5-coder-1.5b'
    assert call['max_tokens'] == 321
    assert call['messages']   == [{'role': 'user', 'content': 'MY_PROMPT'}]


def test_api_adapter_falls_back_to_tokenizer_when_no_usage():
    client = FakeClient([_Completion('SELECT 1 FROM t', usage=None)])
    gen    = make_hf_api_generate_fn(client, model='m', fallback_tokenizer=WordTokenizer())
    sql, n_in, n_out = gen("a b c")     # 3 words in, "SELECT 1 FROM t" → 4 words out
    assert sql   == 'SELECT 1 FROM t'
    assert n_in  == 3
    assert n_out == 4


def test_api_adapter_raises_without_usage_or_fallback():
    client = FakeClient([_Completion('SELECT 1', usage=None)])
    gen    = make_hf_api_generate_fn(client, model='m')
    with pytest.raises(RuntimeError, match="usage"):
        gen("prompt")


# ── _extract_sql ─────────────────────────────────────────────────────────────

def test_extract_sql_plain_passthrough():
    assert _extract_sql("  SELECT 1  ") == 'SELECT 1'


def test_extract_sql_sql_fence():
    assert _extract_sql("```sql\nSELECT 1\n```") == 'SELECT 1'


def test_extract_sql_bare_fence():
    assert _extract_sql("```\nSELECT 1\n```") == 'SELECT 1'


def test_extract_sql_fence_case_insensitive():
    assert _extract_sql("```SQL\nSELECT 1\n```") == 'SELECT 1'


# ── _usage_tokens ────────────────────────────────────────────────────────────

def test_usage_tokens_prefers_server_stats():
    completion = _Completion('out', usage=_Usage(5, 3))
    assert _usage_tokens(completion, 'prompt', 'out', None) == (5, 3)


def test_usage_tokens_fallback_tokenizer():
    completion = _Completion('one two', usage=None)
    n_in, n_out = _usage_tokens(completion, 'a b c d', 'one two', WordTokenizer())
    assert (n_in, n_out) == (4, 2)


def test_usage_tokens_raises_without_either():
    completion = _Completion('out', usage=None)
    with pytest.raises(RuntimeError):
        _usage_tokens(completion, 'prompt', 'out', None)
