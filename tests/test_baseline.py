"""Tests for the full-regen baseline (src/baseline/full_regen.py).

Generation is injected as a generate_fn, so these run without torch /
transformers / huggingface_hub and without the Spider DB (env is faked too).
The local-adapter tests inject a fake ``torch`` module into sys.modules.
"""
import sys
import pytest

from baseline.full_regen import (
    build_baseline_prompt,
    make_hf_api_generate_fn,
    make_local_generate_fn,
    run_baseline,
    _apply_chat_template,
    _extract_sql,
    _is_retryable,
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

    def create(self, model, messages, max_tokens, **kwargs):
        self.calls.append({
            'model': model, 'messages': messages,
            'max_tokens': max_tokens, **kwargs,
        })
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


class _Resp:
    def __init__(self, status_code):
        self.status_code = status_code


class FakeHTTPError(Exception):
    """Stands in for HfHubHTTPError — carries .response.status_code."""
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.response = _Resp(status_code)


class ConnectTimeout(Exception):
    """Class name matches _RETRYABLE_EXC_NAMES; has no .response."""


class _ScriptedCompletions:
    """create() walks a script: raise Exceptions, return _Completions."""
    def __init__(self, script: list):
        self._script = list(script)
        self._i      = 0
        self.calls   = 0

    def create(self, model, messages, max_tokens, **kwargs):
        self.calls += 1
        item = self._script[self._i]
        self._i += 1
        if isinstance(item, Exception):
            raise item
        return item


def make_scripted_client(script: list):
    """Return (client, completions) where client.chat.completions follows script."""
    client = FakeClient([])
    comps  = _ScriptedCompletions(script)
    client.chat.completions = comps
    return client, comps


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
    assert set(out.keys()) == {'predicted_sql', 'token_cost', 'attempts', 'success'}
    assert isinstance(out['predicted_sql'], str)
    assert isinstance(out['token_cost'],    int)
    assert isinstance(out['attempts'],      int)
    assert isinstance(out['success'],       bool)


def test_run_baseline_stops_on_first_correct():
    gen = make_fake_generate_fn([('SELECT 1', 10, 8), ('SELECT 2', 10, 8)])
    out = run_baseline({}, gen, max_retries=3, env=FakeEnv('SELECT 1'))
    assert out['attempts']      == 1
    assert out['predicted_sql'] == 'SELECT 1'
    assert out['token_cost']    == 18
    assert out['success']       is True


def test_run_baseline_retries_up_to_max_when_always_wrong():
    gen = make_fake_generate_fn([('A', 10, 8), ('B', 10, 8), ('C', 10, 8)])
    out = run_baseline({}, gen, max_retries=3, env=FakeEnv('SELECT 1'))
    assert out['attempts']      == 3
    assert out['predicted_sql'] == 'C'        # last attempt, not first
    assert out['success']       is False      # never matched gold


def test_run_baseline_succeeds_on_last_attempt():
    gen = make_fake_generate_fn([('A', 10, 8), ('B', 10, 8), ('SELECT 1', 10, 8)])
    out = run_baseline({}, gen, max_retries=3, env=FakeEnv('SELECT 1'))
    assert out['attempts']      == 3
    assert out['predicted_sql'] == 'SELECT 1'
    assert out['success']       is True


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


def test_api_adapter_forwards_temperature():
    # Pin temperature explicitly so Accuracy@N retries are independent samples,
    # not whatever the provider happens to default to.
    completions = _Completions([_Completion('SELECT 1', usage=_Usage(1, 1))])
    client = FakeClient([])
    client.chat.completions = completions
    gen = make_hf_api_generate_fn(client, model='m', temperature=0.42)
    gen("p")
    assert completions.calls[0]['temperature'] == 0.42


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


# ── make_hf_api_generate_fn: transient-error retries ───────────────────────

def test_api_adapter_retries_transient_then_succeeds():
    good = _Completion('SELECT 1', usage=_Usage(5, 3))
    client, comps = make_scripted_client([FakeHTTPError(504), good])
    gen = make_hf_api_generate_fn(client, model='m', api_retries=4, backoff_secs=0)
    sql, n_in, n_out = gen("p")
    assert sql == 'SELECT 1'
    assert comps.calls == 2           # one 504, then success


def test_api_adapter_does_not_retry_non_transient():
    client, comps = make_scripted_client([FakeHTTPError(400)])
    gen = make_hf_api_generate_fn(client, model='m', api_retries=4, backoff_secs=0)
    with pytest.raises(FakeHTTPError):
        gen("p")
    assert comps.calls == 1           # 400 is permanent — no retry


def test_api_adapter_retries_connection_timeout():
    good = _Completion('SELECT 1', usage=_Usage(1, 1))
    client, comps = make_scripted_client([ConnectTimeout("boom"), good])
    gen = make_hf_api_generate_fn(client, model='m', api_retries=4, backoff_secs=0)
    sql, _, _ = gen("p")
    assert sql == 'SELECT 1'
    assert comps.calls == 2


def test_api_adapter_raises_after_exhausting_retries():
    client, comps = make_scripted_client([FakeHTTPError(503)] * 3)
    gen = make_hf_api_generate_fn(client, model='m', api_retries=3, backoff_secs=0)
    with pytest.raises(FakeHTTPError):
        gen("p")
    assert comps.calls == 3           # tried exactly api_retries times


# ── _is_retryable ────────────────────────────────────────────────────────────

def test_is_retryable_5xx_and_429():
    assert _is_retryable(FakeHTTPError(504))
    assert _is_retryable(FakeHTTPError(503))
    assert _is_retryable(FakeHTTPError(429))


def test_is_not_retryable_4xx():
    assert not _is_retryable(FakeHTTPError(400))
    assert not _is_retryable(FakeHTTPError(401))


def test_is_retryable_connection_timeout_by_name():
    assert _is_retryable(ConnectTimeout("x"))


def test_is_not_retryable_unknown_exception():
    assert not _is_retryable(ValueError("a real bug, not a transient API error"))


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


# ── make_local_generate_fn ──────────────────────────────────────────────────

class _FakeNoGrad:
    """torch.no_grad() context-manager stand-in."""
    def __enter__(self): return self
    def __exit__(self, *exc): return False


class _FakeTorchModule:
    """Drop-in for ``import torch`` inside generate_fn."""
    @staticmethod
    def no_grad():
        return _FakeNoGrad()


@pytest.fixture
def fake_torch(monkeypatch):
    """Install a fake torch module so make_local_generate_fn can run."""
    monkeypatch.setitem(sys.modules, 'torch', _FakeTorchModule())
    yield


class _Shape:
    def __init__(self, n): self._n = n
    @property
    def shape(self): return (1, self._n)


class _OutShape:
    """Indexing [0, n:] returns a 1-D tensor stand-in with .shape[0] == len."""
    def __init__(self, ids):
        self._ids = ids

    def __getitem__(self, key):
        # key is (0, slice(n_in, None))
        _, sl = key
        return _Slice(self._ids[sl])


class _Slice:
    def __init__(self, ids):
        self._ids = ids
        self.shape = (len(ids),)


class FakeTensor:
    """Minimal ``inputs['input_ids']``: has .shape == (1, n) and .to() returns self."""
    def __init__(self, n_tokens):
        self._n = n_tokens
        self.shape = (1, n_tokens)


class FakeInputs(dict):
    """Tokenizer output: dict-like with .to() that returns itself."""
    def to(self, device):
        return self


class FakeTokenizer:
    """Token = word; chat-template prepends '<chat>'."""
    eos_token_id = 0

    def __init__(self, chat_template='<chat>'):
        self.chat_template = chat_template
        self._next_decode  = ''

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        # Tests pass tokenize=False, so return a string.
        assert tokenize is False
        return f"{self.chat_template} {messages[0]['content']}"

    def __call__(self, text, return_tensors):
        n = len(text.split())
        return FakeInputs({'input_ids': FakeTensor(n)})

    def decode(self, ids, skip_special_tokens):
        return self._next_decode


class FakeModel:
    """generate() returns prompt prefix + a configured reply (in ids)."""
    device = 'cpu'

    def __init__(self, reply_tokens: int, reply_text: str, tokenizer: FakeTokenizer):
        self._reply_tokens = reply_tokens
        self._reply_text   = reply_text
        self._tok          = tokenizer
        self.last_kwargs   = None

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.last_kwargs = kwargs
        n_in = kwargs['input_ids'].shape[1]
        # ids: 0..n_in-1 = prompt prefix; n_in..n_in+reply_tokens-1 = generated.
        all_ids = list(range(n_in + self._reply_tokens))
        self._tok._next_decode = self._reply_text
        return _OutShape(all_ids)


def test_local_adapter_returns_sql_and_token_counts(fake_torch):
    tok   = FakeTokenizer()
    model = FakeModel(reply_tokens=5, reply_text='SELECT count(*) FROM t', tokenizer=tok)
    gen   = make_local_generate_fn(model, tok, max_tokens=100, temperature=0.7)

    sql, n_in, n_out = gen("how many rows?")  # 3 words → after chat template: '<chat> ...' (5 tokens)
    assert sql == 'SELECT count(*) FROM t'
    # FakeTokenizer counts words; chat template adds one prefix token '<chat>'.
    assert n_in  == len("<chat> how many rows?".split())  # = 4
    assert n_out == 5


def test_local_adapter_strips_sql_code_fence(fake_torch):
    tok   = FakeTokenizer()
    model = FakeModel(
        reply_tokens=4,
        reply_text="```sql\nSELECT 1\n```",
        tokenizer=tok,
    )
    gen = make_local_generate_fn(model, tok)
    sql, _, _ = gen("q")
    assert sql == 'SELECT 1'


def test_local_adapter_forwards_temperature_and_max_tokens(fake_torch):
    tok   = FakeTokenizer()
    model = FakeModel(reply_tokens=1, reply_text='X', tokenizer=tok)
    gen   = make_local_generate_fn(model, tok, max_tokens=123, temperature=0.42)
    gen("p")
    assert model.last_kwargs['max_new_tokens'] == 123
    assert model.last_kwargs['do_sample']      is True
    assert model.last_kwargs['temperature']    == 0.42


def test_local_adapter_greedy_when_temperature_zero(fake_torch):
    """do_sample must be False at T=0 — otherwise HF errors out."""
    tok   = FakeTokenizer()
    model = FakeModel(reply_tokens=1, reply_text='X', tokenizer=tok)
    gen   = make_local_generate_fn(model, tok, temperature=0.0)
    gen("p")
    assert model.last_kwargs['do_sample']  is False
    # temperature must NOT be forwarded when do_sample is False.
    assert 'temperature' not in model.last_kwargs


def test_local_adapter_falls_back_to_raw_prompt_for_base_models(fake_torch):
    """No chat_template (base model) → raw prompt is fed straight in."""
    tok = FakeTokenizer(chat_template=None)
    # FakeTokenizer.apply_chat_template would assert, so calling it would crash;
    # if _apply_chat_template correctly skips it, we never touch it.
    model = FakeModel(reply_tokens=2, reply_text='SELECT 1', tokenizer=tok)
    gen   = make_local_generate_fn(model, tok)
    sql, n_in, _ = gen("how many")  # 2 words, no chat prefix
    assert sql == 'SELECT 1'
    assert n_in == 2


# ── _apply_chat_template ─────────────────────────────────────────────────────

def test_apply_chat_template_uses_template_when_present():
    tok = FakeTokenizer(chat_template='<chat>')
    assert _apply_chat_template(tok, 'hello') == '<chat> hello'


def test_apply_chat_template_returns_raw_for_base_model():
    tok = FakeTokenizer(chat_template=None)
    assert _apply_chat_template(tok, 'hello') == 'hello'
