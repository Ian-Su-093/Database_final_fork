"""Tests for scripts/evaluate.py — CLI helpers and PPO stub."""
import io
import json
import os
import sys

import pytest

# scripts/ is not on sys.path by default — add it for test imports.
_SCRIPTS = os.path.join(os.path.dirname(__file__), '..', 'scripts')
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import evaluate  # noqa: E402

SPIDER_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'clause_ppo', 'data', 'spider'
)


# ── run_clause_ppo stub ────────────────────────────────────────────────────

def test_run_clause_ppo_raises_not_implemented():
    # PPO inference is not wired up yet — the stub must fail loudly.
    with pytest.raises(NotImplementedError) as excinfo:
        evaluate.run_clause_ppo([{'db_id': 'x'}], ppo_ckpt='ignored', max_retries=3)
    msg = str(excinfo.value)
    # The message should point to the actual gap so the next reader knows what to do.
    assert 'PPO inference' in msg
    assert 'ppo_loop.py'   in msg


# ── build_inference_client token guard ─────────────────────────────────────

def test_build_inference_client_requires_token():
    # No token (and the lazy huggingface_hub import never happens) → fail loudly.
    with pytest.raises(SystemExit, match="HF token"):
        evaluate.build_inference_client(token=None)


# ── print_table output formatting ──────────────────────────────────────────

def test_print_table_includes_n_in_header(capsys):
    evaluate.print_table(
        [{'method': 'Full regen', 'accuracy': 0.6, 'avg_tokens': 123.4}],
        n=3,
    )
    out = capsys.readouterr().out
    assert 'Accuracy@3'    in out
    assert 'Full regen'    in out
    assert '0.600'         in out
    assert '123.4'         in out


def test_print_table_renders_multiple_rows(capsys):
    evaluate.print_table(
        [
            {'method': 'Full regen', 'accuracy': 0.50, 'avg_tokens': 100.0},
            {'method': 'Clause PPO', 'accuracy': 0.75, 'avg_tokens': 250.5},
        ],
        n=3,
    )
    out = capsys.readouterr().out
    assert 'Full regen'   in out
    assert 'Clause PPO'   in out
    assert '0.500'        in out
    assert '0.750'        in out


# ── load_spider (requires Spider data) ─────────────────────────────────────

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(SPIDER_DIR, 'dev.json')),
    reason="Spider dataset not found at clause_ppo/data/spider/",
)


def test_load_spider_dev_returns_nonempty_list():
    samples = evaluate.load_spider('dev', SPIDER_DIR)
    assert isinstance(samples, list)
    assert len(samples) > 0
    s0 = samples[0]
    assert 'question' in s0
    assert 'db_id'    in s0
    assert 'query'    in s0   # raw Spider field — env.step accepts this


def test_load_spider_train_returns_nonempty_list():
    samples = evaluate.load_spider('train', SPIDER_DIR)
    assert isinstance(samples, list)
    assert len(samples) > 4000   # ppo_split_start is 4000


# ── dump_predictions writes valid JSON ─────────────────────────────────────

def test_dump_predictions_writes_valid_json(tmp_path):
    import argparse
    samples = [{
        'db_id':    'dept',
        'question': 'how many?',
        'query':    'SELECT count(*) FROM t',
    }]
    out_path = tmp_path / "preds.json"
    args = argparse.Namespace(split='dev', max_retries=3)
    evaluate.dump_predictions(
        str(out_path), samples,
        preds=['SELECT count(*) FROM t'],
        tokens=[42],
        attempts=[1],
        args=args,
    )
    payload = json.loads(out_path.read_text())
    assert payload['split']       == 'dev'
    assert payload['max_retries'] == 3
    assert payload['samples'][0]['predicted_sql'] == 'SELECT count(*) FROM t'
    assert payload['samples'][0]['gold_sql']      == 'SELECT count(*) FROM t'
    assert payload['samples'][0]['token_cost']    == 42
