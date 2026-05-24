"""
Full-regeneration baseline for NL2SQL evaluation.

run_baseline(): generate full SQL from question + schema; retry up to
  ``max_retries`` if execution does not match gold.

The prompt mirrors build_rewrite_prompt() in
clause_ppo/src/training/ppo_loop.py (minus the [WRONG_SQL] / [TASK] sections)
so the baseline and PPO actor are compared on the same input format.

Scaffolded by Sam — Ian owns the final implementation per
.claude/docs/INTERFACES.md.
"""

import os
import sys
from typing import Optional

# ── Make sibling packages importable ────────────────────────────────────────
_REPO_ROOT       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLAUSE_PPO_SRC  = os.path.join(_REPO_ROOT, 'clause_ppo', 'src')
_SRC_ROOT        = os.path.join(_REPO_ROOT, 'src')
for _p in (_CLAUSE_PPO_SRC, _SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from env.env import NL2SQLEnv, DEFAULT_SPIDER_DIR   # noqa: E402


# ── Module-level defaults ──────────────────────────────────────────────────

DEFAULT_MAX_NEW_TOKENS = 256
# Matches ppo_config.yaml so the baseline samples at the same temperature as
# the PPO actor — without sampling, max_retries is meaningless at T=0.
DEFAULT_TEMPERATURE    = 0.7


# ── Public API ─────────────────────────────────────────────────────────────

def build_baseline_prompt(question: str, schema: str) -> str:
    """
    Full-regen prompt. Same [QUESTION] / [SCHEMA] header as
    build_rewrite_prompt() so input-token counts stay comparable.
    """
    return (
        f"[QUESTION] {question} "
        f"[SCHEMA] {schema} "
        f"[TASK] Generate the full SQL query. "
        f"[SQL]"
    )


def run_baseline(
    sample:        dict,
    model,
    tokenizer,
    max_retries:   int = 3,
    env:           Optional[NL2SQLEnv] = None,
    spider_dir:    str = DEFAULT_SPIDER_DIR,
    tables:        Optional[dict] = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature:   float = DEFAULT_TEMPERATURE,
) -> dict:
    """
    Full-regen baseline for one Spider sample.

    Args:
        sample:         Spider sample with ``question``, ``db_id``, and
                        ``gold_sql`` / ``query``.
        model:          HF causal-LM (CodeLlama-7B per ppo_config.yaml).
        tokenizer:      matching HF tokenizer.
        max_retries:    maximum generation attempts. Returns early on the
                        first attempt whose execution matches gold.
        env:            pre-instantiated NL2SQLEnv (recommended — avoids
                        re-loading tables.json on every call). One is
                        constructed from ``spider_dir`` / ``tables`` when None.
        spider_dir, tables: forwarded to NL2SQLEnv when ``env`` is None.
        max_new_tokens, temperature: forwarded to ``model.generate``.

    Returns:
        ``{"predicted_sql": str, "token_cost": int, "attempts": int}``
        where ``token_cost`` is the cumulative (input + output) token count
        across every attempt made.
    """
    if env is None:
        env = NL2SQLEnv(spider_dir=spider_dir, tables=tables)
    state  = env.reset(sample)
    prompt = build_baseline_prompt(state['question'], state['schema'])

    predicted_sql = ''
    token_cost    = 0
    attempts      = 0

    for _ in range(max_retries):
        attempts += 1
        sql, tokens_used = _generate_sql(
            model, tokenizer, prompt, max_new_tokens, temperature,
        )
        predicted_sql = sql
        token_cost   += tokens_used

        reward, _ = env.step(sql)
        if reward > 0:
            break

    return {
        'predicted_sql': predicted_sql,
        'token_cost':    token_cost,
        'attempts':      attempts,
    }


# ── Helpers ────────────────────────────────────────────────────────────────

def _generate_sql(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> tuple[str, int]:
    """
    Single generation call.

    Returns:
        (sql_string, total_tokens) where ``total_tokens`` is the count of
        input prompt tokens plus newly generated tokens.
    """
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    input_len = input_ids.shape[-1]

    device = getattr(model, 'device', None)
    if device is not None:
        input_ids = input_ids.to(device)

    do_sample  = temperature > 0.0
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs['temperature'] = temperature

    output        = model.generate(input_ids, **gen_kwargs)
    generated_ids = output[0][input_len:]
    output_len    = generated_ids.shape[-1]
    sql           = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return sql, input_len + output_len
