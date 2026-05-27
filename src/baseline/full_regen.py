"""
Full-regeneration baseline for NL2SQL evaluation.

run_baseline(): generate full SQL from question + schema; retry up to
  ``max_retries`` if execution does not match gold.

Generation is injected as a ``generate_fn`` so the loop is backend-agnostic.
Any callable with this contract works:

    generate_fn(prompt: str) -> (sql_text: str, n_input_tokens: int, n_output_tokens: int)

make_hf_api_generate_fn() adapts a huggingface_hub InferenceClient
(chat-completions) to that contract. The baseline backbone is
Qwen2.5-Coder-1.5B served via the HF Inference API — a deliberately small,
remote model. NOTE: this differs from the PPO actor (CodeLlama-7B, local),
so the eval table compares a cheap API baseline against the trained model,
not two configurations of the same backbone. See .claude/docs/PIPELINE.md.

The prompt mirrors build_rewrite_prompt() in
clause_ppo/src/training/ppo_loop.py (minus the [WRONG_SQL] section) so the
baseline and PPO actor see the same input layout.
"""

import os
import re
import sys
from typing import Callable, Optional

# ── Make sibling packages importable ────────────────────────────────────────
_REPO_ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLAUSE_PPO_SRC = os.path.join(_REPO_ROOT, 'clause_ppo', 'src')
_SRC_ROOT       = os.path.join(_REPO_ROOT, 'src')
for _p in (_CLAUSE_PPO_SRC, _SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from env.env import NL2SQLEnv, DEFAULT_SPIDER_DIR   # noqa: E402


# ── Types ──────────────────────────────────────────────────────────────────

# prompt -> (sql_text, n_input_tokens, n_output_tokens)
GenerateFn = Callable[[str], tuple[str, int, int]]


# ── Module-level defaults ──────────────────────────────────────────────────

DEFAULT_MAX_TOKENS = 500   # matches the teammate's InferenceClient call


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
    sample:      dict,
    generate_fn: GenerateFn,
    max_retries: int = 3,
    env:         Optional[NL2SQLEnv] = None,
    spider_dir:  str = DEFAULT_SPIDER_DIR,
    tables:      Optional[dict] = None,
) -> dict:
    """
    Full-regen baseline for one Spider sample.

    Args:
        sample:      Spider sample with ``question``, ``db_id``, and
                     ``gold_sql`` / ``query``.
        generate_fn: callable ``prompt -> (sql, n_in, n_out)``. Build one with
                     make_hf_api_generate_fn() for the HF Inference API.
        max_retries: maximum generation attempts. Returns early on the first
                     attempt whose execution matches gold.
        env:         pre-instantiated NL2SQLEnv (recommended — avoids
                     re-loading tables.json on every call). One is constructed
                     from ``spider_dir`` / ``tables`` when None.
        spider_dir, tables: forwarded to NL2SQLEnv when ``env`` is None.

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
        sql, n_in, n_out = generate_fn(prompt)
        predicted_sql = sql
        token_cost   += n_in + n_out

        reward, _ = env.step(sql)
        if reward > 0:
            break

    return {
        'predicted_sql': predicted_sql,
        'token_cost':    token_cost,
        'attempts':      attempts,
    }


def make_hf_api_generate_fn(
    client,
    model:              str,
    max_tokens:         int = DEFAULT_MAX_TOKENS,
    fallback_tokenizer=None,
) -> GenerateFn:
    """
    Adapt a huggingface_hub InferenceClient to the generate_fn contract.

    Token counts come from the server's ``completion.usage`` when present.
    If a provider omits usage, ``fallback_tokenizer`` (a HF tokenizer) is used
    to count locally; with neither, we fail loudly rather than report 0.

    Args:
        client:             a huggingface_hub.InferenceClient.
        model:              model id, e.g. 'qwen/qwen2.5-coder-1.5b'.
        max_tokens:         max generated tokens per call.
        fallback_tokenizer: optional HF tokenizer used only when the API
                            response carries no usage stats.

    Returns:
        generate_fn(prompt) -> (sql_text, n_input_tokens, n_output_tokens).
    """
    def generate_fn(prompt: str) -> tuple[str, int, int]:
        completion = client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=max_tokens,
        )
        raw_text = completion.choices[0].message.content or ''
        n_in, n_out = _usage_tokens(completion, prompt, raw_text, fallback_tokenizer)
        return _extract_sql(raw_text), n_in, n_out

    return generate_fn


# ── Helpers ────────────────────────────────────────────────────────────────

_FENCE_PATTERN = re.compile(r'```(?:sql)?\s*(.*?)```', re.DOTALL | re.IGNORECASE)


def _extract_sql(text: str) -> str:
    """
    Pull SQL out of a chat model's reply.

    Chat models reliably wrap SQL in ```sql ...``` fences; the env executes the
    raw string, so stripping the fence is required for a fair EX score. When no
    fence is present, the trimmed text is returned unchanged.
    """
    text = text.strip()
    match = _FENCE_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return text


def _usage_tokens(
    completion,
    prompt: str,
    output_text: str,
    fallback_tokenizer,
) -> tuple[int, int]:
    """
    Resolve (input_tokens, output_tokens) for one API call.

    Prefers the server's usage stats; falls back to a local tokenizer; raises
    if neither is available so token_cost is never silently zero.
    """
    usage = getattr(completion, 'usage', None)
    if usage is not None:
        n_in  = getattr(usage, 'prompt_tokens', None)
        n_out = getattr(usage, 'completion_tokens', None)
        if n_in is not None and n_out is not None:
            return n_in, n_out

    if fallback_tokenizer is not None:
        return (
            len(fallback_tokenizer.encode(prompt)),
            len(fallback_tokenizer.encode(output_text)),
        )

    raise RuntimeError(
        "API response carries no usage stats and no fallback_tokenizer was "
        "provided; cannot compute token_cost. Pass fallback_tokenizer="
        "AutoTokenizer.from_pretrained(model) to make_hf_api_generate_fn()."
    )
