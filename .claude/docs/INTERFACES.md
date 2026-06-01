# Interfaces

Actual function signatures as implemented. Update when signatures change.

---

## Sam's Code → Henry (already integrated in ppo_loop.py)

```python
# src/env/env.py

class NL2SQLEnv:
    def __init__(self, spider_dir: str, tables: dict = None, timeout_secs: float = 5.0)

    def reset(self, sample: dict) -> dict:
        """
        Args:   sample from train_spider.json (needs 'question', 'db_id', 'gold_sql'/'query')
        Returns: {"question": str, "schema": str, "db_id": str}
        """

    def step(self, full_sql: str) -> tuple[float, bool]:
        """
        Args:   full reconstructed SQL after rewrite
        Returns: (reward: +1.0/-1.0, done: True)
        """

    def get_faulty_clause(self, clause_scores: dict[str, float]) -> str:
        """argmin helper — not used by ppo_loop.py directly"""
```

---

## Sam's Code → Everyone

```python
# src/eval/metrics.py

def execution_accuracy(
    predictions: list[str],
    samples: list[dict],
    spider_dir: str = "clause_ppo/data/spider",
    timeout_secs: float = 5.0,
) -> float
    """Spider EX: fraction matching gold execution result."""

def partial_match(
    predictions: list[str],
    samples: list[dict],
) -> dict[str, float]
    """Per-clause token F1. Returns {clause_keyword: mean_F1}."""

def split_sql_prefixes(sql: str) -> list[tuple[str, str]]
    """
    Raw SQL string → [(clause_label, cumulative_prefix), ...]
    Example: "SELECT a FROM t WHERE x=1"
      → [('SELECT', 'SELECT a'),
         ('FROM',   'SELECT a FROM t'),
         ('WHERE',  'SELECT a FROM t WHERE x = 1')]
    Used by score_clauses() in best_of_n.py for per-clause PRM scoring.
    """
```

---

## Henry's Code → Sam (used in ppo_loop.py, for reference only)

```python
# clause_ppo/src/utils/execution.py
def queries_produce_same_result(q1, q2, db_path, timeout_secs=5.0) -> bool

# clause_ppo/src/data/clause_splitter.py
def schema_to_string(db_id: str, tables_dict: dict) -> str
def split_into_clauses(sql_dict: dict) -> list[tuple[str, object]]

# clause_ppo/src/training/ppo_loop.py
def build_rewrite_prompt(question, schema, wrong_sql, faulty_clause, clause_names) -> str
def build_prm_prompt(question, schema, clause_names_up_to_faulty) -> str
def compute_reward(terminal, prm_score, alpha) -> float
def get_corrupted_sample(sample, tables_dict) -> tuple[str, str] | None
def train_ppo(config, spider_dir, prm_ckpt) -> list[dict]
```

---

## Baseline (src/baseline/full_regen.py)

```python
GenerateFn = Callable[[str], tuple[str, int, int]]
# contract: prompt -> (sql_text, n_input_tokens, n_output_tokens)

def build_baseline_prompt(question: str, schema: str) -> str
    """[QUESTION] ... [SCHEMA] ... [TASK] ... [SQL] — mirrors build_rewrite_prompt()."""

def run_baseline(
    sample: dict,              # from load_spider() / train_spider.json
    generate_fn: GenerateFn,   # injected backend (see make_hf_api_generate_fn)
    max_retries: int = 3,
    env: NL2SQLEnv = None,     # reused across samples; built from spider_dir/tables if None
    spider_dir: str = "clause_ppo/data/spider",
    tables: dict = None,
) -> dict:
    """
    Returns: {"predicted_sql": str, "token_cost": int, "attempts": int, "success": bool}
      success    = execution correctness (matched gold within max_retries), NOT string equality.
      token_cost = cumulative (input + output) tokens across all attempts.
    """

def make_hf_api_generate_fn(
    client,                    # huggingface_hub.InferenceClient
    model: str,                # 'qwen/qwen2.5-coder-1.5b'
    max_tokens: int = 500,
    fallback_tokenizer = None, # used only if the API omits usage stats
) -> GenerateFn
    """Adapts chat-completions to GenerateFn; token counts from completion.usage."""
```

Backbone differs from the PPO actor (Qwen-1.5B API vs CodeLlama-7B local) —
intentional, documented in PIPELINE.md / QUESTIONS.md.

---

## Plan B (src/eval/best_of_n.py, src/baseline/plan_b_inference.py)

```python
# entry point called by evaluate.py → plan_b_inference.py
def run_plan_b_for_evaluate(
    samples: list[dict],
    prm_ckpt: str,
    max_retries: int = MAX_RETRIES,   # doubles as n_candidates
) -> tuple[list[str], list[int], list[int]]
    """Returns (predictions, token_costs, attempt_counts)."""

# core loop (also callable directly)
def eval_best_of_n_direct(
    samples: list[dict],
    spider_dir: str,
    prm_ckpt: str,
) -> tuple[list[str], list[int], list[int], dict]

# per-sample clause scoring
def score_clauses(
    scorer: PRMScorer,
    question: str,
    schema: str,
    sql: str,
) -> dict[str, float]
    """
    Returns {clause_label: PRM score}; {} if no recognisable clauses.
    Faulty clause = argmin; callers fall back to 'SELECT' on empty dict.
    """
```

All generation/eval knobs (`MAX_TOKENS`, `TEMPERATURE`, `MAX_RETRIES`) come
from `src/config.py` — shared with the baseline; no separate YAML needed.

---

## evaluate.py (Sam)  ✅ DONE (baseline + Plan B; PPO path stubbed)

```
CLI: python scripts/evaluate.py --split dev [--max-retries 3]
       [--backend api|local] [--model <id>]
       [--max-tokens 500] [--max-samples N] [--output preds.json]
       [--plan-b-ckpt <path>] [--ppo-ckpt <path>]

HF token: read from HF_TOKEN in .env (repo root) — never a CLI flag.
Defaults live in src/config.py.

Output:
| Method        | Accuracy@N | Avg Token Cost |
| Full regen    |    ?       |      ?         |
| Plan B (PRM)  |    ?       |      ?         |   ← only if --plan-b-ckpt
| Clause PPO    |    ?       |      ?         |   ← only if --ppo-ckpt (stub)

--output writes per-sample JSON; includes plan_b_sql and plan_b_token_cost
fields when Plan B was run.
```

PPO path raises NotImplementedError until Henry adds an actor-loading
inference entry point (see QUESTIONS.md).