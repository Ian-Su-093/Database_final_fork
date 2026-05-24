# Interfaces

Actual function signatures as implemented. Update when signatures change.

---

## Sam's Code → Henry (already integrated in ppo_loop.py)

```python
# src/env/env.py  ✅ DONE, DO NOT CHANGE SIGNATURES

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
# src/eval/metrics.py  ✅ DONE

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
```

---

## Henry's Code → Sam (used in ppo_loop.py, for reference only)

```python
# clause_ppo/src/utils/execution.py  ✅ DONE (Henry)
def queries_produce_same_result(q1, q2, db_path, timeout_secs=5.0) -> bool

# clause_ppo/src/data/clause_splitter.py  ✅ DONE (Henry/Ian)
def schema_to_string(db_id: str, tables_dict: dict) -> str
def split_into_clauses(sql_dict: dict) -> list[tuple[str, object]]

# clause_ppo/src/training/ppo_loop.py  ✅ DONE (Henry)
def build_rewrite_prompt(question, schema, wrong_sql, faulty_clause, clause_names) -> str
def build_prm_prompt(question, schema, clause_names_up_to_faulty) -> str
def compute_reward(terminal, prm_score, alpha) -> float
def get_corrupted_sample(sample, tables_dict) -> tuple[str, str] | None
def train_ppo(config, spider_dir, prm_ckpt) -> list[dict]
```

---

## Ian → Sam (for evaluate.py, TBD)

```python
# src/baseline/full_regen.py  ← Ian implements this

def run_baseline(
    sample: dict,          # from load_spider() / train_spider.json
    model,                 # Qwen loaded via HuggingFace transformers
    tokenizer,
    max_retries: int = 3,
) -> dict:
    """
    Returns:
        {
            "predicted_sql": str,
            "token_cost":    int,   # input + output tokens, all attempts
            "attempts":      int,
        }
    Note: prompt format MUST match build_rewrite_prompt() style in ppo_loop.py
    for a fair comparison.
    """
```

---

## evaluate.py (Sam, TBD)

```
CLI: python scripts/evaluate.py --split dev [--max-retries 3] [--ppo-ckpt path]

Output:
| Method         | Accuracy@3 | Avg Token Cost |
| Full regen     |    ?       |      ?         |
| Clause PPO     |    ?       |      ?         |
```