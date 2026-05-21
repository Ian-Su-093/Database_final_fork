# src/ — RL Environment & Evaluation

Sam's modules for the clause-PPO pipeline.

---

## Modules

| Module | What it does |
|--------|-------------|
| `src/env/env.py` | RL environment — wraps SQLite execution into a gym-style `reset / step` interface |
| `src/eval/metrics.py` | Evaluation — Spider EX and per-clause token F1 |

---

## Setup

Uses the same venv as the root `requirements.txt`. No extra dependencies.

```bash
source venv/bin/activate
```

`src/` and `clause_ppo/src/` both need to be on `sys.path`.
The root `conftest.py` already handles this for pytest.
For scripts, add both manually (see the example below).

---

## NL2SQLEnv

### Import

```python
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'clause_ppo/src')

from env.env import NL2SQLEnv
```

### Interface

```python
class NL2SQLEnv:
    def reset(self, sample: dict) -> dict:
        """
        Args:
            sample: one dict from load_spider() — needs 'question', 'db_id',
                    and 'gold_sql' (from Ian's loader) or raw 'query'.
        Returns:
            {"question": str, "schema": str, "db_id": str}
            'schema' is the formatted DB schema for the CodeLlama prompt.
        """

    def step(self, full_sql: str) -> tuple[float, bool]:
        """
        Call once after the full SQL is reconstructed.
        Returns:
            (+1.0, True)  if full_sql produces the same result as gold_sql
            (-1.0, True)  otherwise (wrong result, crash, or timeout)
            done is always True — one rewrite per episode.
        """

    def get_faulty_clause(self, clause_scores: dict[str, float]) -> str:
        """
        Args:
            clause_scores: e.g. {"SELECT": 0.91, "FROM": 0.88, "WHERE": 0.21}
                           Henry's score_clause() output, one entry per clause.
        Returns:
            The clause name with the lowest score, e.g. "WHERE".
        """
```

### Minimal episode loop

```python
from env.env import NL2SQLEnv
from reward.model import score_clause          # Henry's module
from data.loader import load_spider            # Ian's module

env     = NL2SQLEnv()
samples = load_spider("train")[4000:]          # PPO split

for sample in samples:
    state = env.reset(sample)
    # state["schema"] goes into the CodeLlama prompt

    # Henry's actor generates one clause at a time:
    clause_scores = {}
    for clause_name, clause_text in generated_clauses.items():
        context = {
            "question":        state["question"],
            "schema":          state["schema"],
            "clauses_so_far":  clause_scores,
        }
        clause_scores[clause_name] = score_clause(clause_name, clause_text, context)

    faulty_clause = env.get_faulty_clause(clause_scores)
    # CodeLlama rewrites faulty_clause → reconstruct full_sql ...

    reward, done = env.step(full_sql)
    # reward fed into PPO update
```

### Constructor options

```python
NL2SQLEnv(
    spider_dir   = "clause_ppo/data/spider",  # default
    tables       = None,   # pass pre-loaded tables dict to skip re-reading tables.json
    timeout_secs = 5.0,    # SQLite hard timeout per query
)
```

---

## Evaluation metrics

### Import

```python
from eval.metrics import execution_accuracy, partial_match
```

### execution_accuracy

Standard Spider EX metric.

```python
acc = execution_accuracy(predictions, dev_samples)
print(f"EX = {acc:.3f}")
```

- `predictions` — list of predicted SQL strings.
- `dev_samples` — list of Spider samples (needs `db_id` and `gold_sql` / `query`).
- Returns a float in [0, 1].

Same oracle as `env.step()`: a prediction that crashes or times out counts as wrong.

### partial_match

Per-clause token F1 — useful for diagnosing *which* clause the model is getting wrong.

```python
f1 = partial_match(predictions, dev_samples)
# {"SELECT": 0.91, "FROM": 0.96, "WHERE": 0.74, "GROUP BY": 0.0, ...}
```

All seven keywords are always present in the returned dict.
Clauses absent from both prediction and gold are excluded from that clause's
average (to avoid inflating scores).

**Note:** uses a flat regex split, not a full SQL parser. Good for coarse
diagnosis; use EX as the primary metric.

---

## Running the tests

```bash
source venv/bin/activate
pip install pytest
python -m pytest tests/test_env.py tests/test_metrics.py -v
```

33 tests, all green, no GPU required.
