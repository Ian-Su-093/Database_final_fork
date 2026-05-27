# Interfaces

Agreed function signatures between modules.  
Update this file whenever a signature is finalized — mark [TBD] until confirmed with teammate.

---

## Ian → Everyone

```python
# src/data/loader.py

def load_spider(split: str) -> list[dict]:
    """
    Args:
        split: "train" | "dev"
    Returns:
        [
            {
                "question": str,
                "gold_sql":  str,
                "db_id":     str,
                "sql":       dict   # parsed AST from Spider (use for clause labels)
            },
            ...
        ]
    """

# src/data/parser.py

def parse_clauses(sql: str) -> dict[str, str]:
    """
    Args:
        sql: a SQL query string
    Returns:
        {
            "SELECT":   "name",
            "FROM":     "singer",
            "WHERE":    "age > 20",   # empty string if clause absent
            "GROUP BY": "",
            "ORDER BY": "",
            "HAVING":   "",
            "LIMIT":    ""
        }
    """
```

---

## Henry → Sam

```python
# src/reward/model.py

def score_clause(clause_name: str, clause_text: str, context: dict) -> float:
    """
    Called once per clause, right after Qwen generates it.
    Args:
        clause_name:  e.g. "WHERE"
        clause_text:  e.g. "age > 200"
        context:      {"question": str, "schema": str, "clauses_so_far": dict}
    Returns:
        float ∈ [0, 1], lower = more likely wrong
    Note:
        No SQL execution — model confidence only.
    """
```

---

## Sam → Henry

```python
# src/env/env.py

class NL2SQLEnv:
    def reset(self, sample: dict) -> dict:
        """
        Args:
            sample: one entry from load_spider()
        Returns:
            state = {
                "question": str,
                "schema":   str,   # formatted DB schema for CodeLlama prompt
                "db_id":    str
            }
        """

    def step(self, full_sql: str) -> tuple[float, bool]:
        """
        Called ONCE after all clauses are generated and reconstructed.
        Args:
            full_sql: complete reconstructed SQL after clause rewrite
        Returns:
            reward: +1.0 if execution result matches gold, -1.0 otherwise
            done:   always True (one rewrite per episode)
        """

    def get_faulty_clause(self, clause_scores: dict[str, float]) -> str:
        """
        Trivial helper — just argmin.
        Args:
            clause_scores: output of Henry's score_clause() calls
        Returns:
            clause name with lowest score, e.g. "WHERE"
        """
```

---

## Sam → Everyone

```python
# src/eval/metrics.py

def execution_accuracy(predictions: list[str], samples: list[dict]) -> float:
    """
    Standard Spider EX metric.
    predictions: list of predicted SQL strings
    samples:     list of Spider samples (needs gold_sql and db_id)
    """

def partial_match(predictions: list[str], samples: list[dict]) -> dict[str, float]:
    """
    Per-clause F1.
    Returns e.g. {"SELECT": 0.91, "WHERE": 0.74, ...}
    """
```
