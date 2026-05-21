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

def score_clauses(sql: str, db_id: str) -> dict[str, float]:
    """
    Args:
        sql:    the (potentially wrong) SQL query
        db_id:  Spider database id
    Returns:
        per-clause score, lower = more likely wrong
        e.g. {"SELECT": 0.95, "FROM": 0.91, "WHERE": 0.12}
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
                "question":      str,
                "schema":        str,   # formatted DB schema for prompt
                "wrong_sql":     str,
                "faulty_clause": str,   # clause name, e.g. "WHERE"
                "db_id":         str
            }
        """

    def step(self, rewritten_clause: str) -> tuple[float, bool]:
        """
        Args:
            rewritten_clause: Qwen's rewrite of the faulty clause (text only)
        Returns:
            reward: +1.0 if result matches gold, -1.0 otherwise
            done:   always True (one clause rewrite per episode)
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
