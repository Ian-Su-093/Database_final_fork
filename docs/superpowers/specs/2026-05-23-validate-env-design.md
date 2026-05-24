# Design: NL2SQL Environment Validation Script

**Date:** 2026-05-23  
**File to create:** `validate_env.py` (repo root)  
**Purpose:** Smoke-test the full episode loop of `NL2SQLEnv` against real Spider data and the pre-built corruption dataset, using a mock ClausePRM scorer. Runs before PPO training to confirm the env is wired up correctly.

---

## Context

The existing `tests/test_env.py` is guarded by `pytestmark` but calls `_tables()` and `_train()` at module level, so it crashes at collection time when Spider is absent. It also cannot run on this Mac without a manual path fix (`spider/` vs `clause_ppo/data/spider/`). This script is a standalone alternative that runs independently of pytest.

---

## Data Sources

| Resource | Path |
|----------|------|
| Spider data | `spider/` (actual location on this machine) |
| Corruption dataset | `clause_ppo/data/processed/corruption_dataset.json` |

The script exits immediately with a clear message if either path is missing.

---

## Components

### MockClausePRM

A drop-in scorer that simulates ClausePRM without loading the real model.

- Accepts `known_faulty_clause` (the ground-truth corrupted clause name from the corruption record).
- For each clause name passed to `score_clauses()`, returns `0.1` for the known faulty clause and `0.9` for all others.
- This ensures `get_faulty_clause(scores)` should always return the correct clause, making it a useful correctness check.

```python
class MockClausePRM:
    def score_clauses(self, clause_names: list[str], known_faulty: str) -> dict[str, float]:
        return {c: (0.1 if c == known_faulty else 0.9) for c in clause_names}
```

### Clause name mapping

The corruption dataset uses Spider's internal clause keys (`from`, `where`, `groupBy`, etc.). `get_faulty_clause()` works on any string keys, so scores are built using those same keys directly — no translation needed.

The mock scorer uses all six keys from `CLAUSE_ORDER = ['from', 'where', 'groupBy', 'having', 'select', 'orderBy']` with a uniform high score of `0.9`, overriding only the known faulty clause with `0.1`. This avoids the need to load `train_spider.json` to recover the parsed sql_dict, and is consistent with the Spider key format stored in `corrupted_clause`.

### Episode runner

For each record in `corruption_dataset.json`:

1. Build a sample dict: `{"question": ..., "db_id": ..., "query": original_query}`
2. `state = env.reset(sample)` — assert keys `{question, schema, db_id}` present and `schema` non-empty
3. `scores = mock_prm.score_clauses(known_faulty=record["corrupted_clause"])`  
   Returns `{clause: 0.1 if clause == known_faulty else 0.9}` for all six clause keys.
4. `predicted = env.get_faulty_clause(scores)` — assert equals `record["corrupted_clause"]`
5. `reward, done = env.step(record["original_query"])` — assert `reward == +1.0` and `done is True`
6. `reward, done = env.step(record["corrupted_query"])` — assert `reward == -1.0` and `done is True`

### Output format

```
Validating NL2SQL environment...
Spider:      spider/
Corruption:  clause_ppo/data/processed/corruption_dataset.json
Episodes:    261

Running episodes: ..........F..........
  [FAIL #11] db_id=concert_singer | check=negative_reward
             question: "What is the most common first name?"
             expected=-1.0  got=1.0

Results
-------
  Reset valid:        261/261
  Faulty clause ID:   261/261
  Positive reward:    261/261
  Negative reward:    260/261

FAILED — 1 episode(s) did not pass all checks.
```

Dots for full pass, `F` for any failure. Failures are printed inline then summarised at the end. Each of the four checks is tracked independently so failures are easy to diagnose.

---

## Error Handling

- Missing Spider dir or corruption file → print path and exit with code 1.
- `env.reset()` or `env.step()` raises an exception → catch, mark episode as failed, continue.
- Summary exit code: 0 if all episodes pass all checks, 1 otherwise.

---

## What This Does NOT Test

- Real ClausePRM scores (no GPU/model on this machine).
- PPO update logic (out of scope — that's Henry's training loop).
- Full 7000-sample train split (only the 261 verified corruption records are used).

---

## Success Criteria

The script passes when all four checks reach 100% across all 261 episodes. Any failure before training starts should be investigated — the most likely causes are the wrong Spider path or a corrupted `.sqlite` file.
