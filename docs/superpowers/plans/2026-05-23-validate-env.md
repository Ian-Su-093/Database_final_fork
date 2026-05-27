# validate_env.py Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `validate_env.py` — a standalone script that smoke-tests `NL2SQLEnv` against the real Spider dataset and corruption records, using a mock ClausePRM scorer.

**Architecture:** Single script at the repo root. Loads `spider/tables.json` and `clause_ppo/data/processed/corruption_dataset.json`, initialises `NL2SQLEnv` with the real Spider path, and runs one episode per corruption record: `reset → mock-score → get_faulty_clause → step(original) → step(corrupted)`. Tracks four named checks independently. Prints dot/F progress and a per-check summary. Exits 0 on full pass, 1 on any failure.

**Tech Stack:** Python 3.10+, stdlib only (`json`, `os`, `sys`). Imports `NL2SQLEnv` from `src/env/env.py` (already implemented).

---

## File Map

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `validate_env.py` | Entire validation script |

No existing files are modified.

---

### Task 1: Skeleton, path validation, and data loading

**Files:**
- Create: `validate_env.py`

- [ ] **Step 1: Create the file with path constants and sys.path setup**

```python
#!/usr/bin/env python3
"""
Standalone smoke-test for NL2SQLEnv.

Runs each record from the corruption dataset as one training episode:
  1. env.reset(sample)             -> check state keys + non-empty schema
  2. MockClausePRM.score_clauses() -> check get_faulty_clause() returns right clause
  3. env.step(original_query)      -> check reward == +1.0
  4. env.step(corrupted_query)     -> check reward == -1.0

Usage:  python validate_env.py
Exit:   0 if all checks pass, 1 if any fail.
"""

import json
import os
import sys

REPO_ROOT       = os.path.dirname(os.path.abspath(__file__))
SPIDER_DIR      = os.path.join(REPO_ROOT, 'spider')
CORRUPTION_FILE = os.path.join(REPO_ROOT, 'clause_ppo', 'data', 'processed', 'corruption_dataset.json')

# Make src/ and clause_ppo/src/ importable (env.py needs both)
for _p in [os.path.join(REPO_ROOT, 'src'),
           os.path.join(REPO_ROOT, 'clause_ppo', 'src')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from env.env import NL2SQLEnv  # noqa: E402

CLAUSE_ORDER = ['from', 'where', 'groupBy', 'having', 'select', 'orderBy']


def _load_data():
    """Validate paths, load tables.json and corruption_dataset.json. Exits on error."""
    errors = []
    if not os.path.isdir(SPIDER_DIR):
        errors.append(f"Spider directory not found: {SPIDER_DIR}")
    if not os.path.isfile(CORRUPTION_FILE):
        errors.append(f"Corruption dataset not found: {CORRUPTION_FILE}")
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        sys.exit(1)

    with open(os.path.join(SPIDER_DIR, 'tables.json')) as f:
        tables = {t['db_id']: t for t in json.load(f)}

    with open(CORRUPTION_FILE) as f:
        records = json.load(f)

    return tables, records


if __name__ == '__main__':
    tables, records = _load_data()
    print(f"Loaded {len(records)} records. Spider OK.")
```

- [ ] **Step 2: Run to verify paths resolve and data loads**

```bash
python3 validate_env.py
```

Expected output:
```
Loaded 261 records. Spider OK.
```

If you see `ERROR: Spider directory not found`, confirm `spider/` exists at the repo root and contains `tables.json`.

- [ ] **Step 3: Commit**

```bash
git add validate_env.py
git commit -m "feat: add validate_env skeleton with path validation and data loading"
```

---

### Task 2: MockClausePRM

**Files:**
- Modify: `validate_env.py`

- [ ] **Step 1: Add MockClausePRM class after the imports block**

Add this class directly below the `CLAUSE_ORDER` constant:

```python
class MockClausePRM:
    """
    Simulates ClausePRM without loading the real model.
    Returns 0.1 for the known faulty clause and 0.9 for all others.
    This guarantees get_faulty_clause() returns the right clause when
    scores are passed to NL2SQLEnv.get_faulty_clause().
    """

    def score_clauses(self, known_faulty: str) -> dict[str, float]:
        return {c: (0.1 if c == known_faulty else 0.9) for c in CLAUSE_ORDER}
```

- [ ] **Step 2: Verify MockClausePRM inline — add a quick print to `__main__` and run**

Temporarily add to the bottom of `if __name__ == '__main__':`:

```python
    prm = MockClausePRM()
    scores = prm.score_clauses(known_faulty='where')
    print("Mock scores:", scores)
    assert scores['where'] == 0.1
    assert scores['from']  == 0.9
    print("MockClausePRM OK.")
```

Run:
```bash
python3 validate_env.py
```

Expected output:
```
Loaded 261 records. Spider OK.
Mock scores: {'from': 0.9, 'where': 0.1, 'groupBy': 0.9, 'having': 0.9, 'select': 0.9, 'orderBy': 0.9}
MockClausePRM OK.
```

- [ ] **Step 3: Remove the temporary print/assert lines from `__main__`**

Leave only:
```python
if __name__ == '__main__':
    tables, records = _load_data()
    print(f"Loaded {len(records)} records. Spider OK.")
```

- [ ] **Step 4: Commit**

```bash
git add validate_env.py
git commit -m "feat: add MockClausePRM to validate_env"
```

---

### Task 3: Four check functions

**Files:**
- Modify: `validate_env.py`

These are pure functions — no I/O, easy to reason about individually.

- [ ] **Step 1: Add the four check functions after the `MockClausePRM` class**

```python
def _check_reset(state: dict) -> tuple[bool, str]:
    required = {'question', 'schema', 'db_id'}
    missing = required - set(state.keys())
    if missing:
        return False, f"state missing keys: {missing}"
    if not state['schema'].strip():
        return False, "schema is empty"
    return True, ""


def _check_faulty_clause(predicted: str, expected: str) -> tuple[bool, str]:
    if predicted != expected:
        return False, f"predicted {predicted!r}, expected {expected!r}"
    return True, ""


def _check_reward(reward: float, done: bool,
                  expected: float, label: str) -> tuple[bool, str]:
    if done is not True:
        return False, f"{label}: done={done!r}, expected True"
    if reward != expected:
        return False, f"{label}: reward={reward}, expected {expected}"
    return True, ""
```

- [ ] **Step 2: Verify inline — add assertions to `__main__` and run**

Temporarily add to the bottom of `if __name__ == '__main__':`:

```python
    # _check_reset
    assert _check_reset({'question': 'q', 'schema': 's', 'db_id': 'd'}) == (True, "")
    assert _check_reset({'question': 'q', 'schema': '  ', 'db_id': 'd'})[0] is False
    assert _check_reset({'question': 'q', 'db_id': 'd'})[0] is False
    # _check_faulty_clause
    assert _check_faulty_clause('where', 'where') == (True, "")
    assert _check_faulty_clause('from', 'where')[0] is False
    # _check_reward
    assert _check_reward(1.0,  True,  1.0,  "orig") == (True, "")
    assert _check_reward(-1.0, True, -1.0, "corr") == (True, "")
    assert _check_reward(1.0,  True, -1.0, "corr")[0] is False
    print("Check functions OK.")
```

Run:
```bash
python3 validate_env.py
```

Expected:
```
Loaded 261 records. Spider OK.
Check functions OK.
```

- [ ] **Step 3: Remove the temporary assertion block from `__main__`**

- [ ] **Step 4: Commit**

```bash
git add validate_env.py
git commit -m "feat: add four episode check functions to validate_env"
```

---

### Task 4: Episode runner

**Files:**
- Modify: `validate_env.py`

- [ ] **Step 1: Add `run_episode()` after the check functions**

```python
def run_episode(env: NL2SQLEnv, prm: MockClausePRM, record: dict) -> dict:
    """
    Run one episode against a corruption record.

    Returns:
        {
            'passed':   bool,
            'failures': list[str],   # empty when passed
            'db_id':    str,
            'question': str,
        }
    """
    failures = []
    sample = {
        'question': record['question'],
        'db_id':    record['db_id'],
        'query':    record['original_query'],
    }

    # Check 1: reset
    try:
        state = env.reset(sample)
        ok, msg = _check_reset(state)
        if not ok:
            failures.append(f"[reset] {msg}")
    except Exception as exc:
        failures.append(f"[reset] raised {type(exc).__name__}: {exc}")
        # Cannot continue without a valid reset
        return {'passed': False, 'failures': failures,
                'db_id': record['db_id'], 'question': record['question']}

    # Check 2: faulty clause identification
    try:
        scores    = prm.score_clauses(known_faulty=record['corrupted_clause'])
        predicted = env.get_faulty_clause(scores)
        ok, msg   = _check_faulty_clause(predicted, record['corrupted_clause'])
        if not ok:
            failures.append(f"[faulty_clause] {msg}")
    except Exception as exc:
        failures.append(f"[faulty_clause] raised {type(exc).__name__}: {exc}")

    # Check 3: positive reward — original SQL should match gold
    # env._current is still set from the reset above
    try:
        reward, done = env.step(record['original_query'])
        ok, msg = _check_reward(reward, done, +1.0, 'original')
        if not ok:
            failures.append(f"[positive_reward] {msg}")
    except Exception as exc:
        failures.append(f"[positive_reward] raised {type(exc).__name__}: {exc}")

    # Check 4: negative reward — corrupted SQL should differ from gold
    # Reset with the same sample so gold is available; then step with corrupted SQL.
    try:
        env.reset(sample)
        reward, done = env.step(record['corrupted_query'])
        ok, msg = _check_reward(reward, done, -1.0, 'corrupted')
        if not ok:
            failures.append(f"[negative_reward] {msg}")
    except Exception as exc:
        failures.append(f"[negative_reward] raised {type(exc).__name__}: {exc}")

    return {
        'passed':   len(failures) == 0,
        'failures': failures,
        'db_id':    record['db_id'],
        'question': record['question'],
    }
```

- [ ] **Step 2: Smoke-test with the first record — add to `__main__` and run**

Temporarily replace the `__main__` block body with:

```python
    tables, records = _load_data()
    env = NL2SQLEnv(spider_dir=SPIDER_DIR, tables=tables)
    prm = MockClausePRM()
    result = run_episode(env, prm, records[0])
    print("Episode result:", result)
    assert result['passed'], f"First episode failed: {result['failures']}"
    print("run_episode OK.")
```

Run:
```bash
python3 validate_env.py
```

Expected:
```
Episode result: {'passed': True, 'failures': [], 'db_id': 'department_management', 'question': 'How many heads of the departments are older than 56 ?'}
run_episode OK.
```

- [ ] **Step 3: Restore `__main__` to just path validation and the loaded-records print**

```python
if __name__ == '__main__':
    tables, records = _load_data()
    print(f"Loaded {len(records)} records. Spider OK.")
```

- [ ] **Step 4: Commit**

```bash
git add validate_env.py
git commit -m "feat: add run_episode() to validate_env"
```

---

### Task 5: main() — loop, output, and summary

**Files:**
- Modify: `validate_env.py`

- [ ] **Step 1: Add `main()` function after `run_episode()`**

```python
def main() -> None:
    tables, records = _load_data()

    env = NL2SQLEnv(spider_dir=SPIDER_DIR, tables=tables)
    prm = MockClausePRM()

    print("Validating NL2SQL environment...")
    print(f"Spider:     {SPIDER_DIR}")
    print(f"Corruption: {CORRUPTION_FILE}")
    print(f"Episodes:   {len(records)}")
    print()
    print("Running episodes: ", end='', flush=True)

    results: list[dict] = []
    failed_episodes: list[tuple[int, dict]] = []

    for i, record in enumerate(records):
        result = run_episode(env, prm, record)
        results.append(result)
        if result['passed']:
            print('.', end='', flush=True)
        else:
            print('F', end='', flush=True)
            failed_episodes.append((i + 1, result))

    print('\n')

    # Print inline failure details
    for ep_num, result in failed_episodes:
        print(f"  [FAIL #{ep_num}] db_id={result['db_id']}")
        print(f"               question: {result['question']!r}")
        for msg in result['failures']:
            print(f"               {msg}")
    if failed_episodes:
        print()

    # Per-check summary
    total = len(results)
    checks = {
        'Reset valid':       lambda r: not any('[reset]'          in f for f in r['failures']),
        'Faulty clause ID':  lambda r: not any('[faulty_clause]'  in f for f in r['failures']),
        'Positive reward':   lambda r: not any('[positive_reward]' in f for f in r['failures']),
        'Negative reward':   lambda r: not any('[negative_reward]' in f for f in r['failures']),
    }

    print("Results")
    print("-------")
    for label, fn in checks.items():
        passed = sum(1 for r in results if fn(r))
        print(f"  {label:<20} {passed}/{total}")

    print()
    all_passed = all(r['passed'] for r in results)
    if all_passed:
        print("PASSED — all episodes passed all checks.")
        sys.exit(0)
    else:
        n_failed = sum(1 for r in results if not r['passed'])
        print(f"FAILED — {n_failed} episode(s) did not pass all checks.")
        sys.exit(1)
```

- [ ] **Step 2: Replace `__main__` block to call `main()`**

```python
if __name__ == '__main__':
    main()
```

- [ ] **Step 3: Run the full script**

```bash
python3 validate_env.py
```

Expected output (all passing):
```
Validating NL2SQL environment...
Spider:     /Users/henrylin/Coding/Database_final/spider
Corruption: /Users/henrylin/Coding/Database_final/clause_ppo/data/processed/corruption_dataset.json
Episodes:   261

Running episodes: .............................................................
.................................................................................
.................................................................................
.................................................................................

Results
-------
  Reset valid         261/261
  Faulty clause ID    261/261
  Positive reward     261/261
  Negative reward     261/261

PASSED — all episodes passed all checks.
```

Exit code must be 0:
```bash
echo "Exit code: $?"
```

Expected: `Exit code: 0`

- [ ] **Step 4: Commit**

```bash
git add validate_env.py
git commit -m "feat: add main loop and summary output to validate_env"
```

---

### Task 6: Final cleanup and README note

**Files:**
- Modify: `validate_env.py` (docstring only)

- [ ] **Step 1: Confirm the top-of-file docstring matches reality**

The docstring should already match. Verify it reads:

```python
"""
Standalone smoke-test for NL2SQLEnv.

Runs each record from the corruption dataset as one training episode:
  1. env.reset(sample)             -> check state keys + non-empty schema
  2. MockClausePRM.score_clauses() -> check get_faulty_clause() returns right clause
  3. env.step(original_query)      -> check reward == +1.0
  4. env.step(corrupted_query)     -> check reward == -1.0

Usage:  python validate_env.py
Exit:   0 if all checks pass, 1 if any fail.
"""
```

No changes needed if it already says this.

- [ ] **Step 2: Run once more to confirm nothing regressed**

```bash
python3 validate_env.py
echo "Exit code: $?"
```

Expected: all 261/261, exit code 0.

- [ ] **Step 3: Final commit**

```bash
git add validate_env.py
git commit -m "feat: complete validate_env smoke-test for NL2SQLEnv"
```

---

## Self-Review Notes

- **Spec coverage:** All four checks implemented (reset, faulty clause, positive reward, negative reward). Path validation, dot/F output, per-check summary, and exit codes all present.
- **Negative reward edge case:** The corruption engine already verifies via the execution oracle that each corrupted SQL actually changes the result (`changed = not queries_produce_same_result(...)`). All 261 records are pre-verified, so `-1.0` should be consistent across all episodes.
- **No placeholders:** Every step has complete code.
- **Type consistency:** `run_episode` returns the same dict shape everywhere it is referenced in `main()`.
