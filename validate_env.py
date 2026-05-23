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


class MockClausePRM:
    """
    Simulates ClausePRM without loading the real model.
    Returns 0.1 for the known faulty clause and 0.9 for all others.
    This guarantees get_faulty_clause() returns the right clause when
    scores are passed to NL2SQLEnv.get_faulty_clause().
    """

    def score_clauses(self, known_faulty: str) -> dict[str, float]:
        return {c: (0.1 if c == known_faulty else 0.9) for c in CLAUSE_ORDER}


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
        'Reset valid':       lambda r: not any('[reset]'           in f for f in r['failures']),
        'Faulty clause ID':  lambda r: not any('[faulty_clause]'   in f for f in r['failures']),
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
    main()
