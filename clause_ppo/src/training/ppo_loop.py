"""
PPO training loop for CLAUSE-PPO Phase 2.

Trains CodeLlama-7B via trl PPOTrainer to repair wrong SQL queries
at the clause level, combining ClausePRM dense rewards with NL2SQLEnv
terminal rewards.

Public API:
  build_rewrite_prompt   — formats the clause-rewrite prompt for the actor
  build_prm_prompt       — formats the scoring prompt for ClausePRM
  compute_reward         — combines terminal + alpha * prm_score
  get_corrupted_sample   — applies corruption engine to a Spider sample
  train_ppo              — main training entry point
"""

import os
import sys

# ── Make sibling packages importable ─────────────────────────────────────────
_SRC = os.path.dirname(os.path.abspath(__file__))          # .../src/training
_CLAUSE_PPO_SRC = os.path.normpath(os.path.join(_SRC, '..'))  # .../src
if _CLAUSE_PPO_SRC not in sys.path:
    sys.path.insert(0, _CLAUSE_PPO_SRC)

from data.clause_splitter import CLAUSE_LABELS, split_into_clauses  # noqa: E402


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_rewrite_prompt(
    question: str,
    schema: str,
    wrong_sql: str,
    faulty_clause: str,
    clause_names: list[str],
) -> str:
    """
    Build the rewrite prompt for the PPO actor.

    Args:
        question:      Natural language question.
        schema:        Formatted DB schema string from schema_to_string().
        wrong_sql:     SQL string with the faulty clause.
        faulty_clause: Spider clause key of the wrong clause (e.g. 'where').
        clause_names:  All clause keys present in the SQL (from split_into_clauses).

    Returns:
        Prompt string ending with '[SQL]'; actor generates everything after that.
    """
    task_parts = []
    for name in clause_names:
        label = CLAUSE_LABELS.get(name, name.upper())
        if name == faulty_clause:
            task_parts.append(f"The {label} clause is wrong.")
        else:
            task_parts.append(f"The {label} clause is correct.")
    faulty_label = CLAUSE_LABELS.get(faulty_clause, faulty_clause.upper())
    task_parts.append(f"Rewrite the full SQL fixing only the {faulty_label} clause.")

    return (
        f"[QUESTION] {question} "
        f"[SCHEMA] {schema} "
        f"[WRONG_SQL] {wrong_sql} "
        f"[TASK] {' '.join(task_parts)} "
        f"[SQL]"
    )


def build_prm_prompt(question: str, schema: str, clause_names_up_to_faulty: list[str]) -> str:
    """
    Build the ClausePRM scoring prompt for a clause.

    Matches PRMDataset prefix_query_str format: space-joined clause labels
    up to and including the faulty clause position (e.g. 'FROM WHERE' for
    position j=1 in [from, where, select] order).

    Args:
        clause_names_up_to_faulty: Ordered clause keys up to and including
            the faulty clause (e.g. ['from', 'where'] when where is faulty).
    """
    labels = [CLAUSE_LABELS.get(n, n.upper()) for n in clause_names_up_to_faulty]
    prefix_str = ' '.join(labels)
    return f"[QUESTION] {question} [SCHEMA] {schema} [PREFIX] {prefix_str}"


# ── Reward ────────────────────────────────────────────────────────────────────

def compute_reward(terminal: float, prm_score: float, alpha: float) -> float:
    """
    Combined PPO reward.

    Args:
        terminal:  env.step() result: +1.0 (correct) or -1.0 (wrong).
        prm_score: ClausePRM score for the rewritten clause, ∈ [0, 1].
        alpha:     Weight on the PRM score (from ppo_config.yaml).

    Returns:
        terminal + alpha * prm_score
    """
    return terminal + alpha * prm_score


# ── Episode helper ────────────────────────────────────────────────────────────

import random

from data.corruption import _CORRUPT_FNS          # noqa: E402
from utils.sql_utils import reconstruct_sql        # noqa: E402


def get_corrupted_sample(
    sample: dict,
    tables_dict: dict,
) -> tuple[str, str] | None:
    """
    Apply the corruption engine to produce a wrong SQL for one Spider sample.

    Calls the per-clause corrupt_* functions directly (not generate_corruptions)
    to avoid the hardcoded spider path in the orchestration function.
    Execution verification is skipped here — env.step() handles that at training time.

    Args:
        sample:      One Spider train entry with 'sql', 'db_id' fields.
        tables_dict: tables.json loaded as {db_id: tables_entry}.

    Returns:
        (wrong_sql_str, faulty_clause_key) or None if no corruption is possible.
    """
    sql_dict = sample.get('sql')
    if not sql_dict:
        return None

    tables = tables_dict.get(sample['db_id'])
    if tables is None:
        return None

    # Get clause names present in this SQL (execution order)
    clauses = split_into_clauses(sql_dict)
    clause_names = [name for name, _ in clauses]

    # Shuffle to avoid always corrupting the same clause
    shuffled = clause_names[:]
    random.shuffle(shuffled)

    for clause_name in shuffled:
        corrupt_fn = _CORRUPT_FNS.get(clause_name)
        if corrupt_fn is None:
            continue
        corrupted_dict = corrupt_fn(sql_dict, tables)
        if corrupted_dict is None:
            continue
        try:
            wrong_sql = reconstruct_sql(corrupted_dict, tables)
            return wrong_sql, clause_name
        except Exception:
            continue

    return None
