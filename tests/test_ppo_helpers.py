"""Tests for pure helper functions in ppo_loop.py."""
import os
import sys

import pytest

# Make clause_ppo/src importable
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [
    os.path.join(_REPO, 'clause_ppo', 'src'),
    os.path.join(_REPO, 'src'),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from training.ppo_loop import build_rewrite_prompt, build_prm_prompt, compute_reward


def test_build_rewrite_prompt_contains_sections():
    prompt = build_rewrite_prompt(
        question="How many singers?",
        schema="Table: singer | Columns: id (number), name (text)",
        wrong_sql="SELECT name FROM singer",
        faulty_clause="select",
        clause_names=["from", "select"],
    )
    assert "[QUESTION]" in prompt
    assert "[SCHEMA]" in prompt
    assert "[WRONG_SQL]" in prompt
    assert "[TASK]" in prompt
    assert "[SQL]" in prompt


def test_build_rewrite_prompt_marks_faulty_clause_as_wrong():
    prompt = build_rewrite_prompt(
        question="Q",
        schema="S",
        wrong_sql="SELECT name FROM singer",
        faulty_clause="select",
        clause_names=["from", "select"],
    )
    assert "SELECT clause is wrong" in prompt
    assert "FROM clause is correct" in prompt


def test_build_rewrite_prompt_rewrite_instruction_present():
    prompt = build_rewrite_prompt(
        question="Q",
        schema="S",
        wrong_sql="SELECT x FROM t WHERE y > 1",
        faulty_clause="where",
        clause_names=["from", "where", "select"],
    )
    assert "Rewrite the full SQL fixing only the WHERE clause" in prompt


def test_build_prm_prompt_matches_prm_training_format():
    prompt = build_prm_prompt(
        question="How many singers?",
        schema="Table: singer | Columns: id (number)",
        faulty_clause="where",
    )
    # Must exactly match PRMDataset._format_input format
    assert prompt == "[QUESTION] How many singers? [SCHEMA] Table: singer | Columns: id (number) [PREFIX] WHERE"


def test_build_prm_prompt_group_by_label():
    prompt = build_prm_prompt(question="Q", schema="S", faulty_clause="groupBy")
    assert "[PREFIX] GROUP BY" in prompt


def test_compute_reward_positive_terminal():
    reward = compute_reward(terminal=1.0, prm_score=0.8, alpha=0.5)
    assert abs(reward - 1.4) < 1e-6


def test_compute_reward_negative_terminal():
    reward = compute_reward(terminal=-1.0, prm_score=0.2, alpha=0.5)
    assert abs(reward - (-0.9)) < 1e-6


def test_compute_reward_alpha_zero():
    # alpha=0 → reward == terminal only
    assert compute_reward(1.0, 0.99, alpha=0.0) == 1.0
    assert compute_reward(-1.0, 0.99, alpha=0.0) == -1.0


def test_compute_reward_alpha_one():
    # alpha=1.0 → reward = terminal + prm_score
    assert abs(compute_reward(1.0, 0.5, alpha=1.0) - 1.5) < 1e-6
