"""
NL2SQL RL environment for clause-level repair.

NL2SQLEnv:
  reset(sample)        — prepare the initial state for one episode.
  step(full_sql)       — execute the reconstructed SQL and return +1 / -1.
  get_faulty_clause()  — argmin over Henry's per-clause confidence scores.

Each episode rewrites exactly one clause, so step() is called once with the
final reconstructed SQL — done is always True.
"""

import json
import os
import sys
from typing import Optional

# ── Make src/ and clause_ppo/src importable (config + Henry's utils) ─────────
_REPO_ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC            = os.path.join(_REPO_ROOT, 'src')
_CLAUSE_PPO_SRC = os.path.join(_REPO_ROOT, 'clause_ppo', 'src')
for _p in (_SRC, _CLAUSE_PPO_SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import SPIDER_DIR, TIMEOUT_SECS, REWARD_CORRECT, REWARD_WRONG  # noqa: E402
from utils.execution import queries_produce_same_result                   # noqa: E402
from data.clause_splitter import schema_to_string                         # noqa: E402

# Back-compat alias — external code may import DEFAULT_SPIDER_DIR from here.
DEFAULT_SPIDER_DIR = SPIDER_DIR


class NL2SQLEnv:
    """RL environment for clause-level NL2SQL repair."""

    def __init__(
        self,
        spider_dir: str = SPIDER_DIR,
        tables: Optional[dict] = None,
        timeout_secs: float = TIMEOUT_SECS,
    ):
        """
        Args:
            spider_dir:   path to a Spider snapshot containing `tables.json` and `database/<db_id>/<db_id>.sqlite`.
            tables:       pre-loaded tables.json keyed by db_id. If None, the file is loaded from spider_dir on init.
            timeout_secs: hard timeout for each SQLite execution.
        """
        self.spider_dir   = spider_dir
        self.timeout_secs = timeout_secs

        if tables is None:
            with open(os.path.join(spider_dir, 'tables.json')) as f:
                tables = {t['db_id']: t for t in json.load(f)}
        self.tables = tables

        self._current: Optional[dict] = None

    # ── Public API ──────────────────────────────────────────────────────────

    def reset(self, sample: dict) -> dict:
        """
        Initialize a new episode from a Spider sample.
        Args:
            sample: one entry from load_spider(); must have `question`, `db_id`, and either `gold_sql` (preferred) or raw Spider `query`.
        Returns:
            state dict with keys `question`, `schema`, `db_id`.
        """
        self._current = sample
        schema_str = schema_to_string(sample['db_id'], self.tables)
        return {
            'question': sample['question'],
            'schema':   schema_str,
            'db_id':    sample['db_id'],
        }

    def step(self, full_sql: str) -> tuple[float, bool]:
        """
        Execute the reconstructed SQL and return the terminal reward.
        Args:
            full_sql: complete reconstructed SQL after clause rewrite.
        Returns:
            (reward, done). reward is +1.0 / -1.0; done is always True.
        """
        if self._current is None:
            raise RuntimeError("step() called before reset()")

        gold_sql = self._current.get('gold_sql') or self._current.get('query')
        if gold_sql is None:
            raise ValueError("sample missing 'gold_sql' / 'query' field")

        db_path = self._db_path(self._current['db_id'])
        same = queries_produce_same_result(
            full_sql, gold_sql, db_path, timeout_secs=self.timeout_secs
        )
        reward = REWARD_CORRECT if same else REWARD_WRONG
        return reward, True

    def get_faulty_clause(self, clause_scores: dict) -> str:
        """
        Return the clause name with the lowest score (argmin).
        Args:
            clause_scores: e.g. {"SELECT": 0.91, "FROM": 0.88, "WHERE": 0.21}.
        Returns:
            the key with the minimum value, e.g. "WHERE".
        """
        if not clause_scores:
            raise ValueError("clause_scores is empty")
        return min(clause_scores, key=clause_scores.get)

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _db_path(self, db_id: str) -> str:
        return os.path.join(self.spider_dir, 'database', db_id, f'{db_id}.sqlite')
