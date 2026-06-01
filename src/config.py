"""
Central configuration for eval + baseline pipeline.
"""

import os

# ── Repo paths ───────────────────────────────────────────────────────────────
_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))   # .../final/src
REPO_ROOT  = os.path.dirname(_THIS_DIR)                    # .../final
SPIDER_DIR = os.path.join(REPO_ROOT, 'clause_ppo', 'data', 'spider')

# ── Execution oracle ─────────────────────────────────────────────────────────
TIMEOUT_SECS   = 5.0      # hard SQLite timeout per query
REWARD_CORRECT = +1.0
REWARD_WRONG   = -1.0

# ── Baseline backbone (HF Inference API) ─────────────────────────────────────
BASELINE_MODEL = 'Qwen/Qwen2.5-Coder-1.5B-Instruct:featherless-ai'
MAX_TOKENS     = 500      # max generated tokens per API call
MAX_RETRIES    = 3        # full-regen attempts per sample (wrong-SQL retries)
TEMPERATURE    = 0.9      # baseline randomness (0.0 = greedy, 1.0 = high variance)

# ── Baseline backbone (local inference) ──────────────────────────────────────
LOCAL_MODEL  = 'Qwen/Qwen2.5-Coder-1.5B-Instruct'
LOCAL_DTYPE  = 'float16'  # 'float16' | 'bfloat16' | 'float32'
LOCAL_DEVICE = 'auto'     # 'auto' | 'cuda' | 'cpu'

# ── API retry policy ─────────────────────────────────────────────────────────
API_RETRIES      = 4      # attempts per API call before giving up
API_BACKOFF_SECS = 2.0    # base for exponential backoff: base * 2**attempt

# ── Eval diagnostics ─────────────────────────────────────────────────────────
CLAUSE_KEYWORDS = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY', 'LIMIT']


# ── .env loading ─────────────────────────────────────────────────────────────

def _load_dotenv(path: str) -> None:
    """
    Minimal `.env` reader (avoids a python-dotenv dependency in the eval venv).
    Sets KEY=VALUE pairs into os.environ; a real env var already set wins.
    """
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, val = line.partition('=')
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv(os.path.join(REPO_ROOT, '.env'))

# Secret — resolved from the environment (populated by .env above) at import.
HF_TOKEN = os.environ.get('HF_TOKEN')
