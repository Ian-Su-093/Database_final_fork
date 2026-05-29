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

### Episode loop (from `ppo_loop.py`)

The training loop lives entirely in Henry's `ppo_loop.py`. `env` is called like this:

```python
from env.env import NL2SQLEnv
from training.ppo_loop import get_corrupted_sample, build_rewrite_prompt, compute_reward

env = NL2SQLEnv(spider_dir=spider_dir, tables=tables_dict)

for sample in ppo_samples:                        # train_spider[4000:]
    # 1. Corruption engine tells us which clause is wrong (no PRM needed here)
    corruption = get_corrupted_sample(sample, tables_dict)
    if corruption is None:
        continue
    wrong_sql, faulty_clause = corruption          # e.g. wrong_sql, "where"

    # 2. Env provides the initial state (question + formatted schema)
    state = env.reset(sample)

    # 3. Build prompt → CodeLlama rewrites the full SQL
    prompt = build_rewrite_prompt(
        state['question'], state['schema'],
        wrong_sql, faulty_clause, clause_names,
    )
    rewritten_sql = ppo_trainer.generate(prompt)   # simplified

    # 4. Terminal reward from env (single call per episode)
    terminal, done = env.step(rewritten_sql)       # +1.0 or -1.0

    # 5. Dense reward from ClausePRM (scored once on the prefix, not per-clause)
    prm_score = prm(build_prm_prompt(...))         # float in [0, 1]

    reward = compute_reward(terminal, prm_score, alpha=0.5)
    # → ppo_trainer.step(...)
```

**Note on `get_faulty_clause`:** not called during training — the corruption engine
already knows the faulty clause. It is used in `validate_env.py` (integration test
with a mock PRM) and will be relevant at inference time when no corruption engine
is available.

**Note on `score_clause`:** there is no per-clause scoring loop during generation.
The PRM scores the corrupted prefix once after generation (see `build_prm_prompt`).
Clause scoring during autoregressive generation is a future extension.

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

## Baseline (`src/baseline/full_regen.py`)

Full-SQL regeneration baseline for the eval table. `run_baseline` is
backend-agnostic — it takes a `generate_fn` rather than a model, so any
backend works as long as it satisfies the contract:

```python
generate_fn(prompt: str) -> (sql_text: str, n_input_tokens: int, n_output_tokens: int)
```

### With the HF Inference API (Qwen2.5-Coder-1.5B)

```python
from huggingface_hub import InferenceClient
from baseline.full_regen import make_hf_api_generate_fn, run_baseline
from config import HF_TOKEN, BASELINE_MODEL   # HF_TOKEN read from .env

client      = InferenceClient(token=HF_TOKEN)
generate_fn = make_hf_api_generate_fn(client, model=BASELINE_MODEL)

result = run_baseline(sample, generate_fn, max_retries=3, env=env)
# {"predicted_sql": ..., "token_cost": ..., "attempts": ..., "success": ...}
# success = execution-correct (matched gold within max_retries), NOT string equality
```

- **`attempts`** — retry-loop count; backend-independent.
- **`token_cost`** — cumulative (input + output) tokens across all attempts.
  Pulled from the server's `completion.usage`; pass `fallback_tokenizer=` to
  `make_hf_api_generate_fn` if your provider omits usage stats.
- Chat replies wrapped in ```` ```sql ... ``` ```` fences are unwrapped before
  execution.

### With local inference (no API, no 504s)

Same Qwen-1.5B model, run on-device. No HF token needed; the first call to
`load_local_model` downloads the weights (~3 GB) into the HF cache
(`~/.cache/huggingface/hub/`), then loads them onto GPU.

```bash
pip install torch transformers      # one-time, if not already in your venv
```

```python
from baseline.full_regen import load_local_model, make_local_generate_fn, run_baseline
from config import LOCAL_MODEL, LOCAL_DTYPE, LOCAL_DEVICE

# First call: ~3 GB download from HF + load into GPU memory.
# Subsequent calls: cached, load is instant.
model, tokenizer = load_local_model(
    model_id=LOCAL_MODEL,    # 'Qwen/Qwen2.5-Coder-1.5B-Instruct'
    dtype=LOCAL_DTYPE,       # 'float16' (~3 GB VRAM) | 'bfloat16' | 'float32'
    device=LOCAL_DEVICE,     # 'auto' (recommended) | 'cuda' | 'cpu'
)
generate_fn = make_local_generate_fn(model, tokenizer)

result = run_baseline(sample, generate_fn, max_retries=3, env=env)
```

What `load_local_model` does under the hood:

```python
# from baseline/full_regen.py
tokenizer = AutoTokenizer.from_pretrained(model_id)
model     = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.float16, device_map='auto',
)
```

Both `from_pretrained` calls auto-download from the HF hub on first use.

- Adjust precision / device in [`src/config.py`](config.py) (`LOCAL_DTYPE`,
  `LOCAL_DEVICE`) — these are deliberately not CLI flags.
- `device='auto'` lets HF spill layers to CPU if VRAM is tight (good safety
  net on a 3050Ti laptop's 4 GB).
- Token counts come straight from the tokenizer / generated ids — no
  `fallback_tokenizer` needed.

### Adding another backend

Write a `generate_fn` (e.g. the PPO actor once inference exists) and pass it
straight to `run_baseline`.

---

## Configuration & secrets

Shared defaults (Spider path, execution oracle, baseline backbone) live in
[`src/config.py`](config.py). The HF token is **not** a constant or a CLI flag —
copy `.env.example` to `.env` and set `HF_TOKEN`; `config.py` loads it on import.
`.env` is gitignored.

## Evaluation driver (`scripts/evaluate.py`)

```bash
# API backend (default) — set HF_TOKEN in .env first
cp .env.example .env
python scripts/evaluate.py --split dev --max-retries 3 --max-samples 20

# Local backend — no API, no 504s. First run downloads ~3 GB weights.
python scripts/evaluate.py --split dev --backend local --max-samples 20
```

Runs the full-regen baseline over the split and prints the comparison table.
Precision / device for `--backend local` come from [`src/config.py`](config.py)
(`LOCAL_DTYPE`, `LOCAL_DEVICE`), not CLI flags.

`--ppo-ckpt` is accepted but the PPO path raises `NotImplementedError` until
Henry exposes an actor-loading inference entry point (see QUESTIONS.md).

---

## Running the tests

```bash
source venv/bin/activate
pip install pytest
python -m pytest tests/ -v
```

`test_env`, `test_metrics`, `test_baseline`, `test_evaluate` — all green, no
GPU and no API calls required (model/tokenizer/env/client are all faked).
