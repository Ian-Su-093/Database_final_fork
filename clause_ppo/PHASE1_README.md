# CLAUSE-PPO Phase 1 — Process Reward Model

## Running on a Windows/Linux PC with RTX 4090

This project was developed on a Mac. Training requires a CUDA GPU (RTX 4090 target).
Follow these steps on your PC:

### 1. Transfer the code

Option A — Git (recommended):
```bash
git clone <your-repo-url>
cd Database_final
```

Option B — Copy manually: transfer the entire `Database_final/` folder to the PC.

### 2. Get the Spider dataset on the PC

The `clause_ppo/data/spider` symlink only works on the Mac.
On the PC, download Spider directly into the right place:

```bash
pip install gdown
cd Database_final/clause_ppo
gdown "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J" -O spider.zip
unzip spider.zip
mv spider_data data/spider          # rename to match expected path
rm spider.zip
```

Verify: `ls data/spider/` should show `train_spider.json`, `dev.json`, `tables.json`, `database/`.

### 3. Set up the Python environment

```bash
cd Database_final/clause_ppo
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

> **Windows note:** `bitsandbytes` requires CUDA to be installed first.
> Install CUDA Toolkit 11.8 or 12.1 from https://developer.nvidia.com/cuda-downloads
> before running `pip install -r requirements.txt`.

### 4. Verify CUDA is visible

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected output: `True` and `NVIDIA GeForce RTX 4090`.

### 5. Accept the CodeLlama model license

The training script downloads `codellama/CodeLlama-7b-hf` from HuggingFace on first run.
You must accept the license first:
1. Go to https://huggingface.co/codellama/CodeLlama-7b-hf
2. Click "Access repository" and accept the terms
3. Log in from the terminal: `huggingface-cli login` (paste your HF token)

---

## Step 1 — Build the Corruption Dataset

Run this from `clause_ppo/` on the PC (CPU is fine, ~10 min):

```bash
python scripts/build_corruption_dataset.py \
    --spider_dir  data/spider \
    --output_dir  data/processed \
    --split       train \
    --max_examples -1 \
    --min_clauses  2 \
    --log_every    200
```

**OR** copy the already-built files from your Mac to save time:

```bash
# On Mac, from Database_final/
scp clause_ppo/data/processed/*.json user@pc-ip:~/Database_final/clause_ppo/data/processed/
```

Expected outputs in `data/processed/`:
- `corruption_dataset.json` — verified corruption records
- `original_dataset.json`   — filtered originals with pre-computed clause splits
- `corruption_stats.json`   — per-clause pass rates and counts

**Diagnostic:** Check `corruption_stats.json`. Healthy pass rates are 0.1–0.6 per
clause. Near-zero means `reconstruct_sql` is broken; near-one means corruptions are
trivially equivalent.

## Step 1 — Build the Corruption Dataset

```bash
cd clause_ppo
python scripts/build_corruption_dataset.py \
    --spider_dir  data/spider \
    --output_dir  data/processed \
    --split       train \
    --max_examples -1 \
    --min_clauses  2 \
    --log_every    200
```

Expected outputs in `data/processed/`:
- `corruption_dataset.json` — verified corruption records
- `original_dataset.json`   — filtered originals with pre-computed clause splits
- `corruption_stats.json`   — per-clause pass rates and counts

Runtime: ~10 minutes on CPU for full 7,000 training examples.

**Diagnostic:** Check `corruption_stats.json`. Healthy pass rates are 0.1–0.6 per
clause. Near-zero means `reconstruct_sql` is broken; near-one means corruptions are
trivially equivalent.

## Step 2 — Train the PRM

```bash
python scripts/train_prm.py \
    --config        configs/prm_config.yaml \
    --spider_dir    data/spider \
    --processed_dir data/processed
```

Training logs: `results/prm_training_log.jsonl` (one JSON per eval step).
Best checkpoint: `results/prm_checkpoints/best_checkpoint/`.
Epoch checkpoints: `results/prm_checkpoints/epoch_N/`.

Expected time: ~6–8 hours for 3 epochs on an RTX 4090.

## Step 3 — Run Inference

After training, score any clause prefix with the trained PRM.

### Input format

The model expects inputs in the same format used during training:

```
[QUESTION] {question} [SCHEMA] {schema} [PREFIX] {prefix_sql}
```

- **question**: the natural language question from Spider
- **schema**: table/column schema string (e.g. `singer(singer_id, name, age), concert(concert_id, theme)`)
- **prefix_sql**: a partial SQL query up to and including the clause being scored (e.g. `SELECT count(*) FROM singer`)

The model returns a score in **(0, 1)**: higher means the prefix is more likely correct.

### CLI

```bash
cd clause_ppo
python scripts/score_clause.py \
    --checkpoint results/prm_checkpoints/best_checkpoint \
    --base_model /home/henrylin0822/models/qwen \
    --question "How many singers are there?" \
    --schema "singer(singer_id, name, age)" \
    --prefix "SELECT count(*) FROM singer"
```

Output:
```
Score: 0.8123  (likely correct)
```

### Python API

```python
import sys
sys.path.insert(0, 'src')
from models.prm_inference import PRMScorer

scorer = PRMScorer(
    checkpoint_dir='results/prm_checkpoints/best_checkpoint',
    base_model='/home/henrylin0822/models/qwen',
)

# Single example
score = scorer.score(
    question="How many singers are there?",
    schema="singer(singer_id, name, age)",
    prefix_sql="SELECT count(*) FROM singer",
)
print(score)  # e.g. 0.812

# Batch
scores = scorer.score_batch([
    ("How many singers?", "singer(singer_id, name)", "SELECT count(*) FROM singer"),
    ("How many singers?", "singer(singer_id, name)", "SELECT name FROM concert"),  # corrupted
])
print(scores)  # e.g. [0.81, 0.23]
```

### Checkpoint contents

Each checkpoint directory contains:

| File | Description |
|------|-------------|
| `adapter_config.json` | LoRA adapter config |
| `adapter_model.safetensors` | LoRA weights |
| `score_head.pt` | Regression head weights |
| `tokenizer.json` / `tokenizer_config.json` | Tokenizer |

> **Note:** `score_clause.py` has not been run end-to-end yet — treat as untested until a full training run completes and produces a `best_checkpoint`.

---

## Expected Outputs

| File | Description |
|------|-------------|
| `data/processed/corruption_dataset.json` | Verified corruption records |
| `data/processed/original_dataset.json` | Original examples with clause splits |
| `data/processed/corruption_stats.json` | Pipeline diagnostics |
| `results/prm_training_log.jsonl` | Per-eval-step metrics (dev_bce, score_gap) |
| `results/prm_checkpoints/best_checkpoint/` | Best PRM weights + tokenizer |
| `results/prm_checkpoints/epoch_N/` | End-of-epoch checkpoints |

## Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| `dev_bce` | Binary cross-entropy on dev set | Lower is better |
| `score_gap` | mean(V_phi\|correct) − mean(V_phi\|corrupt) | > 0.3 after training |

## Component Overview

| Module | Description |
|--------|-------------|
| `src/utils/execution.py` | SQLite oracle with daemon-thread timeout |
| `src/utils/sql_utils.py` | Spider sql dict → executable SQL string |
| `src/data/clause_splitter.py` | Clause splitting in execution order |
| `src/data/corruption.py` | Rule-based SQL corruption engine |
| `src/data/dataset.py` | PRMDataset with cascade labeling + schema dropout |
| `src/models/prm.py` | Qwen2.5-Coder 4-bit QLoRA + LoRA + regression head |
| `src/models/prm_inference.py` | PRMScorer: load checkpoint and score prefixes |
| `src/training/train_prm.py` | BCE + schema-grounding training loop |
| `scripts/score_clause.py` | CLI for scoring a single clause prefix |

## Corruption Dataset Statistics (100-example smoke test)

```
Per-clause verification pass rates:
  from          attempts=100  verified=96   pass_rate=0.960
  where         attempts=39   verified=39   pass_rate=1.000
  select        attempts=100  verified=88   pass_rate=0.880
  groupBy       attempts=23   verified=17   pass_rate=0.739
  having        attempts=4    verified=4    pass_rate=1.000
  orderBy       attempts=32   verified=17   pass_rate=0.531
```

## Known Limitations and Paper Deviations

1. **[CLS] → last token:** The paper uses [CLS] notation from BERT-style encoders.
   CodeLlama is decoder-only with no [CLS]. We use the last non-padding token hidden
   state, the standard decoder-model equivalent. See `src/models/prm.py`.

2. **Approximate schema grounding loss:** Schema-grounding loss decodes token IDs
   back to text per batch (correct but slow). In production, store raw texts in the
   batch dict instead. Disabled on CPU during development.

3. **Localisation Accuracy proxy:** Exact LA requires grouping prefix states by
   example and computing argmin_j V_phi(s_j). The training loop reports `score_gap`
   (mean score difference) as a proxy. Implement exact LA evaluation in Phase 2.

4. **Corruption schema placeholders:** Corruption records in `corruption_dataset.json`
   use a placeholder schema `[schema for {db_id}]` rather than the full schema string,
   because corruption records don't store the full prefix_states. This is acceptable
   for Phase 1 but can be improved in Phase 2 by linking back to `original_dataset.json`.

5. **`reconstruct_sql` limitations:** Handles 6 main clauses + LIMIT + set operations.
   JOINs are simplified to comma-separated tables with merged WHERE conditions. Output
   may differ syntactically from gold SQL but is semantically equivalent for oracle use.

6. **`corrupt_where` strategies 2 and 3** are structurally unreachable given the
   current deterministic ordering (strategy 1 always fires for non-empty WHERE).
   These branches are preserved for spec completeness and future refactoring.
