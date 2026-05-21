# CLAUSE-PPO Phase 1 — Process Reward Model

## Prerequisites

```bash
pip install -r requirements.txt
```

Requires an NVIDIA RTX 4090 (24 GB VRAM) or equivalent for model training.
Dataset exploration and corruption building run on CPU.

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
| `src/models/prm.py` | CodeLlama-7B 4-bit QLoRA + LoRA + sigmoid head |
| `src/training/train_prm.py` | BCE + schema-grounding training loop |

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
