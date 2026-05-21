# Open Questions & Decisions Log

Format: **[open]** / **[resolved]** question — answer

---

## Episode Initialization

**[open]** When CodeLlama's initial output is already correct, do we skip the episode or still run it?

**[open]** Does Henry own the prompt template for CodeLlama's SQL generation, or is it shared?

---

## Faulty Clause Identification

**[resolved]** Reward model scores each clause right after it is generated (not after full SQL). Score ∈ [0,1], lower = more likely wrong. No SQL execution involved — model confidence only.

**[open]** What happens when multiple clauses are wrong? Fix only the lowest-scoring one per episode, or iterate?

---

## Interfaces (see INTERFACES.md)

**[open]** Ian: does `parse_clauses()` return a simple dict `{clause_name: text}`?

**[open]** Henry/Sam: who owns the outer PPO training loop — does Henry's PPOTrainer call `env.step()`, or does Sam's pipeline call Henry's trainer?

---

## Reward Shaping

**[open]** Sparse reward (+1/0) may be insufficient — consider adding a small positive reward for queries that are executable but produce wrong results, to mitigate sparse reward.

**[open]** Partial credit for "close" wrong answers: how to define closeness? Options include per-clause F1, result-set overlap, or edit distance.

---

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-05 | Spider over BIRD | Simpler setup, standard baseline comparison |
| 2026-05 | Option B episode init | Training distribution matches inference |
| 2026-05 | Skip Option A warm-up | Not enough time; add as ablation if time permits |
| 2026-05 | 4000/3000 data split | Prevent reward model from leaking gold answers into PPO |
| 2026-05 | Reward model = dense per-clause signal | Clause scored independently after generation, float ∈ [0,1] |
| 2026-05 | Executor = sparse final signal | Sam's env.step() called once per episode with full SQL |
| 2026-05 | GPU: RTX 4090 | Henry's machine; determines VRAM budget (~24 GB) |
| 2026-05 | Model: CodeLlama-7B | Replaces Qwen2.5-Coder; fits RTX 4090 VRAM |
| 2026-05 | Reward shaping: +1/-1 | Partial credit and executable-but-wrong reward under consideration |