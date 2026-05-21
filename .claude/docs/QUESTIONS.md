# Open Questions & Decisions Log

Format: **[status]** question — answer (once resolved)

---

## Episode Initialization

**[open]** When Qwen's initial output is already correct, do we skip the episode or still run it?

**[open]** Do we use a fixed prompt template for Qwen's initial SQL generation, or does Henry own the prompt design?

---

## Faulty Clause Identification

**[open]** Does Henry's reward model expose `score_clauses(sql, db_id) -> dict[str, float]`, or does it return only the single worst clause?

**[open]** What happens when multiple clauses are wrong? Do we fix them one at a time (sequential episodes) or all at once?

---

## Interfaces (see INTERFACES.md)

**[open]** Ian: does `parse_clauses()` return a simple dict `{clause_name: text}` or a list with position metadata?

**[open]** Henry/Sam: who owns the outer PPO training loop — does Henry's PPOTrainer call `env.step()`, or does Sam's pipeline call Henry's trainer?

---

## Data

**[open]** Exact Qwen model version? (affects VRAM, tokenizer, prompt format)

**[open]** Ian: what evaluation metrics will the survey cover? EX only, or partial match per clause too?

---

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-05 | Spider over BIRD | Simpler setup, standard baseline comparison |
| 2026-05 | Option B episode init | Training distribution matches inference |
| 2026-05 | Skip Option A warm-up | Not enough time; add as ablation if time permits |
| 2026-05 | 4000/3000 data split | Prevent reward model from leaking gold answers into PPO |
