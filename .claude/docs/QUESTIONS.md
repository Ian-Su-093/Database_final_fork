# Open Questions & Decisions Log

---

## Open

**[open]** Ian: `run_baseline()` prompt format — must match `build_rewrite_prompt()` style
in `ppo_loop.py` for a fair comparison. Confirm with Henry what the prompt looks like.

**[open]** `evaluate.py`: how to run RL inference at eval time?
Does Henry expose a function like `run_ppo_inference(sample, model, tokenizer)`
or does Sam call `ppo_trainer.generate()` directly?

**[open]** Multiple wrong clauses: `get_corrupted_sample()` corrupts exactly one clause.
What if Qwen generates SQL with multiple wrong clauses? (Less urgent — corruption engine
handles training, Qwen output only appears at eval time.)

---

## Resolved

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-05 | Spider over BIRD | Simpler setup, standard baseline |
| 2026-05 | Data split: train[0:4000] PRM, train[4000:] PPO | Prevent reward model leaking into PPO |
| 2026-05 | Episode init: corruption engine, NOT Qwen output | ppo_loop.py confirmed — get_corrupted_sample() |
| 2026-05 | env.step() takes full SQL, not clause text | Confirmed from ppo_loop.py line 332 |
| 2026-05 | Henry owns entire PPO loop | ppo_loop.py is self-contained, Sam is not the outer loop |
| 2026-05 | Reward = terminal + alpha * prm_score | compute_reward() in ppo_loop.py |
| 2026-05 | GPU: RTX 4090, WSL2 | Henry's machine |
| 2026-05 | Baseline: full query regeneration, max_retries configurable | Ian implements run_baseline() |
| 2026-05 | Metric: Accuracy@N + avg token cost (input+output) | evaluate.py outputs comparison table |