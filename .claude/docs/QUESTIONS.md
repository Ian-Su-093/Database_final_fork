# Open Questions & Decisions Log

---

## Open

**[open]** `evaluate.py`: how to run RL inference at eval time?
Does Henry expose a function like `run_ppo_inference(sample, model, tokenizer)`
or does Sam call `ppo_trainer.generate()` directly? `run_clause_ppo()` in
`evaluate.py` raises NotImplementedError until this exists.

**[open]** Multiple wrong clauses: `get_corrupted_sample()` corrupts exactly one clause.
What if the model generates SQL with multiple wrong clauses? (Less urgent — corruption
engine handles training, model output only appears at eval time.)

**[deferred]** Self-correct baseline (prior-wrong-SQL feedback).
Currently every retry sees the same prompt — purely independent samples
(pass@N) with explicit `TEMPERATURE`. Plan:
  1. Run PPO vs the current temperature-pinned baseline.
  2. If PPO clearly wins, raise the bar: switch the baseline to include the
     previous wrong SQL in the prompt ("the prior query returned wrong
     results, try a different approach"). That baseline is expected to score
     *higher*, so beating it is a stronger result for PPO.


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
| 2026-05 | Baseline: full query regeneration, max_retries configurable | run_baseline() in src/baseline/full_regen.py |
| 2026-05 | Baseline backbone: Qwen2.5-Coder-1.5B via HF Inference API | Teammate's API caller; cheap remote baseline vs trained CodeLlama-7B — comparison mixes model-size + method, accepted as a deliberate framing |
| 2026-05 | run_baseline takes injected generate_fn, not (model, tokenizer) | Decouples retry loop from backend; API + local both satisfy `prompt -> (sql, n_in, n_out)` |
| 2026-05 | Baseline prompt format = build_baseline_prompt (mirrors build_rewrite_prompt) | [QUESTION][SCHEMA][SQL] header matches the actor for comparable input tokens |
| 2026-05 | Token cost from server completion.usage (fallback: local tokenizer) | InferenceClient returns prompt/completion tokens; fail loudly if neither available |
| 2026-05 | Metric: Accuracy@N + avg token cost (input+output) | evaluate.py outputs comparison table |
| 2026-05 | run_baseline returns `success: bool` (execution correctness) | Per-sample log was using string equality — wrong. The retry loop already knows the truth from env.step(); surfaced it explicitly. |
| 2026-05 | Adapter retries transient API errors (5xx/429/timeouts) with backoff | One 504 was killing the whole eval run; retry network-level failures, fail loud on 4xx. Config: API_RETRIES, API_BACKOFF_SECS. |
| 2026-05 | Pin baseline TEMPERATURE explicitly in config (default 0.7) | Provider defaults are opaque — Accuracy@N retries only mean pass@N if sampling is documented, not inherited. |
| 2026-06 | Baseline backend: option (B) local inference implemented | `--backend local` runs Qwen2.5-Coder-1.5B locally via `load_local_model` + `make_local_generate_fn`; `--backend api` uses Featherless AI via HF token. Both satisfy the `generate_fn` contract. |
| 2026-06 | Plan B (ClausePRM + Best-of-N) added as third eval method | No PPO training needed; PRM identifies faulty clause, LLM generates N repairs, oracle selects first correct one. Shares `MAX_TOKENS`/`TEMPERATURE`/`MAX_RETRIES` from `config.py`. |