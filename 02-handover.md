# Handover Document (02) - 2026-05-12

## Summary of Changes

- Built complete `hermes_dqn/` Python package: `env/`, `agent/`, `training/`, `utils/`, `llm/` submodules (~1500 lines)
- Three OpenSpec changes completed and strict-valid:
  - `bootstrap-dqn-baseline` — vanilla DQN on LunarLander-v3, all 37 tasks done. Seed 42 converged at episode 399, mean reward last-100 = 262.79, success rate 95%, wall-time 24m47s on RTX 4090
  - `establish-project-lifecycle-spec` — governance spec via 3+1 multi-agent collaboration: 46 normative Requirements across 5 capabilities (doc-standards / env-setup / experiments-protocol / evaluation-criteria / final-deliverables)
  - `gemma-reward-generator` — Google Gemma 4 31B writes Python reward functions, AST-sandboxed compile with 3-retry, all 9 task groups done. Smoke 7/7 passed. 1500-ep training with Gemma reward finished in 16m29s (33% faster than baseline)
- `白話架構介紹.md` — plain-language architecture introduction using basketball-coach metaphor (13 sections, Mermaid + ASCII fallback, jargon translation table)
- `aichat_record/Claude實作起步/` — 5 session records documenting this work (~30 KB total)
- `tools/_eval_env_native.py` — apples-to-apples comparison helper (loads model, plays N greedy episodes on env-native reward)
- `hermes_dqn/training/play.py` — pygame visual playback of trained agents
- Security fix: `.env` was missing from `.gitignore` (only `env/` venv dir was) — added `.env` + `.env.local` + `*.env.local` patterns BEFORE the API key was pasted
- Existing `aichat_record/` files reorganized into `方向確立/` subfolder

## Current Status

- Three OpenSpec changes ready to archive (their specs will move to `openspec/specs/` permanently on archive)
- Apples-to-apples evaluation (100 unseen seeds 10000-10099, greedy, env-native reward) shows Gemma reward yields:
  - Mean env reward: 207.72 vs baseline 162.72 (+28%)
  - Success rate (>=200): 78% vs baseline 53% (+25 pp)
  - Crash rate (<0): 7% vs baseline 14% (halved)
- EUREKA open-source replication thesis (Gemma replacing GPT-4) holds for seed 42; multi-seed verification queued
- `runs/` directory contains: `baseline_seed42/`, `gemma_seed42/`, `smoke_env/`, `smoke_llm/`, plus three earlier smoke runs (all gitignored)

## Next Actions

- [ ] Archive completed changes: `openspec archive bootstrap-dqn-baseline establish-project-lifecycle-spec gemma-reward-generator`
- [ ] Propose `hermes-memory-layer` change — 4-tier memory (SQLite FTS5 long-term + working + short-context + procedural). This is the first change that introduces the "教練的筆記本" so future Gemma calls accumulate experience
- [ ] Eventually propose `ast-buffer-manager` and `closed-loop-fitness`
- [ ] Run 5-seed baseline + 5-seed Gemma comparison per `experiments-protocol` spec to make a statistical claim (n=1 result is suggestive, not conclusive)
- [ ] Migrate `pyproject.toml` + `requirements.txt` -> `uv.lock` per `env-setup` spec (separate small change)

## Open Questions

- `hermes-memory-layer`: use SQLite's built-in FTS5 full-text only, or add sentence-transformers embedding for semantic retrieval? FTS5-only is simpler and aligned with the spec; embedding is more powerful but adds dependency.
- Gemma temperature: lock to 0 for full reproducibility, or keep ~0.7 for diversity across iterations? `closed-loop-fitness` must decide. Current default in `google-genai` SDK was used (likely 0.7-1.0).
- License: governance spec defaulted to MIT; if course requires closed-source, add a declaration in `README.md` final section.
- Whether to commit pre-commit hooks now or wait for `env-setup` migration change. Currently strict-validate is enforced by reviewer only.
