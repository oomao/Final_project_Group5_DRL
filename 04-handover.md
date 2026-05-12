# Handover Document (04) - 2026-05-12

## Summary of Changes

This session continued from handover 03 and added two major OpenSpec changes:

### `ast-buffer-manager` (strict-valid, applied)

- 4 new files in `hermes_dqn/buffer/`: `ast_diff.py` (RewardDiff classifier with 4 kinds), `policy.py` (BufferAction enum + decide_policy), `rebuild.py` (apply_policy mutator), `__init__.py` (re-exports)
- `ReplayBuffer` extended with: per-sample `_weights`, sample fast-path (when all weights == 1.0, byte-identical to legacy uniform sampling), `decay_weights(factor)`, `clear()`, `save(path)` / `load(path)` with full RNG-state roundtrip
- 1 capability spec (`ast-buffer-manager` with 3 ADDED Requirements) + `dqn-baseline` MODIFIED (added Requirement "ReplayBuffer persistence and reset" plus scenarios on uniform fast-path determinism and save/load round-trip)
- 27/27 unit cases pass via `tools/_smoke_ast_buffer.py` (incl. decay sampling 1/3 ratio = 0.327 observed vs 0.333 expected)
- Backward-compat verified: env-path 10-ep deterministic vs `runs/baseline_seed42`

### `closed-loop-fitness` (strict-valid, applied)

- `hermes_dqn/training/closed_loop.py`: 7-step iteration loop (memory → Gemma → AST diff → buffer policy → train → env-native eval → memory write)
- `hermes_dqn/training/summary.py`: `IterationSummary` + `ClosedLoopSummary` dataclasses with JSON serialization
- `train.py` extended with optional `pre_loaded_buffer` and `pre_resolved_reward` params; auto-saves `buffer.npz` after every run; backward-compat preserved
- `tools/compare_conditions.py`: pairwise Mann-Whitney U + 5000-bootstrap 95% CI + 3-condition win rule + markdown report + 2 PNG figures (training_curves, iteration_fitness)
- 2 capability specs (closed-loop-engine, condition-comparison) with ~20 verifiable scenarios
- Pilot smoke run: 3 iterations × seed 42 × 1500 ep, total 52.5 min

### Pilot results (1 seed × 3 iter, mechanism verification only)

| Iter | Priors | AST diff | Buffer action | env_native_mean | Success | Crash |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `[]` | — | — | 181.33 | 65% | 22% |
| 2 | `[1]` | STRUCTURAL_DIFF (sim 0.71) | DECAY | 90.35 | 12% | 12% |
| 3 | `[1, 2]` | TOTAL_REWRITE (sim 0.56) | CLEAR | 168.54 | 25% | 2% |

**All four mechanisms confirmed working end-to-end:**
- Memory: iter 2 correctly received iter 1's reward as prior; iter 3 received both
- AST diff: similarity 0.71 (boundary case) classified as STRUCTURAL_DIFF; 0.56 as TOTAL_REWRITE
- Buffer policy: DECAY scaled weights × 0.5 in iter 2; CLEAR emptied buffer in iter 3
- Reproducible artifacts: every iter has config.json + episodes.jsonl + model_final.pt + reward_fn.py + llm_attempts.jsonl + buffer.npz

**Empirical signal too noisy at n=1**: env_native_mean non-monotonic (181 → 90 → 168). Only crash_rate is monotonic (22% → 12% → 2%), consistent with earlier "Gemma trades reward for safety" observation. Cannot conclude memory helps or hurts — n=5 + Mann-Whitney + bootstrap CI per `evaluation-criteria` spec is needed.

## Current Status

- **7 OpenSpec changes strict-valid**:
  - `improve-dev-scripts`, `bootstrap-dqn-baseline`, `establish-project-lifecycle-spec`, `gemma-reward-generator`, `hermes-memory-layer` (applied)
  - `ast-buffer-manager`, `closed-loop-fitness` (this session, applied)
  - `reward-sandbox-isolation` (proposal-only, triggers documented)
- **`hermes_dqn/` package is feature-complete** for the three core contributions (open Gemma reward, 4-tier memory's long-term layer, AST-aware buffer)
- **Full statistical run is set up but not executed** ── `compare_conditions.py` works on the 1-condition pilot data, ready to scale to 6 conditions × 5 seeds when data exists
- **Pilot artifacts**: `runs/pilot/B3-pilot/seed_42/` with 3 iter dirs + `summary.json` + per-condition `memory.sqlite`
- **Pilot report**: `reports/pilot/comparison_report.md` + `figures/training_curves.png` + `iteration_fitness.png`

## Next Actions

- [ ] Archive the 7 strict-valid changes: `openspec archive ast-buffer-manager closed-loop-fitness ...`. specs move to `openspec/specs/` permanently
- [ ] **Plan the experiment week**:
  - Write B1 hand-shaped reward (per `evaluation-criteria` spec, by non-author third party)
  - Define exp_name = "final" and run all 6 conditions × 5 seeds × 5 iterations
  - Estimated compute: ~60 GPU-hours, plan ~4-5 days of 4090 time with overnight runs
  - Use `runs/4090-booking.md` per `env-setup` spec for multi-person coordination
- [ ] After data collection: `python tools/compare_conditions.py --exp final --conditions B0,B1,B2,B3,B3-no-memory,B3-no-AST` produces paper Table 1
- [ ] Write paper §4 Experiments + §5 Discussion using the report
- [ ] Update demo video (current YouTube `b4ad_7xtydk` is v1; v2 should show closed-loop in action)
- [ ] (Optional, future) `reward-sandbox-isolation` apply ── only if supervisor/team triggers per its proposal

## Open Questions

- **Why does iter 2 underperform iter 1 in the pilot?** Possible explanations: (a) Gemma's stochastic output dominates at n=1, (b) prior reward biases Gemma toward overly aggressive shaping, (c) DECAY buffer is too aggressive at 0.5 — old samples may still help more than this assumes. Resolve via experiment-week ablation: try decay_factor ∈ {0.1, 0.5, 0.9} and compare.
- **Buffer fast-path determinism vs DECAY usage**: when closed_loop_fitness is used, the buffer's `_weights` ≠ 1.0 means we leave the deterministic fast-path. Multi-seed runs of the same condition should still be reproducible within their non-uniform-sampling path, but baseline determinism guarantees are scoped to runs that never invoke `decay_weights`.
- **Should `lessons_learned` Gemma reflection be enabled before the formal 5-seed run?** Currently disabled to save API quota. If enabled, every iteration adds 1 more Gemma call. Total extra cost: 5 cond × 5 seed × 5 iter × 1 call = 125 extra API calls. Free tier 15 RPM accommodates this trivially. Recommendation: enable for the final run to give Gemma more context in subsequent iterations.
- **decay_factor=0.5 lacks theory.** Treat as an ablation knob; final paper Table 1 can show the comparison.
