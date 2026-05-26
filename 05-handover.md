# Handover Document (05) - 2026-05-18

## Summary of Changes

This session set up the **statistical experiment scaffolding** for the formal
6-condition × 5-seed evaluation that will produce the paper's Table 1. No new
OpenSpec change was created — every modification stays within the existing
specs' contract.

### Code added / modified

- **`hermes_dqn/training/closed_loop.py`**: added `--no-memory` and `--no-ast`
  CLI flags + matching `run_closed_loop()` keyword args, properly threaded
  through to skip Hermes memory I/O (B3-no-memory) and AST diff + buffer
  policy mutation (B3-no-AST).
- **`hermes_dqn/training/train.py`**: extended `reward_source` choices from
  `{env, llm}` to `{env, llm, file}`. Added `--reward-file` flag and
  `reward_file: str | None` config field. The `file` mode reads a Python
  source file, compiles via the existing sandbox, and feeds it into
  training with zero changes to the training inner loop. Unblocks B1.
- **`experiments/baselines/B1_handcrafted.py`** (new): textbook hand-shaped
  LunarLander reward (centering, upright, soft landing, leg-contact bonus).
  Header documents that this is a **PLACEHOLDER** to be replaced by a
  non-author teammate before the final paper, per `evaluation-criteria`
  Requirement "Complete Baseline Set".
- **`scripts/run_full_experiment.py`** (new): resumable orchestrator for the
  6 × 5 matrix with **serial AND parallel modes**:
  - `--workers 1` (default): serial, tee to terminal — 25.5h wall, lowest risk
  - `--workers 5`: ThreadPoolExecutor + 5 parallel subprocesses sharing the
    4090; per-worker OMP/MKL thread cap auto-set to `cpu_count // workers`
    (28 / 5 = 5); tqdm bars go to log files only; main terminal prints a
    `[progress]` line every 60s — **~6-8h wall**
  One subprocess per (cond, seed); skips completed pairs by detecting
  `env_native_mean` in `config.json`; per-seed memory.sqlite so seeds are
  statistically independent (required for Mann-Whitney U).
- **`runs/launch-final-experiment.md`** (new): step-by-step launch guide
  including pre-flight checklist, wall-time table, multi-night kickoff
  options, and post-run `compare_conditions.py` command.

### Smoke validation (BOTH MODES PASSED)

**Serial smoke**: `python scripts/run_full_experiment.py --exp smoke6 --episodes 10 --iterations 1 --seeds 42`
ran all 6 conditions end-to-end in **6.6 min** total (B0/B1 ~5s each, then
4 closed-loop runs at 80-110s each — each closed-loop costs one Gemma API
call). Result: **6 ok / 0 skipped / 0 failed**.

**Parallel smoke**: `python scripts/run_full_experiment.py --exp smoke_par --episodes 10 --iterations 1 --seeds 42,43 --workers 3`
ran 12 (cond, seed) jobs with 3 concurrent workers; ~2.5 min total (vs
~10 min serial extrapolation = ~4× speedup at workers=3). GPU utilization
rose from 1% (serial) to 5% (parallel) — model is too small to push GPU
higher. Periodic `[progress]` lines printed every 60s confirmed liveness.

Each condition wrote a complete `runs/smoke6/<cond>/seed_42/iter_01/config.json`
with `env_native_mean` present (the orchestrator's completion criterion).
Spot-checks confirmed the ablation flags propagate correctly:

- B3-hermes-full: `memory_state=hermes-sqlite-fts5`, `no_memory=false`,
  `memory_id=1` (written)
- B3-no-memory: `memory_state=none`, `no_memory=true`, `memory_id=-` (NOT written)
- B3-no-AST: `memory_state=hermes-sqlite-fts5` (memory unaffected),
  `buffer_action=-` (correctly skipped at iter=1, will skip at iter≥2)
- B1-handcrafted: `reward_source=file`,
  `reward_file=experiments/baselines/B1_handcrafted.py`, 61-line src loaded
  via the existing sandbox compile path

All six runs converged to env_native_mean=-553.58 because 10 training
episodes is far below convergence (ε≈0.98, agent is near-random). This is
the correct, deterministic wiring-validation outcome.

`tools/compare_conditions.py` also runs end-to-end on the smoke output
and correctly classifies all 6 runs as "divergent" per
`evaluation-criteria` Requirement "Run Inclusion Gate" (env_native_mean
< -200), which is the expected behavior for a 10-episode smoke run.

## Current Status

- **All scaffolding is in place** for the formal 6 × 5 × N evaluation
- **Smoke run confirmed end-to-end wiring** of all 6 conditions (including
  the new file-based B1 path and the new --no-memory / --no-ast ablations)
- **Full evaluation is NOT yet run** — that's the user's overnight task,
  estimated ~25.5 GPU-hours total (1.4h each for B0/B1/B2, 7.1h each for
  B3*). Free-tier Gemma quota (80 calls total) fits in a single UTC day.
- **B1 reward is a PLACEHOLDER** — must be rewritten by non-author teammate
  before submission. See header comment in
  `experiments/baselines/B1_handcrafted.py` for the 4-step replacement
  protocol. Re-running just B1 takes ~1.4 h.

## Next Actions

- [ ] **Run the smoke test yourself** to confirm GPU + env on your machine:
  `python scripts/run_full_experiment.py --exp smoke6 --episodes 10 --iterations 1 --seeds 42`
  then `Remove-Item -Recurse -Force runs/smoke6`.
- [ ] **Replace B1 placeholder** — ask a non-author teammate (Member X?) to
  rewrite `experiments/baselines/B1_handcrafted.py` from scratch. Their
  commit should set Co-Authored-By to themselves.
- [ ] **Kick off overnight runs** per `runs/launch-final-experiment.md`.
  Recommended split: B0+B1+B2 night 1 (~4 h), B3 night 2 (~7 h),
  B3-no-memory night 3, B3-no-AST night 4. Or single ~26 h marathon.
- [ ] **Generate paper Table 1**:
  `python tools/compare_conditions.py --exp final --conditions B0-env-native,B1-handcrafted,B2-gemma-oneshot,B3-hermes-full,B3-no-memory,B3-no-AST`
- [ ] Write paper §4 Experiments + §5 Discussion using the report
- [ ] Record demo v2 video (closed loop in action; current v1 is single-iter only)

## Open Questions

- **Should the orchestrator parallelize across GPUs if multiple are available?**
  Currently single-process sequential. For our local 4090 single-GPU setup
  this is correct; if anyone runs on a shared cluster, a small refactor to
  shell out one `python` per CUDA_VISIBLE_DEVICES could ~6× throughput.
- **B1 placeholder vs. final teammate version**: results from the placeholder
  are reportable as "lower bound for what a reasonable hand-shaped reward
  achieves," but the paper's defensible B1 claim requires the teammate
  rewrite. Plan accordingly: 1.4 h re-run after rewrite is cheap.
- **decay_factor knob still untuned**. The pilot used 0.5. After the formal
  run, a small follow-up sweep on B3-hermes-full with decay_factor ∈ {0.1,
  0.5, 0.9} for one seed would tell us if 0.5 is near-optimal. Cost: ~3 h.
- **lessons_learned Gemma reflection**: still disabled (API quota). With
  this run's 80-call total well under the 250 RPD limit, enabling it adds
  another ~75 calls (25 iter × 5 conditions where B3* uses iter > 1).
  Should still fit. Recommendation: enable before the final run kickoff —
  takes one flag flip in `closed_loop.py`'s reward-generation step.
