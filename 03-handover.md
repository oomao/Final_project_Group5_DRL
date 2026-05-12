# Handover Document (03) - 2026-05-12

## Summary of Changes

- **`hermes-memory-layer` change implemented and strict-valid**:
  - 4 capability specs (memory-store / memory-llm-integration / reward-sandbox / llm-reward-client MODIFIED) covering 19 Requirements with verifiable scenarios
  - 6 new files (`hermes_dqn/memory/{entry,schema,store,__init__}.py`, `hermes_dqn/llm/sandbox.py`, `hermes_dqn/training/eval_env_native.py`)
  - 5 modified files (`hermes_dqn/llm/compile.py` refactored; `prompts.py`, `client.py`, `__init__.py`, `training/train.py` extended)
  - L2 reward sandbox: LLM-generated code is now validated in a hard-killable `multiprocessing.Process` (spawn mode, Linux + Windows). After subprocess validation passes, the source is re-compiled inline in the main process so training-loop calls pay zero IPC cost
  - Long-term memory: SQLite FTS5 backend with WAL mode, `UNIQUE(reward_fn_sha256)` upsert, `top_k_by_fitness` with `COALESCE(env_native_mean, mean_reward_last100)` ordering
  - Inline env-native evaluation: every training run now ends with 100-seed greedy playback against env-native reward, results stored in `config.json`
- **`reward-sandbox-isolation` change proposed (proposal-only, future track)**:
  - Dockerfile + docker-compose + wrapper scripts for L3 container-level isolation
  - Strict-valid; tasks.md task group 1 is "wait for trigger condition"
  - Trigger conditions documented in `proposal.md`; this change MUST NOT be applied until one fires
- **Smoke tests**: sandbox 5/5 (valid, syntax, import, infinite-loop, arity) + training-memory 5/5 (env deterministic, llm first-write, llm second-read, env-native eval inline, --no-memory no-write)
- **1500-ep × 2 verification runs**:
  - `gemma_mem_seed42` (empty memory at start): env_native_mean=235.21, success=80%, crash=3%
  - `gemma_mem_seed43` (read seed 42 as prior): env_native_mean=224.53, success=78%, crash=3%, `memory_priors_used=[1]` confirmed
- **README updated**: baseline runs table now has 4 rows (env / llm-no-memory / llm+mem 1st / llm+mem 2nd) plus n=1 caveats

## Current Status

- 5 OpenSpec changes are strict-valid and tasks-complete:
  - `improve-dev-scripts` (earlier session)
  - `bootstrap-dqn-baseline`
  - `establish-project-lifecycle-spec`
  - `gemma-reward-generator`
  - `hermes-memory-layer` (this session)
- 1 OpenSpec change is proposal-only (intentionally): `reward-sandbox-isolation`
- Two pending feature changes ahead: `ast-buffer-manager`, `closed-loop-fitness`
- `runs/memory.sqlite` exists with 2 production entries (1500-ep), `runs/smoke_mem.sqlite` exists with 3 smoke entries (10-ep)
- All trained models in `runs/` are gitignored; SHA-256 / config / reward_fn.py / llm_attempts.jsonl are committed via run dirs only when explicitly added (which we do NOT do — runs/ is fully ignored)

## Next Actions

- [ ] Archive the 5 strict-valid changes: `openspec archive bootstrap-dqn-baseline establish-project-lifecycle-spec gemma-reward-generator hermes-memory-layer` (improve-dev-scripts is older). Their specs move to `openspec/specs/` permanently
- [ ] Propose `ast-buffer-manager` change: AST diff between consecutive rewards + replay buffer policy (keep / decay / clear) based on diff kind. This addresses the "catastrophic forgetting when reward changes" thesis from README's three core contributions
- [ ] Propose `closed-loop-fitness` change: 7-step outer loop + 5-seed comparison across all 6 conditions (B0/B1/B2/B3/B3-no-memory/B3-no-AST) + Mann-Whitney U + bootstrap CI per `experiments-protocol` spec
- [ ] At some point: migrate `pyproject.toml` + `requirements.txt` → `uv.lock` per `env-setup` Requirement "Environment Manager" (left as a small future change)
- [ ] If course supervisor wants Docker mode: apply `reward-sandbox-isolation` change (currently dormant)

## Open Questions

- **Does Hermes memory actually help at n=5+?** Single-seed results in this session show the *mechanism* works (priors loaded, prompts contain them, fitness written back) but the numbers are within noise. Statistical claim awaits `closed-loop-fitness` running all 6 conditions × 5 seeds and bootstrap-CI compare
- **Should `lessons_learned` be enabled by default?** Currently disabled (saves Gemma API quota). Per design.md it's a `--memory-with-lessons` opt-in. Decide before `closed-loop-fitness` apply
- **What's the right `fitness_floor` for production?** Spec says `0.0` (exclude crashes). Implementation overrides to `-inf` for early-stage runs so undertrained entries surface as priors. Once we have ≥ 10 production runs with positive fitness, can switch back to spec default
- **5-seed default `[42, 43, 44, 45, 46]` is shared with training seeds in `bootstrap-dqn-baseline`.** Should multi-condition experiments use a different seed range to avoid double-counting? Probably yes; `closed-loop-fitness` decides
