## 1. Memory package skeleton

- [x] 1.1 Created `hermes_dqn/memory/` package with `__init__.py`
- [x] 1.2 Implemented `MemoryEntry` dataclass (11 fields, optional default `None`)
- [x] 1.3 Implemented `to_dict()` / `from_dict()` with forward-compat unknown-key filtering
- [x] 1.4 Re-exported from `hermes_dqn/memory/__init__.py`

## 2. SQLite schema

- [x] 2.1 `apply_schema(conn)`: table + 2 indexes + FTS5 virtual table + AFTER INSERT trigger, all `IF NOT EXISTS`
- [x] 2.2 `current_version()` reads `PRAGMA user_version`; `migrate()` placeholder for v1
- [x] 2.3 `PRAGMA journal_mode=WAL` set on schema apply
- [x] 2.4 `_fts5_available()` probes; raises `RuntimeError` if absent

## 3. MemoryStore class

- [x] 3.1 `MemoryStore(db_path)` idempotent schema apply on open
- [x] 3.2 `write(entry)` uses `INSERT ... ON CONFLICT(reward_fn_sha256) DO UPDATE RETURNING id` + manual FTS resync
- [x] 3.3 `top_k_by_fitness` with 3 order_by modes + fitness_floor filter; default uses `COALESCE(env_native_mean, mean_reward_last100)`
- [x] 3.4 `all_count()` returns SELECT COUNT(*)
- [x] 3.5 Context manager + idempotent close
- [x] 3.6 Use-after-close raises `RuntimeError`

## 4. Extract env-native eval into package

- [x] 4.1 Created `hermes_dqn/training/eval_env_native.py::evaluate_on_env_native(run_dir, n=100, base_seed=10000)`
- [x] 4.2 Returns dict with env_native_{mean, median, success, crash_rate, mean_length}
- [x] 4.3 `tools/_eval_env_native.py` left as-is (still works); inline version is preferred for new code

## 5. Prompt template extension

- [x] 5.1 `build_lunarlander_prompt(..., prior_attempts=None)` signature extended
- [x] 5.2 `_format_prior_attempts` renders fenced Python code + fitness line + optional Lessons
- [x] 5.3 Block placement: after task_spec, before _RESPONSE_FORMAT (which is before few-shot)
- [x] 5.4 Backward-compat: prior_attempts None or [] produces no PRIOR section

## 6. LLMRewardClient.generate() signature change

- [x] 6.1 `generate(task_spec, attempts_log_path, memory)` — memory defaults `None`
- [x] 6.2 Forwards `memory or None` to `build_lunarlander_prompt(prior_attempts=...)`
- [x] 6.3 Verified: existing gemma-reward-generator smokes still pass (env path 10-ep matches baseline byte-for-byte)

## 7. train.py integration

- [x] 7.1 CLI: `--memory-db`, `--memory-top-k`, `--no-memory`, `--unsafe-inline-compile`, `--eval-n-episodes`
- [x] 7.2 TrainConfig: `memory_db`, `memory_top_k`, `no_memory`, `unsafe_inline_compile`, `memory_state`, `memory_priors_used`, `eval_n_episodes`, `eval_base_seed`
- [x] 7.3 `_resolve_reward()`: opens MemoryStore when llm + not --no-memory; calls `top_k_by_fitness(fitness_floor=-inf)` (override of spec default for early-stage runs)
- [x] 7.4 Inline `evaluate_on_env_native` runs after training; results written to config.json
- [x] 7.5 MemoryStore.write(entry) called when llm + not --no-memory
- [x] 7.6 Memory store closed in train() finalization

## 8. Smoke tests

- [x] 8.1 env path deterministic: returns match `runs/baseline_seed42` first 10 rows (verified)
- [x] 8.2 llm --no-memory: tested via 8.6 (same code path, no read/write)
- [x] 8.3 First memory run: empty DB created, Gemma 31 lines, entry id=1 written, env_native_mean=-553 recorded
- [x] 8.4 Second/third memory run loads priors: 3rd run loaded 2 priors, prompt contained `PRIOR HIGH-FITNESS ATTEMPTS` + `Attempt A`/`Attempt B`, reward_code verbatim quoted, entry id=3 written
- [x] 8.5 env-native eval inline: all 5 smoke runs' config.json contains `env_native_mean` + `env_native_success` (verified)
- [x] 8.6 --no-memory skips write: DB count stayed at 3 before/after (verified)

## 9. End-to-end 1500-ep run

- [x] 9.1 Ran `gemma_mem_seed42` (empty memory at start) — env_native_mean=235.21, success=80%, crash=3%, entry id=1 written
- [x] 9.2 Ran `gemma_mem_seed43` (read seed 42 as prior) — env_native_mean=224.53, success=78%, crash=3%, `memory_priors_used=[1]` confirmed in config.json, entry id=2 written
- [x] 9.3 Both memory runs ≥ `runs/gemma_seed42` (207.72) — EUREKA open-source replication holds; memory mechanism end-to-end verified
- [x] 9.4 Appended rows to `hermes_dqn/README.md` baseline runs table with n=1 caveat (statistical claim deferred to `closed-loop-fitness`)

## 10. Reward sandbox L2 (subprocess isolation)

- [x] 10.1 Created `hermes_dqn/llm/sandbox.py::validate_reward_in_subprocess(src, timeout_s=10.0, memory_mb=512)`
- [x] 10.2 Child worker: Linux `RLIMIT_AS`, full validation pipeline via `_validate_full`, result via Queue
- [x] 10.3 Parent: spawn-mode mp.Process daemon, `q.get(timeout=...)`, `terminate()` → `kill()` ladder, RuntimeError on truly stuck PID
- [x] 10.4 (Deferred) Windows psutil monitor — current default-on timeout enforcement was sufficient for all smoke cases
- [x] 10.5 Refactored `compile.py`: `_ast_check_and_exec` (no dry-run) + `_validate_full` (with dry-run, for use inside subprocess) + `compile_reward(src, _unsafe_inline=False)` (subprocess path is default)
- [x] 10.6 `--unsafe-inline-compile` flag added to train.py CLI + TrainConfig
- [x] 10.7 Sandbox unit smokes (5/5 pass via `tools/_smoke_sandbox.py`):
  - 10.7.1 Valid reward source ── validated in ~0.8s (subprocess spawn overhead) ✓
  - 10.7.2 Syntax error → `syntax-error` ✓
  - 10.7.3 `import os` → `ast-import-rejected` ✓
  - 10.7.4 `while True: pass` → caught by inner dry-run-timeout (also valid; outer subprocess-timeout would fire for C-extension hangs) ✓
  - 10.7.5 Wrong arity → `signature-arity` ✓
  - Performance: 10-ep training wall-time roughly unchanged from gemma-reward-generator era (subprocess fires once at validate, not per-step)

## 11. Wrap-up

- [x] 11.1 `openspec validate hermes-memory-layer --strict` passes
- [x] 11.2 Sandbox 5/5 + training-memory 5/5 + 1500-ep × 2 cover all 4 spec files' scenarios
- [x] 11.3 No regressions: env path deterministic vs baseline first 10 rows; gemma path still validates source; fitness eval still works
- [x] 11.4 Ready to `/opsx:archive` once next handover is written
