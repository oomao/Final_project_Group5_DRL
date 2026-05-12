## 1. Dependencies & Environment

- [x] 1.1 Add `google-genai~=2.0` and `python-dotenv~=1.0` to `pyproject.toml [project] dependencies` and `requirements.txt`
- [x] 1.2 Run `pip install google-genai python-dotenv` and confirm imports work (google-genai 2.0.1 installed)
- [x] 1.3 Create root-level `.env.example` with `GOOGLE_API_KEY=` and `GEMMA_MODEL=` (default falls back to `gemma-4-31b-it`)
- [x] 1.4 Confirm `.env` is in `.gitignore` (added — was missing from baseline; user's key now safe)

## 2. Prompts (capability: llm-reward-client)

- [x] 2.1 Create `hermes_dqn/llm/__init__.py` re-exporting `LLMRewardClient`, `compile_reward`, `RewardCompileError`, `RewardGenerationError`
- [x] 2.2 Implement `hermes_dqn/llm/prompts.py::build_lunarlander_prompt(task_spec, retry_context=None, force_fallback=False)` with system preamble, task spec, response format, 2 few-shot examples (passthrough + light shaping), and retry/fallback branches
- [x] 2.3 English prompts (per design.md)

## 3. Compile sandbox (capability: llm-reward-client)

- [x] 3.1 Implement `hermes_dqn/llm/compile.py::SAFE_BUILTINS` whitelist dict (20 entries: abs/min/max/sum/len/range/float/int/bool/dict/list/tuple/pow/round/isinstance/type/print + True/False/None)
- [x] 3.2 Implement `RewardCompileError(stage, message, tb)` exception class
- [x] 3.3 Implement `compile_reward(src)` flow: ast.parse → reject Import/ImportFrom + dunder attrs → exec with SAFE_BUILTINS + np → extract `reward` → signature arity == 7 → threaded dry-run with 100ms timeout
- [x] 3.4 Return validated callable on success; raise on any failure with stage + tb

## 4. LLM client (capability: llm-reward-client)

- [x] 4.1 Implement `hermes_dqn/llm/client.py::LLMRewardClient(api_key=None, model=None)` with env-var resolution and ValueError on missing key
- [x] 4.2 Implement `LLMRewardClient.generate()` with 3-attempt loop, API exception handling, code-block extraction (markdown fences), and force_fallback on attempt 3
- [x] 4.3 Implement `RewardGenerationError` exception listing each attempt's failure reason

## 5. Attempt logging (capability: llm-reward-client)

- [x] 5.1 `LLMRewardClient.generate(attempts_log_path=...)` writes JSONL with `{attempt, prompt, response, error, accepted}` per attempt (both success and failure paths)
- [x] 5.2 `train.py` passes `run_dir / "llm_attempts.jsonl"` so the artifact lands beside `episodes.jsonl`

## 6. Train integration (capability: llm-reward-integration)

- [x] 6.1 Add `--reward-source {env,llm}` to argparse in `train.py`, default `env`
- [x] 6.2 `dotenv.load_dotenv()` called inside `_resolve_reward` (lazy, only when reward_source=="llm")
- [x] 6.3 In `train()`, branch on `reward_source` via `_resolve_reward()` helper
- [x] 6.4 Record `reward_fn_sha256` in `config.json` for BOTH paths (computed from bytes-on-disk via `write_bytes`)
- [x] 6.5 Record `reward_source` field in `config.json`
- [x] 6.6 If LLM generate or compile fails: exit code 1 BEFORE training; verified `model_final.pt` does not exist
- [x] 6.7 Lazy import of dotenv + google-genai inside `_resolve_reward` so env path has zero LLM dep cost

## 7. Smoke test

- [x] 7.1 `--reward-source env --episodes 10 --seed 42` produces identical episode returns to `runs/baseline_seed42` first 10 rows (PASS: deterministic byte-match)
- [x] 7.2 Verified `reward_fn.py` (stub passthrough) and `reward_fn_sha256` in env-source run (PASS: SHA match)
- [x] 7.3 `--reward-source llm --episodes 10 --seed 42` with valid `GOOGLE_API_KEY` (PASS: Gemma generated 34-line reward, 1 retry needed due to API 500)
- [x] 7.4 `llm_attempts.jsonl` has 2 lines (attempt 1: api-call 500 error, attempt 2: accepted=true) (PASS)
- [x] 7.5 `reward_fn.py` matches `llm_attempts.jsonl` last accepted response (PASS: byte-identical to extracted code block)
- [x] 7.6 SHA-256 in config.json matches `sha256(reward_fn.py)` (PASS after fixing Windows newline issue with `write_bytes`)
- [x] 7.7 Failure case: `.env` moved aside, `--reward-source llm` exits code 1 with clear message pointing to `.env.example`; no `model_final.pt` created (PASS)

## 8. End-to-end 1500-ep run on 4090

- [x] 8.1 Ran `python -m hermes_dqn.training.train --reward-source llm --episodes 1500 --seed 42 --out-dir runs/gemma_seed42` (16m29s, 33% faster than baseline)
- [x] 8.2 Ran `FitnessEvaluator` on the result; shaped fitness: converge_ep=525, mean_last100=312.21 (shaped), success=0.85 (shaped). NOTE: shaped values not directly comparable to env-native baseline; see 8.3 below for fair comparison
- [x] 8.3 Appended Baseline runs table to `hermes_dqn/README.md` with BOTH in-training shaped fitness AND apples-to-apples env-native eval (100 unseen seeds via `tools/_eval_env_native.py`). Result: Gemma reward gives +28% mean / +25pp success / -50% crash vs env-native baseline. EUREKA open-source replication thesis holds for seed 42

## 9. Wrap-up

- [x] 9.1 `openspec validate gemma-reward-generator --strict` passes
- [x] 9.2 All 25 scenarios across `specs/llm-reward-client/spec.md` (14) + `specs/llm-reward-integration/spec.md` (11) verified via smoke 7.x + 1500-ep run
- [x] 9.3 Cited spec scenarios from `establish-project-lifecycle-spec` (env-setup R5 API key, experiments-protocol R5 reward_fn artifact) and `bootstrap-dqn-baseline` (reward-plugin Injectable env wrapper) confirmed satisfied
- [x] 9.4 Ready to `/opsx:archive` once `02-ending.sh` writes next handover
