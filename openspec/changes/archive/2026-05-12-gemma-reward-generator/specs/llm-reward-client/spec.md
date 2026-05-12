## ADDED Requirements

### Requirement: LLM Reward Client class
The system SHALL provide `LLMRewardClient(api_key: str, model: str | None = None)` that wraps the Google AI Studio (google-genai) SDK and exposes a single `generate(task_spec) -> str` method returning Python source code for a reward function.

#### Scenario: Construct client without explicit model
- **WHEN** a developer calls `LLMRewardClient(api_key="AIza...")` without passing `model`
- **THEN** the client MUST default to the model name in `GEMMA_MODEL` env var, falling back to `"gemma-3-27b-it"` if unset

#### Scenario: Construct client without API key
- **WHEN** `LLMRewardClient()` is called with `api_key=""` or `api_key=None`
- **THEN** the constructor MUST raise `ValueError` with a message naming `GOOGLE_API_KEY` and `.env.example`

### Requirement: Generate Python reward source code
`LLMRewardClient.generate(task_spec)` SHALL prompt Gemma with the task description, the 7-arg `RewardFunction` Protocol signature, and a few-shot example, and SHALL return only Python source code defining a top-level function named `reward`.

#### Scenario: Successful single-shot generation
- **WHEN** Gemma returns a valid Python `def reward(obs, action, next_obs, env_reward, terminated, truncated, info): ...` block in markdown or plain text
- **THEN** `generate()` MUST extract the code block (stripping ``` fences) and return the raw source string

#### Scenario: Markdown noise around code
- **WHEN** Gemma's response contains explanatory prose before and after a fenced code block
- **THEN** `generate()` MUST return only the contents of the fenced block, with prose stripped

### Requirement: Automatic retry up to 3 attempts
When `compile_reward()` rejects the generated source, the system SHALL re-prompt Gemma with the error message and traceback, up to **3 total attempts**. The third attempt MUST instruct Gemma to fall back to the simplest valid form (`return env_reward`).

#### Scenario: Compile error triggers retry
- **WHEN** attempt 1's output fails `compile_reward` with a `SyntaxError`
- **THEN** the client MUST send a second prompt containing the SyntaxError message and instruct Gemma to fix it
- **AND** the second response MUST be re-validated through `compile_reward`

#### Scenario: Three attempts exhausted
- **WHEN** all 3 attempts produce sources that fail `compile_reward`
- **THEN** the client MUST raise `RewardGenerationError` whose message lists the failure reason of each attempt

#### Scenario: Third attempt enforces fallback
- **WHEN** attempt 3 is launched after attempts 1 and 2 failed
- **THEN** the prompt MUST explicitly instruct Gemma: "Return the simplest possible reward: `def reward(obs, action, next_obs, env_reward, terminated, truncated, info): return env_reward`"

### Requirement: Persist LLM attempts to JSONL
For each `train.py` run that uses `--reward-source llm`, the system SHALL write `runs/<run_dir>/llm_attempts.jsonl` with one line per attempt containing `{attempt: int, prompt: str, response: str, error: str | null, accepted: bool}`.

#### Scenario: All attempts recorded
- **WHEN** a run launches with `--reward-source llm` and Gemma needs 2 attempts to produce valid code
- **THEN** `llm_attempts.jsonl` MUST contain exactly 2 lines, the first with `accepted: false` and an `error` field, the second with `accepted: true` and `error: null`

#### Scenario: Failed run still records all attempts
- **WHEN** all 3 attempts fail and `RewardGenerationError` is raised
- **THEN** `llm_attempts.jsonl` MUST contain 3 lines, all with `accepted: false`

### Requirement: Sandboxed compile of generated source
`compile_reward(src: str)` SHALL parse the source via `ast.parse`, reject any `import` statement, and execute the compiled code with a restricted `globals` dict whose `__builtins__` is a whitelist (`abs, min, max, sum, len, range, float, int, bool, dict, list, tuple, print`) plus `np` bound to the `numpy` module.

#### Scenario: Import is rejected
- **WHEN** the LLM source contains `import os` or `from os import path`
- **THEN** `compile_reward` MUST raise `RewardCompileError` with `stage="ast-import-rejected"`

#### Scenario: Whitelisted builtins are available
- **WHEN** the LLM source calls `abs(x)` or `np.linalg.norm(obs)`
- **THEN** `compile_reward` MUST execute without `NameError`

#### Scenario: Non-whitelisted builtin is blocked
- **WHEN** the LLM source calls `open("/etc/passwd")` or `eval("...")` or `exec("...")`
- **THEN** the compiled function MUST raise `NameError` at call time because the builtin is not in the whitelist

### Requirement: Dry-run validation before returning callable
After successful compile, `compile_reward` SHALL invoke the resulting function once with a synthetic transition (random obs/action/next_obs) under a 100ms wall-time cap and SHALL reject the function if the call raises or returns a non-float.

#### Scenario: Function returns non-float
- **WHEN** the LLM-generated function returns a string or `None`
- **THEN** `compile_reward` MUST raise `RewardCompileError` with `stage="dry-run-return-type"`

#### Scenario: Function exceeds time budget
- **WHEN** the dry-run call exceeds 100ms (e.g. an infinite loop)
- **THEN** `compile_reward` MUST raise `RewardCompileError` with `stage="dry-run-timeout"`

#### Scenario: Function signature wrong arity
- **WHEN** the LLM produces `def reward(obs, action):` (2 args instead of 7)
- **THEN** `compile_reward` MUST raise `RewardCompileError` with `stage="signature-arity"`
