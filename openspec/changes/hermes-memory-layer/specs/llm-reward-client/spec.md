## MODIFIED Requirements

### Requirement: Generate Python reward source code
`LLMRewardClient.generate(task_spec, memory=None, attempts_log_path=None)` SHALL prompt Gemma with the task description, the 7-arg `RewardFunction` Protocol signature, a few-shot example, and (when `memory` is a non-empty list of `MemoryEntry` instances) a "PRIOR HIGH-FITNESS ATTEMPTS" block summarizing the provided memory entries. SHALL return only Python source code defining a top-level function named `reward`. The `memory` parameter defaults to `None`, which is treated equivalently to `[]`; existing callers that did not pass `memory` SHALL behave identically to the `gemma-reward-generator` change.

#### Scenario: Successful single-shot generation
- **WHEN** Gemma returns a valid Python `def reward(obs, action, next_obs, env_reward, terminated, truncated, info): ...` block in markdown or plain text
- **THEN** `generate()` MUST extract the code block (stripping ``` fences) and return the raw source string

#### Scenario: Markdown noise around code
- **WHEN** Gemma's response contains explanatory prose before and after a fenced code block
- **THEN** `generate()` MUST return only the contents of the fenced block, with prose stripped

#### Scenario: Backward compatibility — memory omitted
- **WHEN** `client.generate(task_spec)` is called without the `memory` parameter
- **THEN** the assembled prompt MUST NOT contain any "PRIOR HIGH-FITNESS ATTEMPTS" section
- **AND** the behavior MUST be byte-identical to the prompt produced by the `gemma-reward-generator` change for the same `task_spec`

#### Scenario: Empty memory list
- **WHEN** `client.generate(task_spec, memory=[])` is called
- **THEN** the assembled prompt MUST NOT contain any "PRIOR HIGH-FITNESS ATTEMPTS" section
- **AND** the behavior MUST be identical to omitting `memory` entirely

#### Scenario: Memory entries become prompt context
- **WHEN** `client.generate(task_spec, memory=[entry_A, entry_B])` is called with two entries
- **THEN** the prompt MUST contain a "PRIOR HIGH-FITNESS ATTEMPTS" section
- **AND** that section MUST quote `entry_A.reward_code` and `entry_B.reward_code` verbatim inside fenced Python blocks
- **AND** for each entry, the section MUST include its `env_native_mean` (or `mean_reward_last100` if the former is `None`) and `success_rate`

#### Scenario: Lessons learned are surfaced
- **WHEN** any entry in `memory` has a non-null `lessons_learned` field
- **THEN** the rendered "PRIOR HIGH-FITNESS ATTEMPTS" section for that entry MUST include a `Lessons: <text>` line
- **AND** entries with `lessons_learned=None` MUST omit the `Lessons:` line entirely (no empty placeholder)

### Requirement: Sandboxed compile of generated source
`compile_reward(src: str)` SHALL parse the source via `ast.parse`, reject any `import` statement, and reject any attribute access starting with `_`. It SHALL execute validation (parse, exec with restricted builtins, signature check, dry-run) inside an isolated `multiprocessing.Process` via `hermes_dqn.llm.sandbox.validate_reward_in_subprocess`. Only after subprocess validation succeeds MAY the function inline re-compile the source in the parent process to return a callable. The restricted `globals` dict SHALL have `__builtins__` as a whitelist (`abs, min, max, sum, len, range, float, int, bool, dict, list, tuple, pow, round, isinstance, type, print, True, False, None`) plus `np` bound to the `numpy` module. Hard-kill timeout (subprocess) replaces the legacy threaded soft timeout for the validation step.

#### Scenario: Import is rejected
- **WHEN** the LLM source contains `import os` or `from os import path`
- **THEN** `compile_reward` MUST raise `RewardCompileError` with `stage="ast-import-rejected"`
- **AND** the rejection MUST occur inside the subprocess (parent reports stage from queue)

#### Scenario: Whitelisted builtins are available
- **WHEN** the LLM source calls `abs(x)` or `np.linalg.norm(obs)`
- **THEN** `compile_reward` MUST execute without `NameError`

#### Scenario: Non-whitelisted builtin is blocked
- **WHEN** the LLM source calls `open("/etc/passwd")` or `eval("...")` or `exec("...")`
- **THEN** the compiled function MUST raise `NameError` at call time because the builtin is not in the whitelist

#### Scenario: Subprocess timeout supersedes thread timeout
- **WHEN** the LLM source contains a Python or C-level infinite loop
- **THEN** `compile_reward` MUST raise `RewardCompileError(stage="subprocess-timeout", ...)` within bounded time
- **AND** the spawned validation process MUST be terminated (not merely left orphaned as the threaded approach allowed)

#### Scenario: Performance contract — training is in-process
- **WHEN** `compile_reward(src)` returns a callable that the training loop invokes per env.step()
- **THEN** each invocation MUST occur in the main process address space (no IPC round-trip per step)
- **AND** the only sandbox cost SHALL be the one-time subprocess spawn during validation
