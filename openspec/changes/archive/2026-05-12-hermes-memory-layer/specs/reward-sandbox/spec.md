## ADDED Requirements

### Requirement: Subprocess-isolated reward validation
The system SHALL provide `validate_reward_in_subprocess(src: str, timeout_s: float = 10.0, memory_mb: int | None = 512) -> None` in `hermes_dqn/llm/sandbox.py`. The function SHALL spawn a `multiprocessing.Process`, execute the full validation pipeline (ast.parse, restricted exec, signature check, dry-run) inside the child, and communicate the result back via `multiprocessing.Queue`. Validation success returns silently; any failure raises `RewardCompileError` whose `stage` field identifies which step rejected the source.

#### Scenario: Valid reward source passes
- **WHEN** `validate_reward_in_subprocess(src)` is called with the simplest valid reward (returns `env_reward`)
- **THEN** the function MUST return without raising
- **AND** the spawned subprocess MUST exit cleanly with code 0 before the function returns

#### Scenario: Subprocess result carries stage information
- **WHEN** validation fails inside the child due to a syntax error in `src`
- **THEN** the parent MUST raise `RewardCompileError` with `stage="syntax-error"` and a message including the line number
- **AND** the subprocess MUST have exited before the exception propagates

#### Scenario: Subprocess crash is reported
- **WHEN** the child process raises an unhandled exception during dry-run
- **THEN** the parent MUST receive the exception type and message via the queue and raise `RewardCompileError(stage="dry-run-exception", ...)`

### Requirement: Hard timeout via process termination
If the child process does not return a result within `timeout_s` seconds, the parent SHALL call `proc.terminate()`, wait up to 1 second, then `proc.kill()` if still alive. Parent SHALL raise `RewardCompileError(stage="subprocess-timeout", ...)`. The child SHALL NOT survive past the parent's cleanup.

#### Scenario: Infinite loop in Python is killed
- **WHEN** `src` contains `while True: pass` inside the `reward` function and dry-run invokes it
- **THEN** the parent MUST raise `RewardCompileError(stage="subprocess-timeout", ...)` within `timeout_s + 2` seconds
- **AND** the spawned process MUST NOT be alive when the exception is raised

#### Scenario: Infinite loop in C-extension is killed
- **WHEN** `src` triggers a long-running call into a C-extension (e.g. `np.einsum` on a huge synthetic input)
- **THEN** the parent MUST raise `RewardCompileError(stage="subprocess-timeout", ...)` within `timeout_s + 2` seconds (thread-based timeouts could not kill C-extension work)

#### Scenario: Stuck child receives kill after terminate fails
- **WHEN** `proc.terminate()` does not stop the child within 1 second
- **THEN** the parent MUST call `proc.kill()` and join with a final 1-second deadline
- **AND** if join still fails, the parent MUST raise a fatal `RuntimeError` naming the stuck PID (this is a system-level bug, not LLM behavior)

### Requirement: Memory-cap best effort
On Linux, the child SHALL set `resource.setrlimit(RLIMIT_AS, (memory_mb * 1024 * 1024, ...))` at start so excessive allocation triggers `MemoryError`. On Windows, the parent SHOULD monitor child RSS via `psutil` and terminate if it exceeds `memory_mb`. Lack of `psutil` SHALL NOT prevent the sandbox from running; it gracefully degrades to timeout-only enforcement on Windows.

#### Scenario: Linux RLIMIT triggers MemoryError
- **WHEN** running on Linux and `src` allocates more than `memory_mb` MB
- **THEN** the child MUST receive a `MemoryError` and the parent MUST raise `RewardCompileError(stage="dry-run-exception")` mentioning MemoryError

#### Scenario: Windows without psutil falls back to timeout-only
- **WHEN** running on Windows and `psutil` is not importable
- **THEN** `validate_reward_in_subprocess` MUST still run and rely on `timeout_s` as the only enforcement mechanism
- **AND** it MUST print a one-time warning naming `psutil` and pointing at `pip install psutil` for optional hardening

### Requirement: Default-on sandbox in compile_reward
After this change, `compile_reward(src)` SHALL invoke `validate_reward_in_subprocess(src)` first; only after successful validation MAY it re-compile the source inline (ast.parse + exec with restricted globals) to obtain a callable for use in the training process. The inline re-compile SHALL NOT include a dry-run (the dry-run already happened in the subprocess).

#### Scenario: compile_reward defaults to subprocess validation
- **WHEN** `compile_reward(src)` is called without any sandbox-bypass flag
- **THEN** the call MUST go through `validate_reward_in_subprocess` before returning a callable
- **AND** the returned callable MUST be a fresh in-process function suitable for being called millions of times during DQN training (no IPC per call)

#### Scenario: Bypass flag for debug only
- **WHEN** `compile_reward(src, _unsafe_inline=True)` is called (or train.py is invoked with `--unsafe-inline-compile`)
- **THEN** the subprocess step MUST be skipped (falling back to the legacy threaded-timeout dry-run)
- **AND** a one-time stderr warning MUST be printed naming "sandbox bypassed; debug only"

### Requirement: Training-time reward function is in-process
The compiled-and-returned callable SHALL execute inside the main training process (no IPC per env.step()). The subprocess sandbox is the validation barrier; the training trust boundary is the validated source code itself. This is documented in `design.md` Section I as an intentional trade-off (training would be ~1000× slower if every step used IPC).

#### Scenario: Performance preserved during training
- **WHEN** the validated reward function is invoked from inside the DQN training loop
- **THEN** each call MUST run in the main process address space (no Queue / Pipe round-trip)
- **AND** smoke-test wall-time for 10 episodes MUST stay within +20% of the pre-sandbox baseline (the only added cost should be one subprocess spawn per training run, not per step)
