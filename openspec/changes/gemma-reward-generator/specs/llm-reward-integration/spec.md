## ADDED Requirements

### Requirement: train.py --reward-source flag
`train.py` SHALL accept a `--reward-source` CLI flag with values `env` (default) and `llm`. When `env`, training MUST behave identically to `bootstrap-dqn-baseline` (env-native reward). When `llm`, training MUST call `LLMRewardClient.generate()` once before the training loop and inject the resulting callable into `make_env(reward_fn=...)`.

#### Scenario: Default flag preserves baseline
- **WHEN** `python -m hermes_dqn.training.train --episodes 1500 --seed 42` is run without `--reward-source`
- **THEN** training MUST produce per-episode returns byte-identical to `runs/baseline_seed42/episodes.jsonl`

#### Scenario: --reward-source env explicit
- **WHEN** `python -m hermes_dqn.training.train --reward-source env --episodes 10 --seed 42` is run
- **THEN** results MUST match the implicit default and `config.json` MUST record `"reward_source": "env"`

#### Scenario: --reward-source llm path
- **WHEN** `python -m hermes_dqn.training.train --reward-source llm --episodes 10 --seed 42` is run with a valid `GOOGLE_API_KEY`
- **THEN** the run MUST call `LLMRewardClient.generate()` exactly once, write `reward_fn.py` to the run dir, train for 10 episodes, and produce `episodes.jsonl` with 10 rows
- **AND** `config.json` MUST record `"reward_source": "llm"`

### Requirement: API key resolution from .env
When `--reward-source llm` is used, the system SHALL load `GOOGLE_API_KEY` from `.env` via `python-dotenv` (with environment variables taking precedence over `.env` file values) before constructing `LLMRewardClient`.

#### Scenario: Missing API key
- **WHEN** `--reward-source llm` is used and `GOOGLE_API_KEY` is unset in both env and `.env`
- **THEN** `train.py` MUST exit with code 1 and a message naming `GOOGLE_API_KEY`, `.env`, and `.env.example`
- **AND** MUST NOT make any network call

#### Scenario: .env file overridden by environment variable
- **WHEN** `.env` contains `GOOGLE_API_KEY=A` and the shell sets `GOOGLE_API_KEY=B` before launch
- **THEN** the client MUST use `B`

#### Scenario: .env.example present and instructive
- **WHEN** a developer inspects the repo
- **THEN** `.env.example` MUST exist at repo root, MUST be committed to git, and MUST list `GOOGLE_API_KEY=` with an empty value plus a comment explaining where to obtain a key

### Requirement: Mandatory reward_fn.py artifact per run
Every run produced by `train.py` (BOTH `--reward-source env` and `--reward-source llm`) SHALL write `reward_fn.py` to the run directory and record its SHA-256 in `config.json` as `reward_fn_sha256`. This satisfies `establish-project-lifecycle-spec / experiments-protocol` Requirement "Reward Function Artifact and Integrity".

#### Scenario: env-source run writes stub reward_fn.py
- **WHEN** a run is launched with `--reward-source env`
- **THEN** `reward_fn.py` MUST exist in the run dir and contain a single line comment `# env native reward (no custom function)` plus a passthrough `def reward(...)` body
- **AND** `config.json["reward_fn_sha256"]` MUST equal the SHA-256 of that file's bytes

#### Scenario: llm-source run writes generated code verbatim
- **WHEN** a run is launched with `--reward-source llm` and `LLMRewardClient.generate()` returns source string `S`
- **THEN** `reward_fn.py` MUST contain exactly `S` (no re-formatting, no added imports)
- **AND** `config.json["reward_fn_sha256"]` MUST equal `sha256(S.encode("utf-8")).hexdigest()`

#### Scenario: SHA-256 mismatch detection
- **WHEN** an evaluation tool loads a run and the computed SHA-256 of `reward_fn.py` does NOT match `config.json["reward_fn_sha256"]`
- **THEN** the tool MUST refuse to include the run in any statistics and MUST print a tampering warning

### Requirement: Failure modes are non-destructive
If LLM generation or compile fails (any of: API timeout, network error, all 3 retries failing, dry-run rejection), `train.py` SHALL exit non-zero **before** starting any DQN training. The run directory MUST still be created and `llm_attempts.jsonl` written so the failure can be inspected, but `model_final.pt` MUST NOT exist (no partial model).

#### Scenario: All 3 LLM attempts fail
- **WHEN** Gemma produces 3 invalid sources in a row
- **THEN** `train.py` MUST exit code 1
- **AND** `runs/<ts>/llm_attempts.jsonl` MUST contain all 3 attempts
- **AND** `model_final.pt` MUST NOT exist in the run dir

#### Scenario: Network error during generate()
- **WHEN** the `google-genai` SDK raises a network exception
- **THEN** `train.py` MUST exit non-zero with a message indicating the network failure
- **AND** no partial run artifact MUST be left in an inconsistent state (either nothing, or a complete `llm_attempts.jsonl` recording the failure)
