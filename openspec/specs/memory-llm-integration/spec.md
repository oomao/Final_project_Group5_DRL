# memory-llm-integration Specification

## Purpose
TBD - created by archiving change hermes-memory-layer. Update Purpose after archive.
## Requirements
### Requirement: Memory-related CLI flags on train.py
`train.py` SHALL accept three new optional CLI flags: `--memory-db <path>` (default `runs/memory.sqlite`), `--memory-top-k <K>` (default `5`), and `--no-memory` (a boolean toggle disabling all memory read/write). The flags SHALL be parsed by argparse and stored in `TrainConfig` so they appear in `config.json`.

#### Scenario: Default memory-db path
- **WHEN** `train.py` is invoked with `--reward-source llm` and no explicit `--memory-db`
- **THEN** the run MUST resolve memory_db to `runs/memory.sqlite`
- **AND** `config.json["memory_db"]` MUST equal `"runs/memory.sqlite"` (or its OS-equivalent)

#### Scenario: --no-memory disables both read and write
- **WHEN** `train.py --reward-source llm --no-memory` is run
- **THEN** the LLM client MUST NOT receive any memory entries in its prompt
- **AND** no row MUST be written to any database after training

#### Scenario: Custom memory-top-k
- **WHEN** `train.py --reward-source llm --memory-top-k 3` is run with a store containing ≥ 3 entries
- **THEN** the LLM client MUST receive exactly 3 prior entries (those with the highest fitness)

### Requirement: Pre-training memory read
When `--reward-source llm` is active and `--no-memory` is NOT set, `train.py` SHALL open `MemoryStore` and fetch `top_k_by_fitness(k=memory_top_k)` BEFORE calling `LLMRewardClient.generate()`, passing the result as the `memory=` argument.

#### Scenario: Memory read on empty database
- **WHEN** the memory DB does not exist yet (first run)
- **THEN** `MemoryStore` MUST create the DB and `top_k_by_fitness()` MUST return `[]`
- **AND** the LLM call MUST proceed with empty memory (equivalent to gemma-reward-generator behavior)

#### Scenario: Memory read on populated database
- **WHEN** the memory DB contains 3 entries from prior runs
- **THEN** `MemoryStore.top_k_by_fitness(k=5)` MUST return 3 entries
- **AND** `LLMRewardClient.generate()` MUST be invoked with `memory=` containing those 3 entries

### Requirement: Inline env-native evaluation after training
`train.py` SHALL invoke `hermes_dqn.training.eval_env_native.evaluate_on_env_native(model_path, n=100, base_seed=10000)` after training but before writing to memory, and SHALL record `env_native_mean` and `env_native_success` in both `config.json` and the `MemoryEntry`. The eval seeds MUST be disjoint from training seeds (use 10000+).

#### Scenario: env-native eval is mandatory for llm path
- **WHEN** `train.py --reward-source llm --episodes 1500 --seed 42` completes training
- **THEN** an env-native evaluation MUST run on 100 episodes (greedy, ε=0.0)
- **AND** `config.json` MUST contain `env_native_mean` and `env_native_success` as floats

#### Scenario: env-native eval is also mandatory for env path
- **WHEN** `train.py --reward-source env` completes training
- **THEN** the env-native eval MUST still run (so `runs/baseline_seed42` and re-runs can be compared apples-to-apples)
- **AND** `config.json` MUST contain `env_native_mean` and `env_native_success`

### Requirement: Post-training memory write
After env-native eval completes, `train.py` SHALL write a `MemoryEntry` to `MemoryStore` capturing the run's reward source, fitness, and env-native metrics — **only** when `--reward-source llm` is used AND `--no-memory` is NOT set. The `env` path SHALL NOT write to memory (the env stub reward has no learning value for the LLM).

#### Scenario: llm path writes to memory
- **WHEN** `train.py --reward-source llm` finishes training and env-native eval
- **THEN** `MemoryStore.write(entry)` MUST be called with the run's metrics
- **AND** a subsequent `top_k_by_fitness(k=1)` on the same DB MUST return that entry

#### Scenario: env path does NOT write to memory
- **WHEN** `train.py --reward-source env` finishes
- **THEN** the memory DB row count MUST NOT increase

#### Scenario: --no-memory skips write
- **WHEN** `train.py --reward-source llm --no-memory` finishes
- **THEN** the memory DB row count MUST NOT increase

### Requirement: config.json captures memory state
`config.json` SHALL include three new fields whenever memory is touched: `memory_state` (`"none"` when `--no-memory` or `--reward-source env`, otherwise `"hermes-sqlite-fts5"`), `memory_top_k` (the K used or 0 when disabled), and `memory_priors_used` (the list of MemoryEntry ids that were fed into the LLM prompt, or `[]` if none).

#### Scenario: env path records memory_state none
- **WHEN** `train.py --reward-source env` is run
- **THEN** `config.json["memory_state"]` MUST equal `"none"`
- **AND** `config.json["memory_priors_used"]` MUST equal `[]`

#### Scenario: llm path with empty DB records empty priors
- **WHEN** `train.py --reward-source llm` is run with a fresh empty DB
- **THEN** `config.json["memory_state"]` MUST equal `"hermes-sqlite-fts5"`
- **AND** `config.json["memory_priors_used"]` MUST equal `[]`

#### Scenario: llm path with priors records ids
- **WHEN** `train.py --reward-source llm` is run and the LLM call received 3 prior entries
- **THEN** `config.json["memory_priors_used"]` MUST be a list of 3 integers (the entries' ids)

### Requirement: Two-run cumulative learning smoke
Running `train.py --reward-source llm` twice in sequence with the default memory DB SHALL result in the second run's LLM prompt containing the first run's reward and fitness — provided the first run wrote a memory entry and the second was launched against the same DB.

#### Scenario: Sequential runs see each other
- **WHEN** the first run `train.py --reward-source llm --seed 42 --episodes 10` completes and writes a memory entry
- **AND** the second run `train.py --reward-source llm --seed 43 --episodes 10` is launched against the same DB
- **THEN** the second run's `llm_attempts.jsonl[0].prompt` MUST contain a "PRIOR HIGH-FITNESS ATTEMPTS" section
- **AND** that section MUST quote the first run's `reward_code`

