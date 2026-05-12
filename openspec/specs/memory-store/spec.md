# memory-store Specification

## Purpose
TBD - created by archiving change hermes-memory-layer. Update Purpose after archive.
## Requirements
### Requirement: MemoryEntry data shape
The system SHALL define a `MemoryEntry` dataclass with these fields: `id: int | None` (auto-assigned by store), `timestamp: str` (ISO 8601 UTC), `run_dir: str`, `reward_fn_sha256: str`, `reward_code: str`, `converge_episode: int | None`, `mean_reward_last100: float`, `success_rate: float`, `env_native_mean: float | None`, `env_native_success: float | None`, `lessons_learned: str | None`. All future-optional fields default to `None` so legacy entries deserialize cleanly.

#### Scenario: Construct entry with required fields only
- **WHEN** a developer creates `MemoryEntry(timestamp=..., run_dir=..., reward_fn_sha256=..., reward_code=..., mean_reward_last100=..., success_rate=...)`
- **THEN** all unspecified fields MUST default to `None`
- **AND** the instance MUST be JSON-serializable to a dict matching the SQLite column names

### Requirement: MemoryStore persistence backend
The system SHALL provide `MemoryStore(db_path: str | Path)` that opens (or creates) a SQLite database at the given path with FTS5 enabled and WAL journal mode. Schema MUST be applied idempotently — opening the same `db_path` twice MUST NOT corrupt or duplicate tables.

#### Scenario: First-time database creation
- **WHEN** `MemoryStore("runs/memory.sqlite")` is constructed and the file does not exist
- **THEN** the file MUST be created with a `memory` table, indexes (`idx_memory_env_native_mean`, `idx_memory_mean_reward`), `memory_fts` virtual table, and an `AFTER INSERT` trigger keeping FTS in sync
- **AND** `PRAGMA journal_mode` MUST return `wal`

#### Scenario: Re-opening an existing database
- **WHEN** `MemoryStore("runs/memory.sqlite")` is constructed twice in sequence on the same path
- **THEN** the second open MUST NOT raise
- **AND** all data from the first session MUST be readable

#### Scenario: FTS5 not available
- **WHEN** the host SQLite build lacks FTS5 support
- **THEN** the constructor MUST raise a clear `RuntimeError` naming "FTS5 required (SQLite 3.9+)" and pointing the developer at upgrading Python/SQLite

### Requirement: Write entries with reward-hash deduplication
`MemoryStore.write(entry: MemoryEntry) -> int` SHALL insert the entry and return its assigned `id`. The `reward_fn_sha256` column SHALL be UNIQUE; a second write of the same hash SHALL be treated as an upsert that overwrites all fitness columns with the new values and returns the existing `id`.

#### Scenario: Write a new entry
- **WHEN** `store.write(entry)` is called with a previously-unseen `reward_fn_sha256`
- **THEN** the entry MUST be inserted, `id` MUST be a positive integer
- **AND** a subsequent `top_k_by_fitness(k=1)` MUST include this entry

#### Scenario: Upsert same reward
- **WHEN** `store.write(entry1)` then `store.write(entry2)` where `entry1.reward_fn_sha256 == entry2.reward_fn_sha256` but `entry2.mean_reward_last100 > entry1.mean_reward_last100`
- **THEN** the store MUST contain exactly one row for that sha
- **AND** that row's `mean_reward_last100` MUST equal `entry2.mean_reward_last100`
- **AND** both `write` calls MUST return the same `id`

### Requirement: Top-K retrieval by fitness
`MemoryStore.top_k_by_fitness(k: int = 5, fitness_floor: float = 0.0, order_by: str = "env_native_mean_or_mean_reward") -> list[MemoryEntry]` SHALL return up to `k` highest-fitness entries, filtering out entries whose ordering value is below `fitness_floor`.

#### Scenario: Default ordering prefers env-native, falls back to shaped
- **WHEN** the store has entries A (`env_native_mean=210`, `mean_reward_last100=300`) and B (`env_native_mean=None`, `mean_reward_last100=320`)
- **AND** `top_k_by_fitness(k=2)` is called
- **THEN** the returned list MUST contain both entries
- **AND** the order MUST place A before B (A's env_native_mean=210 beats B's fallback mean=320 only if B has env_native_mean; since B falls back to `mean_reward_last100=320` which is > 210, B comes first)
- **AND** the test demonstrates the documented fallback rule precisely

#### Scenario: fitness_floor filters
- **WHEN** the store has 3 entries with env_native_mean 50 / 150 / 250
- **AND** `top_k_by_fitness(k=10, fitness_floor=100)` is called
- **THEN** the returned list MUST contain exactly 2 entries (env_native_mean = 250 then 150)

#### Scenario: Fewer than K entries available
- **WHEN** the store has 2 entries and `top_k_by_fitness(k=5)` is called
- **THEN** the returned list MUST contain exactly 2 entries (no padding, no error)

#### Scenario: Empty database
- **WHEN** the store has 0 entries and `top_k_by_fitness(k=5)` is called
- **THEN** the returned list MUST be empty (`[]`)

### Requirement: Connection lifecycle
`MemoryStore.close()` SHALL be idempotent. The class SHALL also be usable as a context manager (`with MemoryStore(...) as store:` automatically closes on exit). Operations on a closed store SHALL raise `RuntimeError`.

#### Scenario: Context manager
- **WHEN** a developer uses `with MemoryStore(db_path) as store: store.write(entry)`
- **THEN** the connection MUST be closed after the `with` block exits
- **AND** any pending writes MUST be committed before close

#### Scenario: Double-close is safe
- **WHEN** `store.close()` is called twice
- **THEN** the second call MUST NOT raise

#### Scenario: Use after close
- **WHEN** `store.close()` is called, then `store.write(entry)` is attempted
- **THEN** a `RuntimeError` MUST be raised naming the closed state

