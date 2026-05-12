"""MemoryStore: SQLite FTS5-backed long-term store for reward-fitness records.

Single-process assumption (one training process per memory_db at a time).
WAL journal mode keeps reads non-blocking, but concurrent writers are not
supported in this MVP — the experiments-protocol spec serializes 4090 access
anyway.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from hermes_dqn.memory.entry import MemoryEntry
from hermes_dqn.memory.schema import apply_schema


_INSERT_OR_UPDATE = """\
INSERT INTO memory (
    timestamp, run_dir, reward_fn_sha256, reward_code,
    converge_episode, mean_reward_last100, success_rate,
    env_native_mean, env_native_success, lessons_learned
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(reward_fn_sha256) DO UPDATE SET
    timestamp           = excluded.timestamp,
    run_dir             = excluded.run_dir,
    converge_episode    = excluded.converge_episode,
    mean_reward_last100 = excluded.mean_reward_last100,
    success_rate        = excluded.success_rate,
    env_native_mean     = excluded.env_native_mean,
    env_native_success  = excluded.env_native_success,
    lessons_learned     = excluded.lessons_learned
RETURNING id
"""


_ORDER_EXPR = {
    "env_native_mean_or_mean_reward": "COALESCE(env_native_mean, mean_reward_last100)",
    "mean_reward_last100": "mean_reward_last100",
    "success_rate": "success_rate",
}


class MemoryStore:
    """Thin SQLite wrapper exposing write + top-K retrieval."""

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = sqlite3.connect(
            str(self._db_path), isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        apply_schema(self._conn)

    def _require_open(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("MemoryStore is closed; reopen with MemoryStore(db_path)")
        return self._conn

    def write(self, entry: MemoryEntry) -> int:
        conn = self._require_open()
        cursor = conn.execute(
            _INSERT_OR_UPDATE,
            (
                entry.timestamp,
                entry.run_dir,
                entry.reward_fn_sha256,
                entry.reward_code,
                entry.converge_episode,
                entry.mean_reward_last100,
                entry.success_rate,
                entry.env_native_mean,
                entry.env_native_success,
                entry.lessons_learned,
            ),
        )
        row = cursor.fetchone()
        new_id = int(row[0])
        # Keep the trigger-maintained FTS index in sync on UPDATE path
        # (the AFTER INSERT trigger only fires on insert; on conflict-update we
        # refresh FTS row manually).
        conn.execute(
            "INSERT OR REPLACE INTO memory_fts(rowid, reward_code, lessons_learned) "
            "VALUES (?, ?, ?)",
            (new_id, entry.reward_code, entry.lessons_learned),
        )
        entry.id = new_id
        return new_id

    def top_k_by_fitness(
        self,
        k: int = 5,
        fitness_floor: float = 0.0,
        order_by: str = "env_native_mean_or_mean_reward",
    ) -> list[MemoryEntry]:
        conn = self._require_open()
        if order_by not in _ORDER_EXPR:
            raise ValueError(
                f"order_by must be one of {sorted(_ORDER_EXPR)}; got {order_by!r}"
            )
        expr = _ORDER_EXPR[order_by]
        sql = (
            "SELECT id, timestamp, run_dir, reward_fn_sha256, reward_code, "
            "converge_episode, mean_reward_last100, success_rate, "
            "env_native_mean, env_native_success, lessons_learned "
            f"FROM memory WHERE {expr} >= ? "
            f"ORDER BY {expr} DESC LIMIT ?"
        )
        rows = conn.execute(sql, (fitness_floor, k)).fetchall()
        return [MemoryEntry.from_dict(dict(row)) for row in rows]

    def all_count(self) -> int:
        conn = self._require_open()
        row = conn.execute("SELECT COUNT(*) AS n FROM memory").fetchone()
        return int(row["n"])

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
