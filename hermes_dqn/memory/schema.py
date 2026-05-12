"""SQLite schema for the Hermes long-term memory store.

Single-version DDL applied idempotently via `apply_schema(conn)`. Future
column additions go through `migrate(conn, target_version)` with explicit
upgrade steps; for now we are at version 1 and `migrate` is a placeholder.
"""

from __future__ import annotations

import sqlite3

CURRENT_VERSION = 1


_DDL_TABLE = """\
CREATE TABLE IF NOT EXISTS memory (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           TEXT    NOT NULL,
    run_dir             TEXT    NOT NULL,
    reward_fn_sha256    TEXT    NOT NULL UNIQUE,
    reward_code         TEXT    NOT NULL,
    converge_episode    INTEGER,
    mean_reward_last100 REAL    NOT NULL,
    success_rate        REAL    NOT NULL,
    env_native_mean     REAL,
    env_native_success  REAL,
    lessons_learned     TEXT,
    created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
)
"""

_DDL_INDEX_ENV = (
    "CREATE INDEX IF NOT EXISTS idx_memory_env_native_mean "
    "ON memory(env_native_mean DESC)"
)
_DDL_INDEX_REWARD = (
    "CREATE INDEX IF NOT EXISTS idx_memory_mean_reward "
    "ON memory(mean_reward_last100 DESC)"
)

_DDL_FTS = """\
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    reward_code, lessons_learned,
    content='memory', content_rowid='id'
)
"""

_DDL_TRIGGER_INSERT = """\
CREATE TRIGGER IF NOT EXISTS memory_ai AFTER INSERT ON memory BEGIN
    INSERT INTO memory_fts(rowid, reward_code, lessons_learned)
    VALUES (new.id, new.reward_code, new.lessons_learned);
END
"""

_DDL_USER_VERSION = f"PRAGMA user_version = {CURRENT_VERSION}"
_DDL_WAL = "PRAGMA journal_mode = WAL"


def _fts5_available(conn: sqlite3.Connection) -> bool:
    """Probe SQLite build for FTS5 support."""
    try:
        conn.execute("CREATE VIRTUAL TABLE _fts5_probe USING fts5(x)")
        conn.execute("DROP TABLE _fts5_probe")
        return True
    except sqlite3.OperationalError:
        return False


def apply_schema(conn: sqlite3.Connection) -> None:
    """Idempotent: safe to call on a fresh or existing DB."""
    if not _fts5_available(conn):
        raise RuntimeError(
            "FTS5 required (SQLite 3.9+). The current Python's sqlite3 module was "
            "built without FTS5; upgrade Python or rebuild sqlite3."
        )
    conn.execute(_DDL_WAL)
    conn.execute(_DDL_TABLE)
    conn.execute(_DDL_INDEX_ENV)
    conn.execute(_DDL_INDEX_REWARD)
    conn.execute(_DDL_FTS)
    conn.execute(_DDL_TRIGGER_INSERT)
    conn.execute(_DDL_USER_VERSION)
    conn.commit()


def current_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("PRAGMA user_version").fetchone()
    return int(row[0]) if row else 0


def migrate(conn: sqlite3.Connection, target: int = CURRENT_VERSION) -> None:
    """Stub for future schema migrations. Currently only version 1 exists."""
    have = current_version(conn)
    if have >= target:
        return
    if have == 0:
        apply_schema(conn)
        return
    raise NotImplementedError(
        f"No migration path from version {have} to {target} is defined yet"
    )
