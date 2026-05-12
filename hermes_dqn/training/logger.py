"""JSON-lines logger for per-episode training metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlLogger:
    """Append-only logger writing one JSON object per call to ``log()``.

    File handle stays open for the lifetime of the logger; ``close()`` flushes
    and releases it. Supports use as a context manager.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("w", encoding="utf-8")

    def log(self, record: dict[str, Any]) -> None:
        self._fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fp.flush()

    def close(self) -> None:
        if not self._fp.closed:
            self._fp.close()

    def __enter__(self) -> JsonlLogger:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
