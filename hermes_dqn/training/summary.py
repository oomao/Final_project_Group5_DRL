"""Closed-loop iteration summary structures.

Mirrors the schema documented in closed-loop-fitness/design.md section D.
Persisted as ``runs/<exp>/<cond>/seed_<NN>/summary.json``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class IterationSummary:
    """One closed-loop iteration's outcome."""

    iter: int
    reward_fn_sha256: str
    memory_priors_used: list[int]
    diff_from_prev: dict[str, Any] | None  # {"kind": str, "similarity": float, "diff_summary": str}
    buffer_action: str | None  # "keep" / "decay" / "clear" / None for iter 1
    env_native_mean: float
    env_native_success: float
    env_native_crash_rate: float
    shaped_mean_last100: float
    converge_episode: int | None
    wall_time_s: float
    status: str = "ok"  # "ok" or "failed"
    error: str | None = None


@dataclass
class ClosedLoopSummary:
    """One (exp_name, condition_id, seed) closed-loop run summary."""

    exp_name: str
    condition_id: str
    seed: int
    n_iterations: int
    iterations: list[IterationSummary] = field(default_factory=list)
    total_wall_time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json_file(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fp:
            json.dump(self.to_dict(), fp, indent=2)
