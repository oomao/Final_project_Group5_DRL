"""MemoryEntry: a single row of the long-term reward-and-fitness log.

Each entry captures one training run's reward source plus the resulting
fitness, so future LLM calls can ground new generations on past evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class MemoryEntry:
    """One run's record: reward code, training fitness, env-native eval, optional reflection.

    `id` is assigned by the MemoryStore on first write; pass `None` (the default)
    when constructing client-side. Optional fields default to `None` so legacy
    entries written before later columns existed can still deserialize.
    """

    timestamp: str
    run_dir: str
    reward_fn_sha256: str
    reward_code: str
    mean_reward_last100: float
    success_rate: float
    id: int | None = None
    converge_episode: int | None = None
    env_native_mean: float | None = None
    env_native_success: float | None = None
    lessons_learned: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MemoryEntry:
        # Tolerate unknown keys (forward-compat) by filtering to known fields
        known = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)
