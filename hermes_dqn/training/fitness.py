"""Fitness scoring: turn an episodes.jsonl into a comparable metrics report."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class FitnessReport:
    """Quantitative score of a DQN training run.

    ``converge_episode`` is the first 1-indexed episode whose trailing
    ``window`` mean meets ``success_threshold``; ``None`` if no such episode
    exists. ``success_rate`` is computed over the last ``window`` episodes
    (or all of them, if the run is shorter).
    """

    converge_episode: int | None
    mean_reward_last100: float
    success_rate: float
    total_episodes: int
    seed: int


class FitnessEvaluator:
    """Reads an ``episodes.jsonl`` and reports convergence / success metrics."""

    def __init__(self, success_threshold: float = 200.0, window: int = 100):
        self.success_threshold = success_threshold
        self.window = window

    def evaluate(self, jsonl_path: str | Path) -> FitnessReport:
        jsonl_path = Path(jsonl_path)
        returns: list[float] = []
        with jsonl_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                returns.append(float(row["return"]))

        total = len(returns)
        tail = returns[-self.window :] if total > 0 else []
        mean_tail = float(sum(tail) / len(tail)) if tail else 0.0
        success_rate = (
            float(sum(1 for r in tail if r >= self.success_threshold) / len(tail))
            if tail
            else 0.0
        )
        converge_episode = self._first_converge_episode(returns)

        seed = self._read_seed(jsonl_path)

        return FitnessReport(
            converge_episode=converge_episode,
            mean_reward_last100=mean_tail,
            success_rate=success_rate,
            total_episodes=total,
            seed=seed,
        )

    def _first_converge_episode(self, returns: list[float]) -> int | None:
        """Return the 1-indexed episode where the trailing ``window`` mean first hits threshold."""
        if len(returns) < self.window:
            return None
        running_sum = sum(returns[: self.window])
        if running_sum / self.window >= self.success_threshold:
            return self.window
        for i in range(self.window, len(returns)):
            running_sum += returns[i] - returns[i - self.window]
            if running_sum / self.window >= self.success_threshold:
                return i + 1
        return None

    @staticmethod
    def _read_seed(jsonl_path: Path) -> int:
        """Pull ``seed`` from ``config.json`` next to the JSONL; default 0 if absent."""
        config_path = jsonl_path.parent / "config.json"
        if not config_path.exists():
            return 0
        try:
            with config_path.open("r", encoding="utf-8") as fp:
                config = json.load(fp)
            return int(config.get("seed", 0))
        except (json.JSONDecodeError, ValueError):
            return 0

    @staticmethod
    def report_to_dict(report: FitnessReport) -> dict:
        return asdict(report)
