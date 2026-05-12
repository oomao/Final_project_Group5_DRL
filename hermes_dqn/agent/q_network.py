"""MLP Q-network for vanilla DQN."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """MLP that maps observation → Q-values for every discrete action.

    Defaults to two hidden layers of 64 units with ReLU, matching the
    vanilla-DQN baseline in design.md. Hidden sizes are configurable so later
    changes can swap depth/width without touching this file.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: Sequence[int] = (64, 64),
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
