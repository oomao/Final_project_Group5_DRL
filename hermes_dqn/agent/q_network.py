"""MLP Q-network architectures for DQN. Includes vanilla and Dueling variants."""

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


class DuelingQNetwork(nn.Module):
    """Dueling-DQN Q-network (Wang et al., 2016).

    Shares a feature trunk, then branches into a 1-dim state-value stream V(s)
    and an n_actions-dim advantage stream A(s, a). Q values are reconstructed
    as Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a)) — the mean-centering form,
    which is the standard choice that avoids identifiability issues.

    Parameter count is comparable to the vanilla QNetwork at the same hidden
    sizes (an extra +1 head from V; the A head replaces vanilla's output).
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: Sequence[int] = (64, 64),
    ):
        super().__init__()
        # Shared feature trunk: all hidden layers except the last become the trunk;
        # the last hidden width feeds the two heads.
        trunk_layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden[:-1]:
            trunk_layers.append(nn.Linear(prev, h))
            trunk_layers.append(nn.ReLU(inplace=True))
            prev = h
        # If hidden has length 1, trunk is empty; we feed obs straight into the heads.
        self.trunk = nn.Sequential(*trunk_layers) if trunk_layers else nn.Identity()
        head_in = prev

        head_width = hidden[-1]
        # Value stream V(s): head_in -> head_width -> 1
        self.value_head = nn.Sequential(
            nn.Linear(head_in, head_width),
            nn.ReLU(inplace=True),
            nn.Linear(head_width, 1),
        )
        # Advantage stream A(s, a): head_in -> head_width -> n_actions
        self.advantage_head = nn.Sequential(
            nn.Linear(head_in, head_width),
            nn.ReLU(inplace=True),
            nn.Linear(head_width, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        feat = self.trunk(obs)
        v = self.value_head(feat)                       # (B, 1)
        a = self.advantage_head(feat)                   # (B, n_actions)
        # Mean-centered combination — standard Dueling form.
        return v + (a - a.mean(dim=1, keepdim=True))
