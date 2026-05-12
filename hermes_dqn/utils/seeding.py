"""Single-call deterministic seeding across random, numpy, and torch."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Seed every RNG that affects DQN training.

    Covers: stdlib random, numpy, torch (CPU + CUDA), and cudnn determinism.
    Callers must additionally pass the seed to `env.reset(seed=...)` and
    `env.action_space.seed(...)` — see `hermes_dqn.env.lunar_lander.make_env`.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
