"""DQN agent: online + target Q-network, replay buffer, epsilon-greedy.

Supports three Rainbow-family variants via DQNConfig flags:
  - Vanilla DQN (Mnih et al., 2015) — default
  - Double DQN (van Hasselt et al., 2016) — use_double_dqn=True
  - Dueling DQN (Wang et al., 2016) — dueling=True

These are orthogonal — both flags can be combined ("Double + Dueling").
Additional Rainbow components (PER, Multi-Step, Noisy, Distributional) are
not implemented in this codebase; see paper section 6 (Future Work).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hermes_dqn.agent.q_network import DuelingQNetwork, QNetwork
from hermes_dqn.agent.replay_buffer import ReplayBuffer


@dataclass
class DQNConfig:
    """Hyperparameters for the DQN agent.

    Defaults aim to reproduce the IJRPR 2025 LunarLander baseline
    (~1200 episodes to converge, ~92% success rate). See design.md.

    Variant flags (orthogonal — can be combined):
        use_double_dqn: Double DQN target computation (van Hasselt 2016).
            False = vanilla max over target network (Mnih 2015).
            True = online net selects argmax action, target evaluates Q-value.
        dueling: use DuelingQNetwork architecture (Wang 2016) for both
            online and target networks. Splits hidden trunk into V + A heads
            recombined as Q = V + (A - mean(A)).
    """

    obs_dim: int = 8
    n_actions: int = 4
    hidden: tuple[int, ...] = (64, 64)
    gamma: float = 0.99
    lr: float = 5e-4
    buffer_capacity: int = 100_000
    batch_size: int = 64
    train_start: int = 1_000
    target_update_interval: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 50_000
    # Rainbow-family variant flags (default: vanilla DQN — preserves prior
    # experiments byte-identical when both flags are False).
    use_double_dqn: bool = False
    dueling: bool = False


class DQNAgent:
    """Owns the Q-network, target network, replay buffer, and ε-greedy policy.

    External API mirrors a typical training loop:

        agent.act(obs, epsilon=...)        # exploration via ε-greedy
        agent.step(obs, action, reward, next_obs, done)  # push transition
        loss = agent.learn()               # one SGD step (None if warming up)
        agent.save(path) / DQNAgent.load(path)
    """

    def __init__(self, config: DQNConfig, seed: int = 0, device: str | None = None):
        self.config = config
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Pick Q-network architecture based on dueling flag.
        QNetClass = DuelingQNetwork if config.dueling else QNetwork
        self.online = QNetClass(config.obs_dim, config.n_actions, config.hidden).to(self.device)
        self.target = QNetClass(config.obs_dim, config.n_actions, config.hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.lr)
        self.buffer = ReplayBuffer(config.buffer_capacity, config.obs_dim, seed=seed)
        self._rng = np.random.default_rng(seed)
        self._env_steps = 0

    def epsilon(self) -> float:
        """Current ε from a linear schedule, clamped to [epsilon_end, epsilon_start]."""
        c = self.config
        if c.epsilon_decay_steps <= 0:
            return c.epsilon_end
        progress = min(self._env_steps / c.epsilon_decay_steps, 1.0)
        return c.epsilon_start + (c.epsilon_end - c.epsilon_start) * progress

    def act(self, obs: np.ndarray, epsilon: float | None = None) -> int:
        """ε-greedy action selection. Uses self.epsilon() if epsilon is None."""
        eps = self.epsilon() if epsilon is None else epsilon
        if self._rng.random() < eps:
            return int(self._rng.integers(0, self.config.n_actions))
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.online(obs_t)
            return int(q.argmax(dim=1).item())

    def step(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Push the transition into the replay buffer and tick the env counter."""
        self.buffer.push(obs, action, reward, next_obs, done)
        self._env_steps += 1

    def learn(self) -> float | None:
        """One gradient step on a uniform mini-batch. Returns the loss, or None if warming up."""
        c = self.config
        if len(self.buffer) < c.train_start:
            return None

        batch = self.buffer.sample(c.batch_size)
        obs = torch.as_tensor(batch.obs, device=self.device)
        actions = torch.as_tensor(batch.actions, device=self.device)
        rewards = torch.as_tensor(batch.rewards, device=self.device)
        next_obs = torch.as_tensor(batch.next_obs, device=self.device)
        dones = torch.as_tensor(batch.dones, device=self.device)

        q_pred = self.online(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if c.use_double_dqn:
                # Double DQN: ONLINE selects the argmax action, TARGET evaluates
                # the Q-value at that action. Decouples action selection from
                # value estimation, reducing overestimation bias.
                next_actions = self.online(next_obs).argmax(dim=1, keepdim=True)
                q_next = self.target(next_obs).gather(1, next_actions).squeeze(1)
            else:
                # Vanilla DQN: TARGET both selects and evaluates (max).
                q_next = self.target(next_obs).max(dim=1).values
            q_target = rewards + c.gamma * (1.0 - dones) * q_next

        loss = F.smooth_l1_loss(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()

        if self._env_steps % c.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.item())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "env_steps": self._env_steps,
                "config": self.config.__dict__,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        # weights_only=True: checkpoints come from our own save() — no exec risk
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._env_steps = int(ckpt.get("env_steps", 0))
