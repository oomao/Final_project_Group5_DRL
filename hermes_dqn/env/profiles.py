"""Environment profiles: per-env metadata for the Hermes-DQN pipeline.

Adding a new Gymnasium env to the experiment requires registering an
``EnvProfile`` here. The orchestrator looks up `obs_dim`, `success_threshold`,
default training budget, and the Gemma task-spec string from this registry.

LunarLander-v3 is the default and the original target of the project; CartPole-v1
was added as a secondary env to test cross-task generalization of the
"memory hurts" finding (see report `reports/final/comparison_report.md`).
"""

from __future__ import annotations

from dataclasses import dataclass

from hermes_dqn.llm.prompts import (
    ACROBOT_TASK_SPEC,
    CARTPOLE_TASK_SPEC,
    LUNARLANDER_TASK_SPEC,
    MOUNTAINCAR_TASK_SPEC,
)


@dataclass(frozen=True)
class EnvProfile:
    """All env-specific knobs in one place. Frozen so it's safe to pass around."""

    env_id: str  # gym.make() argument
    obs_dim: int  # observation vector length (assumes Box obs)
    n_actions: int  # discrete action count
    success_threshold: float  # env-native return for "success" in env_native eval
    default_episodes: int  # canonical training budget per the experiments-protocol
    task_spec: str  # Gemma prompt TASK section
    b1_reward_file: str  # path (repo-relative) to B1 hand-shaped reward
    eval_base_seed: int = 10000  # eval seed offset (disjoint from training)


LUNARLANDER_V3 = EnvProfile(
    env_id="LunarLander-v3",
    obs_dim=8,
    n_actions=4,
    success_threshold=200.0,
    default_episodes=1500,
    task_spec=LUNARLANDER_TASK_SPEC,
    b1_reward_file="experiments/baselines/B1_handcrafted.py",
)


CARTPOLE_V1 = EnvProfile(
    env_id="CartPole-v1",
    obs_dim=4,
    n_actions=2,
    # CartPole-v1 max return is 500 (500 steps alive). "Solved" by Gymnasium's
    # historical definition is mean >= 195 over 100 episodes. We use 475 (95%
    # of max) as the stricter success threshold so that B0 baseline doesn't
    # trivially hit 100% and obscure the comparison.
    success_threshold=475.0,
    # CartPole DQN converges in ~200-500 ep; 1500 is wasteful. 500 keeps
    # apples-to-apples per Mann-Whitney U while halving compute.
    default_episodes=500,
    task_spec=CARTPOLE_TASK_SPEC,
    b1_reward_file="experiments/baselines/B1_cartpole_handcrafted.py",
)


MOUNTAINCAR_V0 = EnvProfile(
    env_id="MountainCar-v0",
    obs_dim=2,
    n_actions=3,
    # Gymnasium's MountainCar-v0 returns -1 per step; max episode is 200 steps.
    # "Solved" = mean episode return >= -110 (i.e. reaches goal in <110 steps avg).
    success_threshold=-110.0,
    # MountainCar DQN typically solves in 200-500 ep when shaping is decent. 300
    # is enough to differentiate methods without wasting compute on already-solved.
    default_episodes=300,
    task_spec=MOUNTAINCAR_TASK_SPEC,
    b1_reward_file="experiments/baselines/B1_mountaincar_handcrafted.py",
)


ACROBOT_V1 = EnvProfile(
    env_id="Acrobot-v1",
    obs_dim=6,
    n_actions=3,
    # Acrobot-v1 returns -1 per step; max episode is 500 steps.
    # "Solved" = mean episode return >= -100 (i.e. swings up in <100 steps avg).
    success_threshold=-100.0,
    # Acrobot is harder than MountainCar but the swing-up phase converges in
    # ~500 ep with shaping. 500 episode budget keeps wall-time reasonable.
    default_episodes=500,
    task_spec=ACROBOT_TASK_SPEC,
    b1_reward_file="experiments/baselines/B1_acrobot_handcrafted.py",
)


ENV_PROFILES: dict[str, EnvProfile] = {
    p.env_id: p for p in [LUNARLANDER_V3, CARTPOLE_V1, MOUNTAINCAR_V0, ACROBOT_V1]
}


def get_profile(env_id: str) -> EnvProfile:
    """Return the EnvProfile for ``env_id`` or raise ValueError listing known envs."""
    if env_id not in ENV_PROFILES:
        raise ValueError(
            f"Unknown env_id: {env_id!r}. Known: {sorted(ENV_PROFILES.keys())}. "
            f"Add a new EnvProfile in hermes_dqn/env/profiles.py to support new envs."
        )
    return ENV_PROFILES[env_id]
