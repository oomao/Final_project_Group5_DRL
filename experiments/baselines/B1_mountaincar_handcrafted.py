"""B1-handcrafted (MountainCar-v0): third-party hand-shaped reward.

STATUS: PLACEHOLDER (author-written stand-in; replace with non-author teammate
version before final paper per `evaluation-criteria` Requirement
"Complete Baseline Set"). See `B1_handcrafted.py` for the 4-step replacement
protocol.

Design philosophy (textbook MountainCar shaping):
    * Native reward is -1 per step (sparse). Without help, DQN flounders.
    * Add bonus for being on the right side of the valley (height proxy).
    * Add bonus for absolute velocity (encourage momentum building).
    * Keep magnitudes small so the native -1 still drives "reach the goal fast".

Observation indices (MountainCar-v0):
    obs[0] = car position  (valley ~ -0.5, goal at >= 0.5)
    obs[1] = car velocity  (range [-0.07, 0.07])
"""


def reward(obs, action, next_obs, env_reward, terminated, truncated, info):
    pos = next_obs[0]
    vel = next_obs[1]

    # Native time penalty
    r = float(env_reward)

    # Height bonus: pos ranges roughly [-1.2, 0.6]; bonus ~0 in valley, +0.11 at goal.
    r += 0.1 * (pos + 0.5)

    # Velocity bonus: |vel| up to 0.07; bonus up to 0.35. Encourages momentum.
    r += 5.0 * abs(vel)

    return float(r)
