"""B1-handcrafted (CartPole-v1): third-party hand-shaped reward.

Per `openspec/specs/evaluation-criteria/spec.md` Requirement "Complete Baseline Set",
B1 must be authored by a *non-author* third party and frozen before evaluation.

STATUS: PLACEHOLDER (committed by primary author to unblock pipeline smoke tests).
See header of `B1_handcrafted.py` for the same 4-step replacement protocol.

Design philosophy (purposely simple, textbook CartPole shaping):
    * Penalize large pole tilt (most important — termination at 12 deg)
    * Penalize off-center cart position (termination at |x| > 2.4)
    * Mild penalty on angular velocity to discourage oscillation
    * Keep magnitudes small so total reward stays roughly in [-2, +1] per step
      (env's native +1 anchor preserved)

Observation indices (CartPole-v1, gymnasium):
    obs[0] = cart x position  (negative=left, positive=right)
    obs[1] = cart velocity
    obs[2] = pole angle in radians (positive = tilted right; |a|>0.21 ends ep)
    obs[3] = pole angular velocity
"""


def reward(obs, action, next_obs, env_reward, terminated, truncated, info):
    x = next_obs[0]
    angle = next_obs[2]
    v_angle = next_obs[3]

    # Native +1 anchor: we still want "alive" to be rewarded as the dominant
    # signal (env_reward = +1 every step).
    r = float(env_reward)

    # Tilt penalty (most informative — pole near upright should be valued).
    # |angle| termination bound is ~0.21 rad. Penalty of 0.5*|angle| reaches
    # -0.105 at the brink of termination, well under the +1 alive reward.
    r -= 0.5 * abs(angle)

    # Off-center penalty (normalized by |x|=2.4 termination bound).
    r -= 0.1 * (abs(x) / 2.4)

    # Mild angular-velocity dampening, encourages smooth control.
    r -= 0.05 * abs(v_angle)

    return float(r)
