## ADDED Requirements

### Requirement: RewardFunction protocol
The system SHALL define a `RewardFunction` typing.Protocol that any external reward function (hand-written, scripted, or LLM-generated) can satisfy without inheriting from any base class.

#### Scenario: Protocol signature
- **WHEN** a developer reads `hermes_dqn/env/reward.py`
- **THEN** they find a `RewardFunction` Protocol whose `__call__` accepts `(obs, action, next_obs, env_reward, terminated, truncated, info)` and returns `float`

#### Scenario: Duck-typed acceptance
- **WHEN** a plain Python function with the matching signature is passed as `reward_fn` to the env factory
- **THEN** the env wrapper accepts it without raising a type error

### Requirement: Injectable reward in env wrapper
The Gymnasium env wrapper SHALL accept an optional `reward_fn` at construction time and, on every `step()`, return the value produced by that function in place of the env's native reward.

#### Scenario: Default passthrough
- **WHEN** the env is constructed with `reward_fn=None`
- **THEN** every `step()` returns the env's native reward unchanged

#### Scenario: Custom shaping
- **WHEN** the env is constructed with a `reward_fn` that always returns `env_reward + 1.0`
- **THEN** every `step()` returns `env_reward + 1.0`
- **AND** the original `env_reward` is still passed into the function as the `env_reward` argument

#### Scenario: Reward function has access to full transition
- **WHEN** any `reward_fn` is called during `step()`
- **THEN** it receives the pre-step observation as `obs`, the action just taken as `action`, the post-step observation as `next_obs`, the env's native reward as `env_reward`, the `terminated` and `truncated` flags, and the env's `info` dict

### Requirement: Reward function failures are isolated
A reward function that raises an exception SHALL NOT corrupt training state; the exception propagates to the caller with a clear traceback identifying the reward-function source.

#### Scenario: Reward function raises
- **WHEN** a supplied `reward_fn` raises `ValueError("bad obs")` mid-episode
- **THEN** `env.step()` propagates the exception
- **AND** the agent's replay buffer is not mutated for that transition
