## 1. Project Skeleton

- [x] 1.1 Create `hermes_dqn/` package tree (`env/`, `agent/`, `training/`, `utils/`) with empty `__init__.py` files
- [x] 1.2 Create `pyproject.toml` declaring the package (setuptools build backend, Python ≥ 3.11)
- [x] 1.3 Create `requirements.txt` with `torch~=2.5.0`, `gymnasium[box2d]~=1.0`, `numpy~=2.0`, `tqdm~=4.66`
- [x] 1.4 Update root `.gitignore` to exclude `runs/`, `__pycache__/`, `*.pt`, `.venv/`, `*.egg-info/`
- [x] 1.5 Create `hermes_dqn/README.md` with install steps (incl. Windows Box2D note) and a "first training run" snippet

## 2. Utilities

- [x] 2.1 Implement `hermes_dqn/utils/seeding.py::set_global_seed(seed)` covering `random`, `numpy`, `torch`, CUDA, and cudnn determinism flags

## 3. Env Layer (capability: reward-plugin)

- [x] 3.1 Define `RewardFunction` Protocol in `hermes_dqn/env/reward.py` with the 7-arg signature from design.md
- [x] 3.2 Implement `default_reward_fn` (passthrough) in the same module for documentation purposes
- [x] 3.3 Implement `hermes_dqn/env/lunar_lander.py::make_env(seed, reward_fn=None)` that returns a Gymnasium env wrapper applying `reward_fn` on every `step()` when provided
- [x] 3.4 Confirm `env.reset(seed=...)` and `env.action_space.seed(seed)` are both wired so reseeding the wrapper reseeds both
- [x] 3.5 Verify reward-plugin spec scenarios: passthrough, custom shaping, exception propagation, full-transition access

## 4. Agent Layer (capability: dqn-baseline)

- [x] 4.1 Implement `hermes_dqn/agent/q_network.py::QNetwork` — MLP 64-64 ReLU, configurable `obs_dim` / `n_actions`
- [x] 4.2 Implement `hermes_dqn/agent/replay_buffer.py::ReplayBuffer` — numpy circular buffer, `push` / `sample(batch_size)` / `__len__`
- [x] 4.3 Implement `hermes_dqn/agent/dqn_agent.py::DQNAgent` with `act(obs, epsilon)`, `step(transition)`, `learn()`, `save(path)`, `load(path)`; auto-detect CUDA
- [x] 4.4 Hard-copy target-network sync every `target_update_interval` env steps inside `DQNAgent.step()` (or `learn()`)
- [x] 4.5 Linear ε decay from 1.0 → 0.01 over `epsilon_decay_steps`; computed from a step counter held by the agent
- [x] 4.6 Skip gradient updates until `len(buffer) >= train_start`

## 5. Training Loop (capability: dqn-baseline)

- [x] 5.1 Implement `hermes_dqn/training/logger.py::JsonlLogger` writing one line per episode to `episodes.jsonl`
- [x] 5.2 Implement `hermes_dqn/training/train.py::main` with argparse: `--episodes`, `--seed`, `--config` (JSON file), `--out-dir`
- [x] 5.3 At run start, create `runs/<YYYY-MM-DD_HH-MM-SS>/`, write `config.json` with all hyperparams + seed + env name
- [x] 5.4 Run training loop: env step → agent.step → agent.learn → episode bookkeeping → log one row per episode
- [x] 5.5 Save `model_final.pt` after the last episode
- [x] 5.6 Make `python -m hermes_dqn.training.train` work (i.e. `__main__` guard wired correctly)

## 6. Fitness Evaluation (capability: fitness-evaluation)

- [x] 6.1 Implement `hermes_dqn/training/fitness.py::FitnessReport` dataclass with the 5 fields from spec
- [x] 6.2 Implement `FitnessEvaluator(success_threshold=200.0, window=100)` with `evaluate(jsonl_path) -> FitnessReport`
- [x] 6.3 Cover the three scenarios: standard converged run, never-converges, fewer-than-100 episodes
- [x] 6.4 Reading config.json for the `seed` field to embed in the report

## 7. Smoke Test

- [x] 7.1 Run `python -m hermes_dqn.training.train --episodes 10 --seed 42` and confirm `runs/<ts>/` contains `config.json`, `episodes.jsonl` (10 rows), `model_final.pt`
- [x] 7.2 Run `FitnessEvaluator().evaluate(...)` on the smoke run's JSONL; confirm `total_episodes == 10`, `converge_episode is None`, no exceptions
- [x] 7.3 Re-run with the same seed and assert episode returns match byte-for-byte (determinism check)

## 8. Baseline Convergence Run

- [x] 8.1 Run `python -m hermes_dqn.training.train --episodes 1500 --seed 42` to completion
- [x] 8.2 Run fitness evaluator on the result; confirm `mean_reward_last100 ≥ 200.0` and `success_rate ≥ 0.90` (got 262.79 / 0.95)
- [x] 8.3 If convergence targets are not met, tune `lr` / `target_update_interval` / `epsilon_decay_steps` and re-run (NOT NEEDED — targets met on first try)
- [x] 8.4 Append a one-line baseline summary (date, seed, fitness metrics) to `hermes_dqn/README.md` for future comparison

## 9. Wrap-up

- [x] 9.1 `openspec validate bootstrap-dqn-baseline --strict` passes
- [x] 9.2 All scenarios from `specs/dqn-baseline/spec.md`, `specs/reward-plugin/spec.md`, `specs/fitness-evaluation/spec.md` verified (smoke test for dqn-baseline + reward-plugin determinism; fitness-evaluation tested on 10-ep and 1500-ep runs)
- [x] 9.3 Ready to `/opsx:archive` once `02-ending.sh` writes the next handover
