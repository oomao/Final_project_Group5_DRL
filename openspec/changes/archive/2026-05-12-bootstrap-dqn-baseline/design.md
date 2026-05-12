## Context

The repository today contains documentation, OpenSpec scaffolding, and dev-session shell scripts — but no Python source. Hermes-DQN's full closed-loop design (LLM generates reward → AST/Buffer manages replay → DQN trains → fitness feeds back to memory) needs a concrete DQN training pipeline to attach to. This change builds only that pipeline, in a way that lets every later module plug in without touching the trainer.

Constraints:
- Windows 11 dev environment (paths use backslashes), but training must also run unchanged on Linux (collaborators / cloud runs)
- LunarLander-v3 requires `gymnasium[box2d]`, which on Windows pulls `swig` + `Box2D` wheels — needs to be documented in the README so install doesn't silently fail
- The literature target (IJRPR 2025) is **~1200 episodes to converge, 92% success rate** on LunarLander-v3 — our baseline should reproduce roughly this range
- No GPU is assumed; DQN training on LunarLander runs comfortably on CPU but benefits from CUDA if available

Stakeholders: solo developer (csm088220@gmail.com) building the project for a DRL course final.

## Goals / Non-Goals

**Goals:**
- A working `python -m hermes_dqn.training.train` command that trains DQN on LunarLander-v3 end-to-end
- Reward function is **injected**, not hard-coded — the env wrapper accepts a `RewardFunction` callable; if none is supplied, it passes through the env's native reward
- A `FitnessEvaluator` that reads a training-run log (JSON lines) and returns metrics matching the proposal's contract
- Reproducible runs: seed every RNG (`numpy`, `torch`, `gymnasium`), persist config + seed alongside logs in `runs/<timestamp>/`
- Roughly matches IJRPR 2025 baseline numbers (mean reward ≥ 200 over last 100 eps, ≥ 90 % success) within tolerance

**Non-Goals:**
- Hermes Agent memory layers, Gemma API calls, AST analysis, replay-buffer surgery — all later changes
- Hyperparameter search / Rainbow extensions (Double, Dueling, PER) — vanilla DQN only
- Multiple environments (CartPole, Atari) — LunarLander-v3 only
- Distributed / vectorized training — single process, single env
- Web UI / TensorBoard server — file-based logs are enough at this stage

## Decisions

### Package layout
```
hermes_dqn/
├── __init__.py
├── env/
│   ├── __init__.py
│   ├── lunar_lander.py          # make_env(seed, reward_fn=None)
│   └── reward.py                # RewardFunction protocol + default passthrough
├── agent/
│   ├── __init__.py
│   ├── q_network.py             # MLP Q-network (PyTorch)
│   ├── replay_buffer.py         # uniform replay (numpy-backed circular buffer)
│   └── dqn_agent.py             # DQNAgent: act / step / learn / save / load
├── training/
│   ├── __init__.py
│   ├── train.py                 # entrypoint, CLI args, run loop
│   ├── logger.py                # JSONL writer for per-episode metrics
│   └── fitness.py               # FitnessEvaluator: log → metrics dict
└── utils/
    ├── __init__.py
    └── seeding.py               # one-call deterministic seed across libs
```

**Alternative considered:** flatter layout with everything in `hermes_dqn/*.py`. Rejected — once Hermes Agent + AST + Buffer manager arrive, the tree needs sub-packages anyway; doing it now avoids a churn-only rename later.

### Reward function as a plug-in (`RewardFunction` protocol)

```python
class RewardFunction(Protocol):
    def __call__(
        self,
        obs: np.ndarray,
        action: int,
        next_obs: np.ndarray,
        env_reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> float: ...
```

The env wrapper holds an optional `reward_fn`. On `step()`:
- If `reward_fn is None` → return `env_reward` untouched (baseline mode)
- Else → return `reward_fn(obs, action, next_obs, env_reward, terminated, truncated, info)`

**Why this signature:** Hermes-generated reward functions need access to `env_reward` (so they can shape *on top of* it) and `info` (for env-specific landing flags). Passing the action lets shaping penalize specific motor outputs. Passing both `obs` and `next_obs` covers potential-based shaping (`Φ(s') − Φ(s)`).

**Alternative considered:** Pass the full `gymnasium.Env` to the reward function. Rejected — couples reward authors to gym internals, and Hermes will need to call this from generated Python code where `env` reference may not exist.

### Fitness evaluator contract

```python
@dataclass
class FitnessReport:
    converge_episode: int | None      # first ep where mean(last 100) ≥ threshold; None if never
    mean_reward_last100: float
    success_rate: float               # fraction of last-100 eps with reward ≥ 200
    total_episodes: int
    seed: int

class FitnessEvaluator:
    def __init__(self, success_threshold: float = 200.0, window: int = 100): ...
    def evaluate(self, jsonl_path: Path) -> FitnessReport: ...
```

The `success_threshold = 200.0` and `window = 100` defaults come from the official LunarLander-v3 success criterion. Reading from a JSONL log (not in-memory) means Hermes' later async loop can score completed runs without holding training state.

### Training-run logging: JSONL not TensorBoard

Each episode appends one line to `runs/<timestamp>/episodes.jsonl`:
```json
{"episode": 42, "return": 187.3, "length": 312, "epsilon": 0.42, "loss_mean": 0.018, "wall_time_s": 12.4}
```
A separate `runs/<timestamp>/config.json` captures hyperparams + seed + env name.

**Why JSONL:** trivial to parse from Hermes/Python, no extra service, diffable in git if needed, and reward-function source code can be stored alongside as `reward_fn.py` for reproducibility.

**Alternative considered:** TensorBoard. Rejected for now — Hermes needs *programmatic* access to fitness metrics, not a UI. Can be added later as an additional sink.

### DQN hyperparameters (vanilla baseline)

| Param | Value | Source |
|---|---|---|
| Q-network | MLP 64-64 ReLU | Mnih et al. style, sufficient for LunarLander |
| γ | 0.99 | Standard |
| Replay buffer | 100 000 | Standard for LunarLander |
| Batch size | 64 | |
| Learning rate | 5e-4 (Adam) | Common default |
| ε start / end / decay | 1.0 / 0.01 / linear over 50 000 steps | |
| Target update | every 1000 env steps (hard copy) | Vanilla DQN |
| Train start | after 1000 env steps in buffer | |
| Train every | 1 env step | |
| Max episodes | 1500 (CLI override) | Allows reaching IJRPR 2025's ~1200 conv point with headroom |

These live in a `TrainConfig` dataclass; CLI flags override individual fields. Defaults aim to reproduce the IJRPR 2025 numbers, not to win the benchmark.

### Determinism

`hermes_dqn.utils.seeding.set_global_seed(seed)` sets:
- `random.seed`, `numpy.random.seed`
- `torch.manual_seed`, `torch.cuda.manual_seed_all`
- `torch.backends.cudnn.deterministic = True`, `benchmark = False`
- Passes seed into `env.reset(seed=...)` and `env.action_space.seed(seed)`

Default seed = 42. Same seed + same code + same OS ⇒ identical episode returns. This matters because Hermes' later memory entries need to be comparable across reward-function iterations.

### Dependency pinning

`requirements.txt` uses **lower-bound + compatible** (`~=`) rather than hard pins, except for `torch` which gets exact major.minor to avoid CUDA wheel surprises:
- `torch~=2.5.0`
- `gymnasium[box2d]~=1.0`
- `numpy~=2.0`
- `tqdm~=4.66`

`pyproject.toml` declares the package with `[build-system]` = `setuptools`, so `pip install -e .` works for local development.

## Risks / Trade-offs

- **Risk:** Box2D wheel install fails on Windows → **Mitigation:** README documents `swig` + Visual C++ Build Tools prerequisites; CI later can pre-pull wheels
- **Risk:** Vanilla DQN doesn't converge within 1500 eps (worse than IJRPR baseline) → **Mitigation:** smoke-test convergence as part of the `## 4. Verify Baseline` task; if it fails, tune target-update / lr before declaring done
- **Risk:** Reward plug-in signature has to change once Hermes generates code → **Mitigation:** signature is broad on purpose (all 7 args), but it's behind a Protocol — later changes can adapt
- **Trade-off:** JSONL-only logging means no live training curves in a UI; acceptable until a stakeholder asks for it
- **Trade-off:** No PER/Double-DQN means baseline numbers may sit at the low end of the IJRPR 2025 range; that's fine — fitness is *relative* to baseline across Hermes iterations, not absolute

## Migration Plan

This is a greenfield addition. No data migration. To deploy:
1. Create `hermes_dqn/` tree with stubs (empty `__init__.py` etc.) so imports work
2. Implement bottom-up: utils → env → agent → training → CLI
3. Smoke test with `--episodes 50` on CPU to confirm wiring
4. Full baseline run with `--episodes 1500` to verify convergence
5. Commit `runs/2026-05-12_baseline/episodes.jsonl` summary (not the full log) into `docx/` or `README.md` so future PRs can compare

Rollback: delete the `hermes_dqn/` directory and the two top-level config files — nothing else in the repo depends on them.

## Open Questions

- None blocking. Open for later: should `FitnessReport` include training wall-time as a comparison axis? Probably yes once Hermes iterates — defer until needed.
