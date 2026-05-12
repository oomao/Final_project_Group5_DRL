"""Entry point: ``python -m hermes_dqn.training.train``.

Trains a vanilla DQN on LunarLander-v3, writes per-episode metrics to
``runs/<timestamp>/episodes.jsonl``, persists the final model, and stores
every hyperparameter alongside in ``config.json`` for reproducibility.

The ``--reward-source`` flag selects between the env's native reward
(``env``, default — identical to the bootstrap-dqn-baseline behavior) and
an LLM-generated reward function from Gemma (``llm``). Both paths write
``reward_fn.py`` + ``reward_fn_sha256`` for traceability.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from hermes_dqn.agent.dqn_agent import DQNAgent, DQNConfig
from hermes_dqn.env.lunar_lander import make_env
from hermes_dqn.training.logger import JsonlLogger
from hermes_dqn.utils.seeding import set_global_seed


_ENV_STUB_REWARD_SRC = (
    "# env native reward (no custom function)\n"
    "def reward(obs, action, next_obs, env_reward, terminated, truncated, info):\n"
    "    return float(env_reward)\n"
)


@dataclass
class TrainConfig:
    """Top-level training config: env, seed, episode budget, output, DQN hyperparams."""

    env_id: str = "LunarLander-v3"
    seed: int = 42
    episodes: int = 1500
    max_steps_per_episode: int = 1000
    out_dir: str | None = None
    reward_source: str = "env"
    dqn: DQNConfig = field(default_factory=DQNConfig)

    @classmethod
    def from_overrides(cls, overrides: dict) -> "TrainConfig":
        """Build a TrainConfig from a flat or nested override dict.

        Nested ``dqn`` hyperparams can be passed under the ``"dqn"`` key
        (recommended) or as top-level fields matching DQNConfig field names.
        """
        cfg = cls()
        dqn_overrides = dict(overrides.pop("dqn", {})) if "dqn" in overrides else {}
        for key, value in overrides.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            elif hasattr(cfg.dqn, key):
                dqn_overrides[key] = value
            else:
                raise ValueError(f"Unknown config field: {key}")
        for key, value in dqn_overrides.items():
            if not hasattr(cfg.dqn, key):
                raise ValueError(f"Unknown DQN config field: {key}")
            setattr(cfg.dqn, key, value)
        if cfg.reward_source not in ("env", "llm"):
            raise ValueError(f"reward_source must be 'env' or 'llm' (got {cfg.reward_source!r})")
        return cfg

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _make_run_dir(out_dir: str | None) -> Path:
    if out_dir is not None:
        path = Path(out_dir)
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = Path("runs") / ts
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_reward(config: TrainConfig, run_dir: Path):
    """Return ``(source_str, reward_fn_or_None)`` for the configured reward_source.

    - ``env``: a passthrough stub is returned as source; reward_fn is None so
      make_env uses the env's native reward directly (zero overhead).
    - ``llm``: Gemma is called via LLMRewardClient with up to 3 retries.
      The validated source string is returned along with the compiled callable.
      On generation failure, prints to stderr and ``sys.exit(1)`` BEFORE training.
    """
    if config.reward_source == "env":
        return _ENV_STUB_REWARD_SRC, None

    if config.reward_source != "llm":
        raise ValueError(f"Unknown reward_source: {config.reward_source!r}")

    # Lazy import so the env path doesn't pay for google-genai import time
    from dotenv import load_dotenv

    from hermes_dqn.llm import LLMRewardClient, RewardGenerationError, compile_reward

    load_dotenv()

    try:
        client = LLMRewardClient()
    except ValueError as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[llm] Asking {client.model} for a reward function...", flush=True)
    attempts_log = run_dir / "llm_attempts.jsonl"
    try:
        src = client.generate(attempts_log_path=attempts_log)
    except RewardGenerationError as e:
        print(
            f"[FAIL] LLM generation failed after {len(e.attempts)} attempts. "
            f"See {attempts_log}:",
            file=sys.stderr,
        )
        for a in e.attempts:
            print(f"  attempt {a.attempt}: {a.error}", file=sys.stderr)
        sys.exit(1)

    fn = compile_reward(src)
    print(f"[llm] Accepted reward function ({len(src.splitlines())} lines).", flush=True)
    return src, fn


def train(config: TrainConfig) -> Path:
    """Run a full training session and return the run directory."""
    set_global_seed(config.seed)
    run_dir = _make_run_dir(config.out_dir)
    config.out_dir = str(run_dir)

    # Resolve reward BEFORE writing config.json so we can record SHA-256 there.
    # If --reward-source llm fails, this exits non-zero before training starts.
    reward_src, reward_fn = _resolve_reward(config, run_dir)

    # write_bytes (not write_text) so Windows newline translation does not
    # corrupt the SHA-256 — bytes on disk match exactly what we hashed.
    reward_fn_path = run_dir / "reward_fn.py"
    reward_src_bytes = reward_src.encode("utf-8")
    reward_fn_path.write_bytes(reward_src_bytes)
    reward_fn_sha256 = hashlib.sha256(reward_src_bytes).hexdigest()

    config_data = config.to_dict()
    config_data["reward_fn_sha256"] = reward_fn_sha256
    with (run_dir / "config.json").open("w", encoding="utf-8") as fp:
        json.dump(config_data, fp, indent=2)

    env = make_env(seed=config.seed, reward_fn=reward_fn)
    obs_space = env.observation_space
    act_space = env.action_space
    config.dqn.obs_dim = int(obs_space.shape[0])
    config.dqn.n_actions = int(act_space.n)

    agent = DQNAgent(config.dqn, seed=config.seed)

    jsonl_path = run_dir / "episodes.jsonl"
    start = time.time()
    with JsonlLogger(jsonl_path) as logger:
        pbar = tqdm(range(1, config.episodes + 1), desc="train", unit="ep")
        for ep in pbar:
            obs, _ = env.reset(seed=config.seed + ep)
            ep_return = 0.0
            ep_length = 0
            losses: list[float] = []
            terminated = truncated = False

            while not (terminated or truncated):
                action = agent.act(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.step(obs, action, reward, next_obs, done)
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)
                obs = next_obs
                ep_return += reward
                ep_length += 1
                if ep_length >= config.max_steps_per_episode:
                    break

            logger.log(
                {
                    "episode": ep,
                    "return": ep_return,
                    "length": ep_length,
                    "epsilon": agent.epsilon(),
                    "loss_mean": float(sum(losses) / len(losses)) if losses else 0.0,
                    "wall_time_s": round(time.time() - start, 2),
                }
            )
            pbar.set_postfix(ret=f"{ep_return:.1f}", eps=f"{agent.epsilon():.2f}")

    agent.save(run_dir / "model_final.pt")
    env.close()
    return run_dir


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train vanilla DQN on LunarLander-v3.")
    p.add_argument("--episodes", type=int, default=None, help="Override episode budget.")
    p.add_argument("--seed", type=int, default=None, help="Override RNG seed.")
    p.add_argument("--config", type=str, default=None, help="Path to JSON config overrides.")
    p.add_argument("--out-dir", type=str, default=None, help="Override run output directory.")
    p.add_argument(
        "--reward-source",
        choices=["env", "llm"],
        default=None,
        help="Reward source: 'env' (native, baseline) or 'llm' (Gemma-generated). Default: env.",
    )
    return p


def main() -> None:
    args = _build_argparser().parse_args()

    overrides: dict = {}
    if args.config is not None:
        with Path(args.config).open("r", encoding="utf-8") as fp:
            overrides.update(json.load(fp))
    if args.episodes is not None:
        overrides["episodes"] = args.episodes
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.out_dir is not None:
        overrides["out_dir"] = args.out_dir
    if args.reward_source is not None:
        overrides["reward_source"] = args.reward_source

    config = TrainConfig.from_overrides(overrides)
    run_dir = train(config)
    print(f"\n[OK] Run complete. Artifacts written to: {run_dir}")


if __name__ == "__main__":
    main()
