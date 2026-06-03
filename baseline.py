"""Load an RL-Zoo3-trained PPO baseline and evaluate it on the TRUE env reward.

Reuses the baselines trained for v1 (the `baselines/` symlink points at them).
The SB3 model + frozen VecNormalize stats are reloaded and scored on the raw
environment reward, cached to artifacts/<env>/baseline/metrics.json.

    python baseline.py --env HalfCheetah-v4 --episodes 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import helper  # noqa: F401  (env-warning suppression)
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from paths import baseline_metrics_path

ALGO = "ppo"


def _latest_run(env_id: str, baseline_dir: str) -> Path:
    root = Path(baseline_dir) / ALGO
    runs = list(root.glob(f"{env_id}_*"))
    if not runs:
        raise FileNotFoundError(
            f"No RL-Zoo3 baseline for {env_id} under {root}. Train one with "
            f"rl_zoo3 (see the v1 train_baselines.py) or point --baseline-dir at it."
        )

    def _run_id(p: Path) -> int:
        tail = p.name.rsplit("_", 1)[-1]
        return int(tail) if tail.isdigit() else -1

    return max(runs, key=_run_id)


def load_baseline(env_id: str, baseline_dir: str = "baselines"):
    run = _latest_run(env_id, baseline_dir)
    model_zip = run / f"{env_id}.zip"
    if not model_zip.exists():
        alt = run / "best_model.zip"
        model_zip = alt if alt.exists() else model_zip
    model = PPO.load(str(model_zip), device="cpu")

    venv = DummyVecEnv([lambda: gym.make(env_id)])
    stats = run / env_id / "vecnormalize.pkl"
    if stats.exists():
        venv = VecNormalize.load(str(stats), venv)
        venv.training = False
        venv.norm_reward = False
    return model, venv


def evaluate_baseline(
    env_id: str,
    n_episodes: int = 100,
    baseline_dir: str = "baselines",
    artifact_dir: str = "artifacts",
    cache: bool = True,
    success_fn=None,
    seed: int = 1,
) -> Dict[str, object]:
    model, venv = load_baseline(env_id, baseline_dir)
    venv.seed(seed)
    obs = venv.reset()

    returns: List[float] = []
    lengths: List[int] = []
    successes: List[bool] = []
    ep_ret, ep_len = 0.0, 0
    while len(returns) < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = venv.step(action)
        ep_ret += float(reward[0])
        ep_len += 1
        if bool(done[0]):
            returns.append(ep_ret)
            lengths.append(ep_len)
            if success_fn is not None:
                successes.append(bool(success_fn(ep_len, ep_ret)))
            ep_ret, ep_len = 0.0, 0
    venv.close()

    result: Dict[str, object] = {
        "env_id": env_id,
        "source": "rl-zoo3 (ppo)",
        "episode_env_rewards": returns,
        "episode_lengths": lengths,
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
        "success": successes if success_fn is not None else None,
    }
    if cache:
        path = baseline_metrics_path(env_id, artifact_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate an RL-Zoo3-trained PPO baseline")
    p.add_argument("--env", default="HalfCheetah-v4")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--baseline-dir", default="baselines")
    p.add_argument("--artifact-dir", default="artifacts")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    res = evaluate_baseline(args.env, args.episodes, args.baseline_dir,
                            args.artifact_dir, seed=args.seed)
    print(f"{args.env} baseline | return {res['mean_return']:.1f} | "
          f"length {res['mean_length']:.0f}")


if __name__ == "__main__":
    main()
