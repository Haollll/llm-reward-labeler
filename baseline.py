"""Load an RL-Zoo3-trained PPO baseline and evaluate it on the TRUE env reward.

Baselines are trained with RL Baselines3 Zoo (`train_baselines.py`), which saves a
SB3 model plus its VecNormalize statistics under
`<baseline_dir>/ppo/<env_id>_<run>/`. Because the normalization stats are saved
alongside the weights, these baselines reload faithfully — no warmup hacks needed.

We evaluate on the raw environment reward (VecNormalize with `norm_reward=False`),
over n_episodes, and cache the result to artifacts/<env>/baseline/metrics.json so the
plotting suite and evaluate.py can pick it up.

CLI:
    python baseline.py --env HalfCheetah-v4 --episodes 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import helper  # noqa: F401  (applies env-warning suppression on import)
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from paths import baseline_metrics_path

ALGO = "ppo"


def zoo_n_timesteps(env_id: str, algo: str = ALGO, default: int = 1_000_000) -> int:
    """Total training timesteps RL-Zoo3 uses for `env_id` (from its tuned
    hyperparameters). Falls back to `default` if the env has no entry."""
    import os
    import rl_zoo3
    import yaml

    yml = os.path.join(os.path.dirname(rl_zoo3.__file__), "hyperparams", f"{algo}.yml")
    cfg = yaml.safe_load(open(yml))
    entry = cfg.get(env_id) or {}
    n = entry.get("n_timesteps")
    return int(float(n)) if n is not None else default


def _latest_run(env_id: str, baseline_dir: str) -> Path:
    root = Path(baseline_dir) / ALGO
    runs = list(root.glob(f"{env_id}_*"))
    if not runs:
        raise FileNotFoundError(
            f"No RL-Zoo3 baseline for {env_id} under {root}. "
            f"Train one first:  python train_baselines.py --envs {env_id}"
        )

    def _run_id(p: Path) -> int:
        tail = p.name.rsplit("_", 1)[-1]
        return int(tail) if tail.isdigit() else -1

    return max(runs, key=_run_id)


def load_baseline(env_id: str, baseline_dir: str = "baselines"):
    """Return (model, eval_venv). The venv applies the saved VecNormalize obs
    stats (frozen) and returns the *raw* reward so we measure true return."""
    run = _latest_run(env_id, baseline_dir)
    model_zip = run / f"{env_id}.zip"
    if not model_zip.exists():  # rl-zoo sometimes names it best_model.zip
        alt = run / "best_model.zip"
        model_zip = alt if alt.exists() else model_zip
    model = PPO.load(str(model_zip), device="cpu")

    venv = DummyVecEnv([lambda: gym.make(env_id)])
    stats = run / env_id / "vecnormalize.pkl"
    if stats.exists():
        venv = VecNormalize.load(str(stats), venv)
        venv.training = False      # freeze running stats
        venv.norm_reward = False   # return raw reward → true env return
    return model, venv


def evaluate_baseline(
    env_id: str,
    n_episodes: int = 10,
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
        "env_id":              env_id,
        "source":              "rl-zoo3 (ppo)",
        "episode_env_rewards": returns,
        "episode_lengths":     lengths,
        "mean_return":         float(np.mean(returns)),
        "mean_length":         float(np.mean(lengths)),
        "success":             successes if success_fn is not None else None,
    }
    if cache:
        path = baseline_metrics_path(env_id, artifact_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate an RL-Zoo3-trained PPO baseline")
    p.add_argument("--env", default="HalfCheetah-v4")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--baseline-dir", default="baselines")
    p.add_argument("--artifact-dir", default="artifacts")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    res = evaluate_baseline(
        args.env, n_episodes=args.episodes,
        baseline_dir=args.baseline_dir, artifact_dir=args.artifact_dir, seed=args.seed,
    )
    print(f"{args.env} baseline | {args.episodes} ep | "
          f"return {res['mean_return']:.1f} | length {res['mean_length']:.0f}")
    print(f"  saved → {baseline_metrics_path(args.env, args.artifact_dir)}")


if __name__ == "__main__":
    main()
