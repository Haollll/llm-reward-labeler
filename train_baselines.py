"""Train PPO baselines with RL Baselines3 Zoo (rl_zoo3) for the supported envs.

RL-Zoo3 ships tuned PPO hyperparameters for each env and saves the trained model
together with its VecNormalize statistics, so the baselines reload faithfully
(unlike weights-only checkpoints). Output goes to
`<baseline_dir>/ppo/<env_id>_<run>/`, which baseline.py / evaluate.py read.

    python train_baselines.py                              # all six envs
    python train_baselines.py --envs HalfCheetah-v4 Ant-v4
    python train_baselines.py --n-timesteps 1000000        # override per-env default

Pass --n-timesteps -1 (default) to use each env's tuned default budget.
"""

import argparse
import subprocess
import sys

import helper  # noqa: F401  (env-warning suppression)
from helper import SUPPORTED_ENVS

DEFAULT_ENVS = SUPPORTED_ENVS


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO baselines with rl_zoo3")
    p.add_argument("--envs", nargs="*", default=DEFAULT_ENVS)
    p.add_argument("--baseline-dir", default="baselines")
    p.add_argument("--n-timesteps", type=int, default=-1,
                   help="-1 uses each env's tuned default budget")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results = {}
    for env_id in args.envs:
        print("\n" + "=" * 70)
        print(f"  TRAINING BASELINE: {env_id}")
        print("=" * 70)
        cmd = [
            sys.executable, "-m", "rl_zoo3.train",
            "--algo", "ppo",
            "--env", env_id,
            "--log-folder", args.baseline_dir,
            "--seed", str(args.seed),
        ]
        if args.n_timesteps and args.n_timesteps > 0:
            cmd += ["--n-timesteps", str(args.n_timesteps)]
        print("  $ " + " ".join(cmd))
        rc = subprocess.call(cmd)
        results[env_id] = "ok" if rc == 0 else f"FAILED (exit {rc})"

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for env_id, status in results.items():
        print(f"  {env_id:20} {status}")
    print(f"\nBaselines saved under: {args.baseline_dir}/ppo/<env>_<run>/")
    print("Evaluate with:  python baseline.py --env <env>")


if __name__ == "__main__":
    main()
