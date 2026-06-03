"""Train the v2 two-phase pipeline across multiple envs in sequence.

    python train_all.py                               # all six default envs
    python train_all.py --envs Hopper-v4 Ant-v4       # a subset
"""

import argparse
import traceback

import helper  # noqa: F401  (env-warning suppression)
from helper import task_for_env, SUPPORTED_ENVS
from trainer import Trainer, TrainerConfig


def parse_args():
    p = argparse.ArgumentParser(description="Train the v2 pipeline across envs")
    p.add_argument("--envs", nargs="*", default=SUPPORTED_ENVS)
    p.add_argument("--k1", type=int, default=5)
    p.add_argument("--k2", type=int, default=5)
    p.add_argument("--ppo-steps", type=int, default=100_000)
    p.add_argument("--num-trajs", type=int, default=10,
                   help="Trajectories per round; all C(num_trajs,2) pairs are compared")
    p.add_argument("--reward-epochs", type=int, default=50)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--no-reflect", action="store_true")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--reflect-model", default="gpt-4o")
    p.add_argument("--artifact-dir", default="artifacts")
    p.add_argument("--no-progress-bar", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results = {}
    for env_id in args.envs:
        print("\n" + "=" * 70)
        print(f"  TRAINING {env_id}")
        print("=" * 70)
        cfg = TrainerConfig(
            env_id=env_id,
            task_name=task_for_env(env_id),
            k1=args.k1, k2=args.k2, ppo_steps=args.ppo_steps,
            num_trajs=args.num_trajs,
            reward_epochs=args.reward_epochs, alpha=args.alpha,
            eval_episodes=args.eval_episodes, reflect=not args.no_reflect,
            llm_model=args.model, reflect_model=args.reflect_model,
            artifact_dir=args.artifact_dir,
            progress_bar=not args.no_progress_bar, verbose=not args.quiet,
        )
        try:
            t = Trainer(cfg)
            t.run()
            results[env_id] = t.best_return
        except Exception:
            print(f"[{env_id}] FAILED:\n{traceback.format_exc()}")
            results[env_id] = "FAILED"

    print("\n" + "=" * 70)
    print("  SUMMARY (best eval return)")
    print("=" * 70)
    for env_id, r in results.items():
        val = f"{r:.1f}" if isinstance(r, float) else str(r)
        print(f"  {env_id:24} {val}")


if __name__ == "__main__":
    main()
