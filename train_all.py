"""Train the LLM-reward pipeline on several MuJoCo v4 envs in sequence.

Each env writes into its own directory (artifacts/<env_id>/...), so runs stay
organized and can be plotted/compared afterwards with make_plots.py.

    python train_all.py                               # all six default envs
    python train_all.py --envs Hopper-v4 Ant-v4       # a subset
    python train_all.py --rounds 9 --ppo-steps 50000  # shared hyperparameters
"""

import argparse
import traceback

import helper  # noqa: F401  (env-warning suppression)
from helper import task_for_env, SUPPORTED_ENVS
from trainer import Trainer, TrainerConfig
from train import default_ppo_steps

DEFAULT_ENVS = SUPPORTED_ENVS


def parse_args():
    p = argparse.ArgumentParser(description="Train the pipeline across multiple envs")
    p.add_argument("--envs", nargs="*", default=DEFAULT_ENVS)
    p.add_argument("--rounds", type=int, default=9)
    p.add_argument("--queries", type=int, default=10)
    p.add_argument("--ppo-steps", type=int, default=None,
                   help="PPO steps per round. Default (per env): rl-zoo3 total "
                        "timesteps for the env / (rounds + 1).")
    p.add_argument("--reward-epochs", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--artifact-dir", default="artifacts")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--lambda-smooth", type=float, default=0.05)
    p.add_argument("--dynamic-batch", action="store_true")
    p.add_argument("--no-progress-bar", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results = {}
    for env_id in args.envs:
        ppo_steps = (args.ppo_steps if args.ppo_steps is not None
                     else default_ppo_steps(env_id, args.rounds))
        print("\n" + "=" * 70)
        print(f"  TRAINING {env_id}  (ppo-steps/round = {ppo_steps})")
        print("=" * 70)
        cfg = TrainerConfig(
            env_id=env_id,
            task_name=task_for_env(env_id),
            n_queries=args.queries,
            ppo_steps=ppo_steps,
            reward_epochs=args.reward_epochs,
            eval_every=args.eval_every,
            rounds=args.rounds,
            artifact_dir=args.artifact_dir,
            progress_bar=not args.no_progress_bar,
            llm_model=args.model,
            verbose=not args.quiet,
            lambda_smooth=args.lambda_smooth,
            dynamic_batch=args.dynamic_batch,
        )
        try:
            trainer = Trainer(cfg)
            trainer.run(n_rounds=cfg.rounds)
            results[env_id] = trainer.eval_rewards[-1] if trainer.eval_rewards else None
        except Exception:
            print(f"[{env_id}] FAILED:\n{traceback.format_exc()}")
            results[env_id] = "FAILED"

    print("\n" + "=" * 70)
    print("  SUMMARY (final eval return)")
    print("=" * 70)
    for env_id, r in results.items():
        val = f"{r:.1f}" if isinstance(r, float) else str(r)
        print(f"  {env_id:24} {val}")


if __name__ == "__main__":
    main()
