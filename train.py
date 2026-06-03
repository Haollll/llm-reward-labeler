"""Single-env entry point for the v2 two-phase pipeline.

    python train.py --env HalfCheetah-v4
    python train.py --env Hopper-v4 --k1 5 --k2 4 --ppo-steps 100000
"""

import argparse

import helper  # noqa: F401  (env-warning suppression)
from helper import task_for_env
from trainer import Trainer, TrainerConfig


def parse_args() -> TrainerConfig:
    p = argparse.ArgumentParser(description="LLM reward labeller v2 (two-phase)")
    p.add_argument("--env", default="HalfCheetah-v4")
    p.add_argument("--task", default=None)
    p.add_argument("--k1", type=int, default=5, help="Phase I rounds (coded reward + reflection)")
    p.add_argument("--k2", type=int, default=5, help="Phase II rounds (mixed reward)")
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
    args = p.parse_args()

    return TrainerConfig(
        env_id=args.env,
        task_name=args.task or task_for_env(args.env),
        k1=args.k1,
        k2=args.k2,
        ppo_steps=args.ppo_steps,
        num_trajs=args.num_trajs,
        reward_epochs=args.reward_epochs,
        alpha=args.alpha,
        eval_episodes=args.eval_episodes,
        reflect=not args.no_reflect,
        llm_model=args.model,
        reflect_model=args.reflect_model,
        artifact_dir=args.artifact_dir,
        progress_bar=not args.no_progress_bar,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    cfg = parse_args()
    Trainer(cfg).run()
