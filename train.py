import argparse
from trainer import Trainer, TrainerConfig
from helper import task_for_env
from baseline import zoo_n_timesteps


def default_ppo_steps(env_id: str, rounds: int) -> int:
    """Match the rl-zoo3 baseline's total budget, split across the training
    phases: zoo total timesteps / (rounds + 1). The +1 accounts for the cold
    start phase."""
    return max(1, zoo_n_timesteps(env_id) // (rounds + 1))


def parse_args() -> TrainerConfig:
    p = argparse.ArgumentParser(description="LLM reward labeller training")
    p.add_argument("--env", default="HalfCheetah-v4")
    p.add_argument("--task", default=None,
                   help="Task file name (without .txt). Defaults to the env's "
                        "task via task_for_env().")
    p.add_argument("--rounds", type=int, default=9)
    p.add_argument("--queries", type=int, default=10)
    p.add_argument("--ppo-steps", type=int, default=None,
                   help="PPO steps per round. Default: rl-zoo3 total timesteps for "
                        "the env / (rounds + 1).")
    p.add_argument("--reward-epochs", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--artifact-dir", default="artifacts")
    p.add_argument("--no-progress-bar", action="store_true")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--lambda-smooth", type=float, default=1.0)
    p.add_argument("--dynamic-batch", action="store_true")
    args = p.parse_args()
    task_name = args.task or task_for_env(args.env)
    ppo_steps = args.ppo_steps if args.ppo_steps is not None else default_ppo_steps(args.env, args.rounds)

    return TrainerConfig(
        env_id=args.env,
        task_name=task_name,
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


if __name__ == "__main__":
    cfg = parse_args()
    trainer = Trainer(cfg)
    trainer.run(n_rounds=cfg.rounds)
