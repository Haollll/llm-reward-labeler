import argparse
from trainer import Trainer, TrainerConfig


def parse_args() -> TrainerConfig:
    p = argparse.ArgumentParser(description="LLM reward labeller training")
    p.add_argument("--env",           default="HalfCheetah-v5")
    p.add_argument("--task",          default="halfcheetah")
    p.add_argument("--oracle",        action="store_true",
                   help="Use oracle labels instead of LLM (ablation)")
    p.add_argument("--rounds",        type=int, default=9)
    p.add_argument("--queries",       type=int, default=10)
    p.add_argument("--ppo-steps",     type=int, default=20_000)
    p.add_argument("--reward-epochs", type=int, default=50)
    p.add_argument("--eval-every",    type=int, default=2)
    p.add_argument("--model",         default="gpt-4o-mini")
    p.add_argument("--quiet",         action="store_true")
    args = p.parse_args()

    return TrainerConfig(
        env_id        = args.env,
        task_name     = args.task,
        use_oracle    = args.oracle,
        n_queries     = args.queries,
        ppo_steps     = args.ppo_steps,
        reward_epochs = args.reward_epochs,
        eval_every    = args.eval_every,
        llm_model     = args.model,
        verbose       = not args.quiet,
    )


if __name__ == "__main__":
    cfg     = parse_args()
    trainer = Trainer(cfg)
    trainer.run(n_rounds=cfg.rounds if hasattr(cfg, 'rounds') else 9)