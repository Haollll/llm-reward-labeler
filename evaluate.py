import argparse
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from env_setup import CustomRewardWrapper
from helper import load_task
from llm_utils import cache_key, generate_reward_fn
from reward import CompositeReward
from reward_model import RewardModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved PPO policy and reward model")
    parser.add_argument("--env", default="HalfCheetah-v5")
    parser.add_argument("--task", default="halfcheetah")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--policy-path", default=None)
    parser.add_argument("--reward-model-dir", default=None)
    parser.add_argument("--plot-path", default=None)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def default_paths(env_id: str, task_name: str, artifact_dir: str):
    task = load_task(task_name)
    env = gym.make(env_id)
    try:
        run_id = cache_key(env, task)
    finally:
        env.close()

    root = Path(artifact_dir)
    return (
        root / "policies" / run_id / "policy.zip",
        root / "reward_models" / run_id,
        root / "plots" / run_id / "episode_rewards.png",
    )


def main() -> None:
    args = parse_args()
    task = load_task(args.task)
    default_policy, default_reward_model, default_plot = default_paths(
        args.env, args.task, args.artifact_dir
    )
    policy_path = Path(args.policy_path) if args.policy_path else default_policy
    reward_model_dir = Path(args.reward_model_dir) if args.reward_model_dir else default_reward_model
    plot_path = Path(args.plot_path) if args.plot_path else default_plot

    render_mode = "human" if args.render else None
    base_env = gym.make(args.env, render_mode=render_mode)
    reward_model = RewardModel(base_env)
    reward_model.load(str(reward_model_dir))
    _, reward_fn = generate_reward_fn(base_env, task, model=args.model)
    composite_reward = CompositeReward(
        r_fixed=reward_fn,
        cache_key=cache_key(base_env, task),
        reward_model=reward_model,
        alpha=args.alpha,
    )
    env = CustomRewardWrapper(base_env, composite_reward)
    policy = PPO.load(str(policy_path), device="cpu")

    combined_rewards = []
    true_rewards = []
    obs, _ = env.reset()
    done = False

    try:
        while not done:
            action, _ = policy.predict(obs, deterministic=args.deterministic)
            next_obs, combined_reward, terminated, truncated, info = env.step(action)
            combined_rewards.append(float(combined_reward))
            true_rewards.append(float(info["env_reward"]))
            obs = next_obs
            done = terminated or truncated
    finally:
        env.close()

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    steps = range(len(true_rewards))
    plt.figure(figsize=(10, 5))
    plt.plot(steps, true_rewards, label="True env reward")
    plt.plot(steps, combined_rewards, label="Combined reward")
    plt.xlabel("Episode step")
    plt.ylabel("Reward")
    plt.title(f"{args.env} reward traces (alpha={args.alpha:.2f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"Episode steps       : {len(true_rewards)}")
    print(f"Total true reward   : {sum(true_rewards):.3f}")
    print(f"Total combined reward: {sum(combined_rewards):.3f}")
    print(f"Plot saved          : {plot_path}")


if __name__ == "__main__":
    main()
