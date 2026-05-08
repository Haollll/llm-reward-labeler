from typing import Any, Callable, List, Tuple

import gymnasium as gym
import torch

from llm_utils import compare_trajectories, generate_reward_fn, generate_semantic_fn
from helper import load_task, section

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


class CustomRewardWrapper(gym.RewardWrapper):
    """Gym wrapper that replaces the environment reward with a custom reward function."""
    def __init__(self, env: gym.Env, reward_fn: Callable[[Any, Any, Any], float]):
        super().__init__(env)
        self._reward_fn = reward_fn
        self._prev_obs: Any = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_obs = obs
        return obs, info

    def step(self, action):
        prev_obs = self._prev_obs
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        self._prev_obs = obs
        info["env_reward"] = float(env_reward)
        custom_reward = float(self._reward_fn(prev_obs, action, obs))
        return obs, custom_reward, terminated, truncated, info

    def reward(self, reward): # required by gym.RewardWrapper
        return reward


def collect_trajectory(env, policy_fn, max_steps: int = 100) -> List[Tuple]:
    trajectory: List[Tuple] = []
    obs, _ = env.reset()
    for _ in range(max_steps):
        action = policy_fn(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        trajectory.append((obs, action, next_obs, reward, done))
        obs = next_obs
        if done:
            break
    return trajectory


if __name__ == "__main__":
    task = load_task("halfcheetah")

    env = gym.make("HalfCheetah-v5")
    reward_code, reward_fn = generate_reward_fn(env, task)
    semantic_code, semantic_fn = generate_semantic_fn(env, task)

    wrapped_env = CustomRewardWrapper(env, reward_fn)

    print(section("Task"))
    print(task)
    print(section("Generated reward code"))
    print(reward_code)
    print(section("Generated semantic code"))
    print(semantic_code)

    random_policy = lambda obs: wrapped_env.action_space.sample()
    traj_a = collect_trajectory(wrapped_env, random_policy, max_steps=50)
    traj_b = collect_trajectory(wrapped_env, random_policy, max_steps=50)

    print(section("Trajectory A summary"))
    print(semantic_fn(traj_a))
    print(section("Trajectory B summary"))
    print(semantic_fn(traj_b))

    label, explanation = compare_trajectories(traj_a, traj_b, semantic_fn, task)
    preferred = "A" if label == 1 else "B"
    print(section("Preference"))
    print(f"LLM preferred trajectory {preferred} (label = {label})")
    print(f"Reason: {explanation}")

    wrapped_env.close()