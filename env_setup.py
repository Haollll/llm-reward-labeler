from typing import Any, Callable, List, Tuple

import gymnasium as gym

from llm_utils import cache_key, compare_trajectories, generate_reward_fn, generate_semantic_fn
from helper import load_task, section
from reward import CompositeReward


class CustomRewardWrapper(gym.RewardWrapper):
    """Gym wrapper that replaces the environment reward with a custom reward function."""

    def __init__(self, env: gym.Env, reward_fn: Callable[[Any, Any, Any], float]):
        super().__init__(env)
        self._reward_fn = reward_fn
        self._prev_obs: Any = None
        self.true_reward_history: List[float] = []
        self._episode_true_reward = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_obs = obs
        self._episode_true_reward = 0.0
        return obs, info

    def step(self, action):
        prev_obs = self._prev_obs
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        self._prev_obs = obs
        done = terminated or truncated
        self._episode_true_reward += float(env_reward)
        info["env_reward"] = float(env_reward)
        info["true_reward_history"] = self.true_reward_history
        if done:
            self.true_reward_history.append(self._episode_true_reward)
            info["true_episode_reward"] = self._episode_true_reward
        custom_reward = float(self._reward_fn(prev_obs, action, obs))
        return obs, custom_reward, terminated, truncated, info

    def reward(self, reward):  # required by gym.RewardWrapper
        return reward

    def set_alpha(self, alpha: float) -> None:
        if hasattr(self._reward_fn, "set_alpha"):
            self._reward_fn.set_alpha(alpha)


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

    composite = CompositeReward(
        r_fixed=reward_fn,
        cache_key=cache_key(env, task),
        # g=...           # defaults to constant 1
        # reward_model=...  # plug in Bradley-Terry model once trained
    )
    wrapped_env = CustomRewardWrapper(env, composite)

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
