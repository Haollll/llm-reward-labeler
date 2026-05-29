from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

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


def eval_with_components(
    env,
    policy_fn,
    reward_fn: Optional[Callable[[Any, Any, Any], Any]] = None,
    n_episodes: int = 5,
    max_steps: int = 10_000,
) -> Dict[str, Any]:
    """Run n_episodes from reset, collect per-episode metrics.

    For each episode we record: total environment reward, length, success flag
    (from info["is_success"] if the env reports it), and the trajectory sum of
    each reward component produced by reward_fn (if provided).

    Returns:
        {
          "episode_lengths":      [int, ...]   length per episode,
          "episode_env_rewards":  [float, ...] sum of env reward per episode,
          "success":              [bool, ...] or None if env never reports it,
          "component_sums":       {name: [sum_per_episode, ...]}  ← keys exclude "total"
        }
    """
    lengths: List[int] = []
    env_rewards: List[float] = []
    successes: List[bool] = []
    per_episode_components: List[Dict[str, float]] = []
    saw_success_field = False

    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_env_reward = 0.0
        ep_length = 0
        ep_components: Dict[str, float] = {}
        ep_success = False

        for _ in range(max_steps):
            action = policy_fn(obs)
            next_obs, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_env_reward += float(env_reward)
            ep_length += 1

            if reward_fn is not None:
                raw = reward_fn(obs, action, next_obs)
                if isinstance(raw, dict):
                    for k, v in raw.items():
                        if k == "total":
                            continue
                        # sanitize: a single NaN/Inf step would poison the
                        # whole snapshot and then the reflection history
                        v_clean = float(np.nan_to_num(
                            float(v), nan=0.0, posinf=0.0, neginf=0.0,
                        ))
                        ep_components[k] = ep_components.get(k, 0.0) + v_clean

            if "is_success" in info:
                saw_success_field = True
                if info["is_success"]:
                    ep_success = True

            obs = next_obs
            if done:
                break

        lengths.append(ep_length)
        env_rewards.append(ep_env_reward)
        successes.append(ep_success)
        per_episode_components.append(ep_components)

    # transpose per-episode component sums → {name: [s_ep0, s_ep1, ...]}
    all_names: set = set()
    for d in per_episode_components:
        all_names.update(d.keys())
    component_sums = {
        name: [float(d.get(name, 0.0)) for d in per_episode_components]
        for name in sorted(all_names)
    }

    return {
        "episode_lengths":     lengths,
        "episode_env_rewards": env_rewards,
        "success":             successes if saw_success_field else None,
        "component_sums":      component_sums,
    }


def collect_trajectory(
    env,
    policy_fn,
    max_steps: int = 100,
    reward_fn: Optional[Callable[[Any, Any, Any], Any]] = None,
) -> List[Tuple]:
    """Collect a trajectory. The 4th slot of each tuple is r_comp: a dict of
    reward components (e.g. {"total": ..., "velocity": ..., "energy": ...}).
    If reward_fn is None, falls back to {"total": float(env_reward)} so the
    tuple shape is consistent regardless of whether the env is wrapped."""
    trajectory: List[Tuple] = []
    obs, _ = env.reset()
    for _ in range(max_steps):
        action = policy_fn(obs)
        next_obs, env_reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if reward_fn is not None:
            raw = reward_fn(obs, action, next_obs)
            if isinstance(raw, dict):
                r_comp: Dict[str, float] = {k: float(v) for k, v in raw.items()}
                if "total" not in r_comp:
                    r_comp["total"] = float(sum(r_comp.values()))
            else:
                r_comp = {"total": float(raw)}
        else:
            r_comp = {"total": float(env_reward)}

        trajectory.append((obs, action, next_obs, r_comp, done))
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
    traj_a = collect_trajectory(wrapped_env, random_policy, max_steps=50, reward_fn=reward_fn)
    traj_b = collect_trajectory(wrapped_env, random_policy, max_steps=50, reward_fn=reward_fn)

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
