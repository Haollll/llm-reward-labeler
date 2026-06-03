"""Env wrapper, trajectory collection, and evaluation utilities for v2."""

from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np


class CustomRewardWrapper(gym.Wrapper):
    """Replace the environment reward with a custom (composite) reward function
    while still surfacing the true env reward via info["env_reward"]."""

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

    def set_alpha(self, alpha: float) -> None:
        if hasattr(self._reward_fn, "set_alpha"):
            self._reward_fn.set_alpha(alpha)

    def set_reward_model(self, reward_model) -> None:
        if hasattr(self._reward_fn, "set_reward_model"):
            self._reward_fn.set_reward_model(reward_model)


def collect_trajectory(
    env,
    policy_fn,
    max_steps: int = 1000,
    reward_fn: Optional[Callable[[Any, Any, Any], Any]] = None,
) -> List[Tuple]:
    """Collect a FULL episode (until done or max_steps).

    Each step is (obs, action, next_obs, r_comp, done) where r_comp is the dict
    of reward components from reward_fn (or {"total": env_reward} if none)."""
    trajectory: List[Tuple] = []
    obs, _ = env.reset()
    for _ in range(max_steps):
        action = policy_fn(obs)
        next_obs, env_reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if reward_fn is not None:
            raw = reward_fn(obs, action, next_obs)
            if isinstance(raw, dict):
                r_comp = {k: float(v) for k, v in raw.items()}
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


def relabel_reward_components(trajectories: List[List[Tuple]], reward_fn) -> List[List[Tuple]]:
    """Recompute the r_comp dict of every step under a new reward_fn (cheap, no
    env interaction). Used in Phase I after reflection rewrites r_fixed."""
    out = []
    for traj in trajectories:
        new_traj = []
        for obs, action, next_obs, _old, done in traj:
            raw = reward_fn(obs, action, next_obs)
            if isinstance(raw, dict):
                r_comp = {k: float(v) for k, v in raw.items()}
                if "total" not in r_comp:
                    r_comp["total"] = float(sum(r_comp.values()))
            else:
                r_comp = {"total": float(raw)}
            new_traj.append((obs, action, next_obs, r_comp, done))
        out.append(new_traj)
    return out


def eval_with_components(
    env,
    policy_fn,
    reward_fn: Optional[Callable[[Any, Any, Any], Any]] = None,
    n_episodes: int = 100,
    max_steps: int = 10_000,
) -> Dict[str, Any]:
    """Run n_episodes; record per-episode true env reward, length, success flag,
    and per-component trajectory sums (excluding "total")."""
    lengths: List[int] = []
    env_rewards: List[float] = []
    successes: List[bool] = []
    per_episode_components: List[Dict[str, float]] = []
    saw_success = False

    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_reward, ep_len, ep_success = 0.0, 0, False
        ep_components: Dict[str, float] = {}
        for _ in range(max_steps):
            action = policy_fn(obs)
            next_obs, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += float(env_reward)
            ep_len += 1
            if reward_fn is not None:
                raw = reward_fn(obs, action, next_obs)
                if isinstance(raw, dict):
                    for k, v in raw.items():
                        if k == "total":
                            continue
                        ep_components[k] = ep_components.get(k, 0.0) + float(
                            np.nan_to_num(float(v), nan=0.0, posinf=0.0, neginf=0.0))
            if "is_success" in info:
                saw_success = True
                ep_success = ep_success or bool(info["is_success"])
            obs = next_obs
            if done:
                break
        lengths.append(ep_len)
        env_rewards.append(ep_reward)
        successes.append(ep_success)
        per_episode_components.append(ep_components)

    names: set = set()
    for d in per_episode_components:
        names.update(d.keys())
    component_sums = {
        name: [float(d.get(name, 0.0)) for d in per_episode_components]
        for name in sorted(names)
    }
    return {
        "episode_lengths": lengths,
        "episode_env_rewards": env_rewards,
        "success": successes if saw_success else None,
        "component_sums": component_sums,
    }
