from collections import deque
from typing import Any, Callable, List, Tuple
 
import numpy as np
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
    """Gym wrapper that replaces the environment reward with:
       r_total = r_fixed(prev_obs, action, obs) + g * normalise(R_phi(prev_obs, action))
    """
    def __init__(
        self,
        env: gym.Env,
        reward_fn: Callable[[Any, Any, Any], float],
        reward_model=None,   # EnsembleRewardModel, optional
    ):
        super().__init__(env)
        self._reward_fn    = reward_fn
        self._reward_model = reward_model
        self._prev_obs: Any = None
        self._recent_r      = deque(maxlen=200)
 
    def reset(self, **kwargs):
        obs, info      = self.env.reset(**kwargs)
        self._prev_obs = obs
        return obs, info
 
    def step(self, action):
        prev_obs = self._prev_obs
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        self._prev_obs = obs
        info["env_reward"] = float(env_reward)
 
        # r_fixed: LLM-generated rule-based reward
        r_fixed = float(self._reward_fn(prev_obs, action, obs))
 
        # r_learned: Bradley-Terry reward model prediction (if available)
        r_learned = 0.0
        if self._reward_model is not None:
            r_learned = self._reward_model.predict(prev_obs, action)
            self._recent_r.append(abs(r_learned))
            std       = float(np.std(self._recent_r)) if len(self._recent_r) > 10 else 1.0
            r_learned = r_learned / (std + 1e-8)
 
        # soft gate
        g = 1.0 if np.all(np.abs(action) <= 1.0 + 1e-4) else 0.0
 
        custom_reward = r_fixed + g * r_learned
        return obs, custom_reward, terminated, truncated, info
 
    def reward(self, reward):   # required by gym.RewardWrapper
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