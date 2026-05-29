from typing import Callable

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env_setup import CustomRewardWrapper
from reward import CompositeReward
from reward_model import RewardModel


class PPOAgent:
    """
    Thin wrapper around stable-baselines3 PPO.

    The training environment uses CustomRewardWrapper so PPO optimizes
    alpha * r_fixed + (1 - alpha) * g * R_phi instead of the gym reward.
    Evaluation runs on a raw gym env so it reports true environment reward.
    """

    def __init__(
        self,
        env_id: str,
        reward_fn: Callable,
        reward_model: RewardModel,
        lr: float = 3e-4,
        n_steps: int = 2_048,
        batch_size: int = 64,
        n_epochs: int = 10,
        cache_key: str = "",
        progress_bar: bool = True,
        verbose: int = 0,
    ):
        self._reward_fn = reward_fn
        self._reward_model = reward_model
        self._env_id = env_id
        self._cache_key = cache_key or env_id
        self._progress_bar = progress_bar

        self._train_env = DummyVecEnv([self._make_train_env])
        self._model = PPO(
            "MlpPolicy",
            self._train_env,
            learning_rate=lr,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            device="cpu",
            verbose=verbose,
        )

    def train(self, total_timesteps: int) -> None:
        self._model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=False,
            progress_bar=self._progress_bar,
        )

    def predict(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self._model.predict(obs, deterministic=False)
        return action

    def predict_deterministic(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self._model.predict(obs, deterministic=True)
        return action

    def evaluate(self, n_episodes: int = 5) -> float:
        rewards = []
        eval_env = gym.make(self._env_id)
        try:
            for _ in range(n_episodes):
                obs, _ = eval_env.reset()
                total, done = 0.0, False
                while not done:
                    action, _ = self._model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = eval_env.step(action)
                    total += float(reward)
                    done = terminated or truncated
                rewards.append(total)
        finally:
            eval_env.close()
        return float(np.mean(rewards))

    @property
    def true_reward_history(self) -> list[float]:
        histories = self._train_env.get_attr("true_reward_history")
        return list(histories[0]) if histories else []

    def set_alpha(self, alpha: float) -> None:
        self._train_env.env_method("set_alpha", alpha)

    def save(self, path: str) -> None:
        self._model.save(path)

    def load(self, path: str) -> None:
        self._model = PPO.load(path, env=self._train_env)

    def _make_train_env(self) -> CustomRewardWrapper:
        env = gym.make(self._env_id)
        composite = CompositeReward(
            r_fixed=self._reward_fn,
            cache_key=self._cache_key,
            reward_model=self._reward_model,
        )
        return CustomRewardWrapper(env, composite)
