import numpy as np
import gymnasium as gym
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from env_setup import CustomRewardWrapper
from reward_model import RewardModel


# ─────────────────────────────────────────────────────────────
# Callback: record true reward during training
# ─────────────────────────────────────────────────────────────

class _EvalCallback(BaseCallback):
    """
    Periodically evaluates the policy on a raw (unwrapped) environment
    and stores the true-reward history.
    """

    def __init__(self, eval_env: gym.Env, eval_freq: int = 5_000):
        super().__init__(verbose=0)
        self.eval_env  = eval_env
        self.eval_freq = eval_freq
        self.history: list[float] = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self.history.append(self._eval())
        return True

    def _eval(self, n_episodes: int = 5) -> float:
        rewards = []
        for _ in range(n_episodes):
            obs, _      = self.eval_env.reset()
            total, done = 0.0, False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, terminated, truncated, _ = self.eval_env.step(action)
                total += r
                done   = terminated or truncated
            rewards.append(total)
        return float(np.mean(rewards))


# ─────────────────────────────────────────────────────────────
# PPO Agent
# ─────────────────────────────────────────────────────────────

class PPOAgent:
    """
    Thin wrapper around stable-baselines3 PPO.

    The training environment uses CustomRewardWrapper so PPO optimises
    r_fixed + g * R_phi instead of the gym ground-truth reward.
    The evaluation environment is always the raw gym env so that
    evaluate() returns comparable true-reward numbers.
    """

    def __init__(
        self,
        env_id: str,
        reward_fn: Callable,
        reward_model: RewardModel,
        lr: float       = 3e-4,
        n_steps: int    = 2_048,
        batch_size: int = 64,
        n_epochs: int   = 10,
        eval_freq: int  = 5_000,
        verbose: int    = 0,
    ):
        self._reward_fn    = reward_fn
        self._reward_model = reward_model
        self._env_id       = env_id

        # Training env: reward replaced by custom reward
        self._train_env = DummyVecEnv([self._make_train_env])

        # Eval env: raw gym env — measures true performance
        self._eval_env  = gym.make(env_id)
        self._callback  = _EvalCallback(self._eval_env, eval_freq=eval_freq)

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
            verbose=verbose,
        )

    # ── public ───────────────────────────────────────────────

    def train(self, total_timesteps: int) -> None:
        """Run PPO for total_timesteps steps."""
        self._model.learn(
            total_timesteps=total_timesteps,
            callback=self._callback,
            reset_num_timesteps=False,
        )

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Stochastic action — used during trajectory collection."""
        action, _ = self._model.predict(obs, deterministic=False)
        return action

    def evaluate(self, n_episodes: int = 5) -> float:
        """Evaluate on the raw env. Returns mean true reward."""
        return self._callback._eval(n_episodes)

    @property
    def true_reward_history(self) -> list[float]:
        return self._callback.history

    def save(self, path: str) -> None:
        self._model.save(path)

    def load(self, path: str) -> None:
        self._model = PPO.load(path, env=self._train_env)

    # ── private ──────────────────────────────────────────────

    def _make_train_env(self) -> CustomRewardWrapper:
        env = gym.make(self._env_id)
        return CustomRewardWrapper(env, self._reward_fn, self._reward_model)