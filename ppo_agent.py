"""stable-baselines3 PPO wrapper for v2.

By default the agent loads RL-Zoo3's tuned PPO hyperparameters for the env
(learning rate, n_steps, batch_size, n_epochs, gamma, gae_lambda, clip_range,
ent_coef, vf_coef, max_grad_norm, policy_kwargs, use_sde, ...), the number of
parallel training envs (`n_envs`), and the VecNormalize setting.

The training env stack is:  DummyVecEnv([CustomRewardWrapper(gym.make)]*n_envs)
optionally wrapped in VecNormalize. CustomRewardWrapper sits *under* VecNormalize,
so the composite reward function always sees RAW observations (correct for
r_fixed and the BT model), while the policy sees normalized observations.

Because the policy is trained on normalized observations, `predict` /
`predict_deterministic` apply the same obs normalization (using the live
VecNormalize stats) so trajectory collection and evaluation stay consistent. The
VecNormalize stats are saved next to the policy so evaluate.py can reload them.
"""

from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env_utils import CustomRewardWrapper
from reward_fn import CompositeReward
from zoo_hyperparams import load_zoo_ppo_hyperparams


class PPOAgent:
    def __init__(
        self,
        env_id: str,
        reward_fn: Callable,
        reward_model=None,
        use_zoo_hyperparams: bool = True,
        progress_bar: bool = True,
        verbose: int = 0,
    ):
        self._env_id = env_id
        self._progress_bar = progress_bar
        self._composite = CompositeReward(reward_fn, reward_model, alpha=1.0)

        zoo = load_zoo_ppo_hyperparams(env_id) if use_zoo_hyperparams else {
            "ppo_kwargs": {}, "n_envs": 1, "normalize": None, "found": False}
        self.zoo = zoo
        self._n_envs = max(1, int(zoo["n_envs"]))
        self._norm_kwargs = zoo["normalize"]

        venv = DummyVecEnv([self._make_train_env for _ in range(self._n_envs)])
        if self._norm_kwargs is not None:
            kw = dict(self._norm_kwargs)
            kw.setdefault("gamma", zoo["ppo_kwargs"].get("gamma", 0.99))
            venv = VecNormalize(venv, **kw)
        self._train_env = venv
        self._vecnorm: Optional[VecNormalize] = venv if isinstance(venv, VecNormalize) else None

        ppo_kwargs = dict(zoo["ppo_kwargs"])
        ppo_kwargs.setdefault("gamma", 0.99)
        ppo_kwargs.setdefault("gae_lambda", 0.95)
        self._model = PPO(
            "MlpPolicy", self._train_env,
            device="cpu", verbose=verbose, **ppo_kwargs,
        )

    def _make_train_env(self) -> CustomRewardWrapper:
        return CustomRewardWrapper(gym.make(self._env_id), self._composite)

    # ── training ─────────────────────────────────────────────
    def train(self, total_timesteps: int) -> None:
        self._model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=False,
            progress_bar=self._progress_bar,
        )

    def set_alpha(self, alpha: float) -> None:
        self._composite.set_alpha(alpha)

    def attach_reward_model(self, reward_model) -> None:
        self._composite.set_reward_model(reward_model)

    # ── inference (obs normalized to match training) ─────────
    def _norm_obs(self, obs: np.ndarray) -> np.ndarray:
        if self._vecnorm is not None:
            return self._vecnorm.normalize_obs(obs)
        return obs

    def predict(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self._model.predict(self._norm_obs(obs), deterministic=False)
        return action

    def predict_deterministic(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self._model.predict(self._norm_obs(obs), deterministic=True)
        return action

    # ── persistence ──────────────────────────────────────────
    def save(self, path: str) -> None:
        """Save policy.zip and, if used, the VecNormalize stats next to it."""
        path = Path(path)
        self._model.save(str(path))
        if self._vecnorm is not None:
            self._vecnorm.save(str(path.parent / "vecnormalize.pkl"))

    def load(self, path: str) -> None:
        self._model = PPO.load(path, env=self._train_env, device="cpu")
