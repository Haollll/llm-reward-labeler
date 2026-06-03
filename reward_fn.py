"""Composite reward used by the PPO training env.

Phase I:  alpha = 1.0  → reward = r_fixed only.
Phase II: 0 < alpha < 1 → reward = alpha * r_fixed + (1 - alpha) * R_phi,
          where R_phi is the (normalised) Bradley-Terry model prediction.

Unlike v1, alpha is held CONSTANT within Phase II (it does not decay to 0), so
the coded reward always anchors the learned reward and PPO cannot fully chase a
misspecified R_phi.
"""

from typing import Any, Callable

import numpy as np


class CompositeReward:
    def __init__(
        self,
        r_fixed: Callable[[Any, Any, Any], Any],
        reward_model=None,
        alpha: float = 1.0,
    ):
        self.r_fixed = r_fixed
        self.reward_model = reward_model
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.last_components = {"total": 0.0}

    def set_alpha(self, alpha: float) -> None:
        self.alpha = float(np.clip(alpha, 0.0, 1.0))

    def set_reward_model(self, reward_model) -> None:
        self.reward_model = reward_model

    def __call__(self, obs, action, next_obs) -> float:
        raw = self.r_fixed(obs, action, next_obs)
        if isinstance(raw, dict):
            self.last_components = {k: float(v) for k, v in raw.items()}
            r_fix = self.last_components.get("total", 0.0)
        else:
            r_fix = float(raw)
            self.last_components = {"total": r_fix}

        if self.reward_model is None or self.alpha >= 1.0:
            return r_fix

        r_phi = float(self.reward_model.predict_normalized(obs, action))
        return self.alpha * r_fix + (1.0 - self.alpha) * r_phi
