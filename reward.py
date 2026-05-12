from collections import deque
from typing import Any, Callable, Optional

import numpy as np


class CompositeReward:
    """Composes alpha * r_fixed + (1 - alpha) * g * normalize(R_phi).

    alpha starts at 1.0 so PPO trains only on r_fixed. As alpha decays,
    the learned reward model contributes more of the training signal.

    Identified by `cache_key` (env_id + task hash) so different (env, task) pairs
    can be tracked and saved separately.
    """

    def __init__(
        self,
        r_fixed: Callable[[Any, Any, Any], float],
        cache_key: str,
        g: Optional[Callable[[Any, Any, Any], float]] = None,
        reward_model=None,
        alpha: float = 1.0,
    ):
        self.r_fixed = r_fixed
        self.cache_key = cache_key
        self.g = g if g is not None else (lambda *_: 1.0)
        self.reward_model = reward_model
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self._recent_abs: deque = deque(maxlen=200)

    def set_alpha(self, alpha: float) -> None:
        self.alpha = float(np.clip(alpha, 0.0, 1.0))

    def __call__(self, obs, action, next_obs) -> float:
        r_fix = float(self.r_fixed(obs, action, next_obs))

        if self.reward_model is None:
            return r_fix

        r_phi = float(self.reward_model.predict(obs, action))
        self._recent_abs.append(abs(r_phi))
        std = float(np.std(self._recent_abs)) if len(self._recent_abs) > 10 else 1.0
        r_phi_norm = r_phi / (std + 1e-8)

        gate = float(self.g(obs, action, next_obs))
        return self.alpha * r_fix + (1.0 - self.alpha) * gate * r_phi_norm
