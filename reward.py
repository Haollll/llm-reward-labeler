from collections import deque
from typing import Any, Callable

import numpy as np


class CompositeReward:

    def __init__(
        self,
        r_fixed: Callable[[Any, Any, Any], float],
        cache_key: str,
        reward_model=None,
        alpha: float = 1.0,
    ):
        self.r_fixed = r_fixed
        self.cache_key = cache_key
        self.reward_model = reward_model
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self._recent_abs: deque = deque(maxlen=200)

    def set_alpha(self, alpha: float) -> None:
        self.alpha = float(np.clip(alpha, 0.0, 1.0))

    def __call__(self, obs, action, next_obs) -> float:
        raw = self.r_fixed(obs, action, next_obs)

        # handle dict (EUREKA-style) or plain float
        if isinstance(raw, dict):
            self.last_components = {k: float(v) for k, v in raw.items()}
            r_fix = self.last_components.get("total", 0.0)
        else:
            r_fix = float(raw)
            self.last_components = {"total": r_fix}
 
        if self.reward_model is None:
            return r_fix
 
        r_phi = float(self.reward_model.predict(obs, action))
        self._recent_abs.append(abs(r_phi))
        std        = float(np.std(self._recent_abs)) if len(self._recent_abs) > 10 else 1.0
        r_phi_norm = r_phi / (std + 1e-8)

        return self.alpha * r_fix + (1.0 - self.alpha) * r_phi_norm
