"""Load RL-Zoo3's tuned PPO hyperparameters for an env and translate them into
stable-baselines3 PPO kwargs (+ the training-env settings n_envs / normalize).

RL-Zoo3 ships per-env tuned hyperparameters in `rl_zoo3/hyperparams/ppo.yml`.
The raw entries mix three concerns we have to separate:
  * direct PPO model kwargs        (n_steps, batch_size, learning_rate, ...)
  * training-env settings          (n_envs, normalize)
  * bookkeeping we ignore here     (n_timesteps, policy)

Special encodings handled:
  * "lin_X"           → a linear schedule (SB3 expects a callable)
  * policy_kwargs str → eval'd with `dict` and `nn` (torch.nn) in scope
  * normalize         → bool or a dict-string of VecNormalize kwargs
"""

import os
from typing import Any, Dict, Optional, Tuple

import torch.nn as nn

# PPO constructor kwargs we forward verbatim (after value parsing).
_PPO_KEYS = {
    "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "ent_coef",
    "learning_rate", "clip_range", "clip_range_vf", "max_grad_norm", "vf_coef",
    "use_sde", "sde_sample_freq", "target_kl", "policy_kwargs",
}


def linear_schedule(initial_value: float):
    """SB3-style linear schedule: value decays from `initial_value` to 0 as
    progress_remaining goes 1 → 0."""
    initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def _parse_value(val: Any) -> Any:
    if isinstance(val, str) and val.startswith("lin_"):
        return linear_schedule(float(val[len("lin_"):]))
    return val


def _parse_policy_kwargs(s: Any) -> Any:
    if not isinstance(s, str):
        return s
    return eval(s, {"dict": dict, "nn": nn})  # zoo strings use dict(...) + nn.*


def _parse_normalize(val: Any):
    """Return VecNormalize kwargs dict if normalization is requested, else None."""
    if val in (None, False):
        return None
    if val is True:
        return {"norm_obs": True, "norm_reward": True}
    if isinstance(val, str):
        try:
            d = eval(val, {"dict": dict})
            if isinstance(d, dict):
                return d
        except Exception:
            pass
        return {"norm_obs": True, "norm_reward": True}
    if isinstance(val, dict):
        return val
    return {"norm_obs": True, "norm_reward": True}


def _yaml_path(algo: str = "ppo") -> str:
    import rl_zoo3
    return os.path.join(os.path.dirname(rl_zoo3.__file__), "hyperparams", f"{algo}.yml")


def zoo_n_timesteps(env_id: str, algo: str = "ppo", default: int = 1_000_000) -> int:
    """Total training timesteps RL-Zoo3 uses for `env_id`."""
    import yaml
    cfg = yaml.safe_load(open(_yaml_path(algo)))
    entry = cfg.get(env_id) or {}
    n = entry.get("n_timesteps")
    return int(float(n)) if n is not None else default


def load_zoo_ppo_hyperparams(env_id: str, algo: str = "ppo") -> Dict[str, Any]:
    """Return a dict:
        {
          "ppo_kwargs":   {...},          # forwardable to PPO(...)
          "n_envs":       int,
          "normalize":    dict | None,    # VecNormalize kwargs, or None
          "n_timesteps":  int,
          "found":        bool,           # whether the env had a tuned entry
        }
    Unknown / missing envs fall back to SB3 defaults (empty ppo_kwargs)."""
    import yaml
    cfg = yaml.safe_load(open(_yaml_path(algo)))
    entry = cfg.get(env_id)
    found = entry is not None
    entry = entry or {}

    ppo_kwargs: Dict[str, Any] = {}
    for k, v in entry.items():
        if k == "policy_kwargs":
            ppo_kwargs[k] = _parse_policy_kwargs(v)
        elif k in _PPO_KEYS:
            ppo_kwargs[k] = _parse_value(v)

    return {
        "ppo_kwargs": ppo_kwargs,
        "n_envs": int(entry.get("n_envs", 1)),
        "normalize": _parse_normalize(entry.get("normalize")),
        "n_timesteps": int(float(entry.get("n_timesteps", 1_000_000))),
        "found": found,
    }
