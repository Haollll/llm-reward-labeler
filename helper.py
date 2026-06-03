"""Shared helpers: env warning suppression, task loading, env→task map,
success criteria, and pretty-printing. Self-contained for v2."""

import warnings
from pathlib import Path

import gymnasium as gym

TASK_DIR = Path(__file__).parent / "tasks"


# ─────────────────────────────────────────────────────────────
# Warning suppression
# ─────────────────────────────────────────────────────────────
def silence_env_warnings() -> None:
    """Mute the noisy MuJoCo-v4 "out of date" deprecation notice and the legacy
    `gym` import banner so training logs stay readable."""
    warnings.filterwarnings(
        "ignore",
        message=r".*(out of date|upgrading to version).*",
        category=DeprecationWarning,
    )
    try:
        gym.logger.set_level(gym.logger.ERROR)
    except Exception:
        pass
    import os
    import contextlib
    try:
        with contextlib.redirect_stderr(open(os.devnull, "w")):
            import gym as _legacy_gym  # noqa: F401
    except Exception:
        pass


# applied on import so every entry point is covered
silence_env_warnings()


# ─────────────────────────────────────────────────────────────
# Tasks / envs
# ─────────────────────────────────────────────────────────────
# Canonical set of envs the v2 pipeline + baselines target.
SUPPORTED_ENVS = [
    "Pendulum-v1",
    "Swimmer-v4",
    "HalfCheetah-v4",
    "Hopper-v4",
    "Walker2d-v4",
    "Ant-v4",
]

ENV_TASKS = {
    "Pendulum-v1":    "pendulum",
    "Swimmer-v4":     "swimmer",
    "HalfCheetah-v4": "halfcheetah",
    "HalfCheetah-v5": "halfcheetah",
    "Hopper-v4":      "hopper",
    "Walker2d-v4":    "walker2d",
    "Ant-v4":         "ant",
}


def task_for_env(env_id: str) -> str:
    """Default task-file name for an env id. Falls back to the lowercased base
    name (e.g. 'Humanoid-v4' -> 'humanoid')."""
    if env_id in ENV_TASKS:
        return ENV_TASKS[env_id]
    return env_id.split("-")[0].lower()


def load_task(name: str) -> str:
    """Load task description text from TASK_DIR."""
    return (TASK_DIR / f"{name}.txt").read_text().strip()


# ─────────────────────────────────────────────────────────────
# Success criterion
# ─────────────────────────────────────────────────────────────
# None of the supported MuJoCo locomotion / control envs report a natural binary
# success criterion, so this returns None for every env and the success-rate
# metric is omitted from the plots. Kept as a hook for goal-reaching tasks.

def success_fn_for_env(env_id: str):
    """Return f(episode_length, episode_return) -> bool, or None if N/A."""
    return None


# ─────────────────────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────────────────────

def section(title: str, width: int = 60) -> str:
    pad = max(width - len(title) - 4, 4)
    return f"\n\033[1;36m── {title} {'─' * pad}\033[0m"
