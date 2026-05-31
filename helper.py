import warnings
from pathlib import Path

import gymnasium as gym

TASK_DIR = Path(__file__).parent / "tasks"


# ─────────────────────────────────────────────────────────────
# Warning suppression
# ─────────────────────────────────────────────────────────────
# The MuJoCo v4 envs we train on are flagged "out of date" by gymnasium, which
# emits a DeprecationWarning at every gym.make. We deliberately stay on v4 (the
# most up-to-date version RL-Zoo3 ships tuned hyperparameters for), so silence
# that specific warning.

def silence_env_warnings() -> None:
    # MuJoCo v4 "out of date" deprecation notice.
    warnings.filterwarnings(
        "ignore",
        message=r".*(out of date|upgrading to version).*",
        category=DeprecationWarning,
    )
    # gymnasium routes the deprecation notice through its own logger too.
    try:
        gym.logger.set_level(gym.logger.ERROR)
    except Exception:
        pass
    # The legacy `gym` package (pulled in transitively by stable-baselines3)
    # prints a one-time "Gym has been unmaintained" notice to stderr at import —
    # a raw print, not a warning, so filterwarnings can't catch it. Pre-import it
    # once with stderr muted so the notice is swallowed before SB3 triggers it.
    import os
    import sys
    import contextlib
    try:
        with contextlib.redirect_stderr(open(os.devnull, "w")):
            import gym  # noqa: F401  (legacy gym; imported only to mute its notice)
    except Exception:
        pass


# applied on import so every entry point (train, evaluate, plots, …) is covered
silence_env_warnings()


# ─────────────────────────────────────────────────────────────
# Tasks
# ─────────────────────────────────────────────────────────────
# Default task file for each supported environment. The task file carries the
# human-curated obs-layout emphasis fed to the LLM alongside the env docstring.
# Canonical set of envs the pipeline + baselines target (most up-to-date version
# supported by RL-Zoo3 for each).
SUPPORTED_ENVS = [
    "Pendulum-v1",
    "Swimmer-v4",
    "HalfCheetah-v4",
    "Hopper-v4",
    "Walker2d-v4",
    "Ant-v4",
]

ENV_TASKS = {
    "Pendulum-v1":         "pendulum",
    "Swimmer-v4":          "swimmer",
    "HalfCheetah-v4":      "halfcheetah",
    "HalfCheetah-v5":      "halfcheetah",
    "Hopper-v4":           "hopper",
    "Walker2d-v4":         "walker2d",
    "Ant-v4":              "ant",
}


def task_for_env(env_id: str) -> str:
    """Default task-file name for an env id. Falls back to the lowercased base
    name (e.g. 'Humanoid-v4' -> 'humanoid')."""
    if env_id in ENV_TASKS:
        return ENV_TASKS[env_id]
    return env_id.split("-")[0].lower()


def load_task(name: str) -> str:
    """Load task description from TASK_DIR."""
    return (TASK_DIR / f"{name}.txt").read_text().strip()


# ─────────────────────────────────────────────────────────────
# Success criterion (for the "success rate" metric, where applicable)
# ─────────────────────────────────────────────────────────────
# None of the supported envs report info["is_success"], and none has a natural
# binary success criterion (Pendulum/Swimmer/locomotion are all continuous-return
# tasks with a fixed or health-gated horizon), so this returns None for every env
# and the success-rate metric is omitted from the plots. Kept as a hook for envs
# that do define success (e.g. goal-reaching tasks) added later.

def success_fn_for_env(env_id: str):
    """Return f(episode_length, episode_return) -> bool, or None if N/A."""
    return None


# ─────────────────────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────────────────────

def section(title: str, width: int = 60) -> str:
    """For pretty printing."""
    pad = max(width - len(title) - 4, 4)
    return f"\n\033[1;36m── {title} {'─' * pad}\033[0m"
