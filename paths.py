"""Env-first artifact layout.

Everything an env produces lives under a single directory so multi-env runs stay
organized and a whole env can be archived or deleted in one move:

    artifacts/<env_id>/
        policies/policy.zip
        reward_models/{member*.pt, metadata.pt}
        baseline/metrics.json          # RL-Zoo3 baseline returns/lengths
        reflection/{reflection_log.json, reward_round<r>.py, semantic_round<r>.py}
        metrics.json                   # per-round training metrics (for plots)
        plots/*.pdf
"""

from pathlib import Path


def env_dir(env_id: str, artifact_dir: str = "artifacts") -> Path:
    return Path(artifact_dir) / env_id


def policy_dir(env_id: str, artifact_dir: str = "artifacts") -> Path:
    return env_dir(env_id, artifact_dir) / "policies"


def reward_model_dir(env_id: str, artifact_dir: str = "artifacts") -> Path:
    return env_dir(env_id, artifact_dir) / "reward_models"


def baseline_dir(env_id: str, artifact_dir: str = "artifacts") -> Path:
    return env_dir(env_id, artifact_dir) / "baseline"


def reflection_dir(env_id: str, artifact_dir: str = "artifacts") -> Path:
    return env_dir(env_id, artifact_dir) / "reflection"


def plots_dir(env_id: str, artifact_dir: str = "artifacts") -> Path:
    return env_dir(env_id, artifact_dir) / "plots"


def metrics_path(env_id: str, artifact_dir: str = "artifacts") -> Path:
    return env_dir(env_id, artifact_dir) / "metrics.json"


def eval_metrics_path(env_id: str, artifact_dir: str = "artifacts") -> Path:
    """evaluate.py's pipeline-policy evaluation results (drives the cross-env plot)."""
    return env_dir(env_id, artifact_dir) / "eval.json"


def baseline_metrics_path(env_id: str, artifact_dir: str = "artifacts") -> Path:
    return baseline_dir(env_id, artifact_dir) / "metrics.json"
