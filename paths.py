"""Env-first artifact layout for v2.

Everything an env produces lives under a single directory:

    artifacts/<env_id>/
        policies/policy.zip            # trained PPO policy
        reward_model/{model.pt, meta.pt}
        baseline/metrics.json          # RL-Zoo3 baseline returns/lengths (cached)
        reflection/{reflection_log.json, reward_round<r>.py, semantic_round<r>.py}
        metrics.json                   # per-round training metrics (drives plots)
        eval.json                      # trained-policy evaluation (100 ep)
        bt_vs_env.json                 # per-step BT reward vs true env reward (1 ep)
        plots/*.pdf
"""

from pathlib import Path


def env_dir(env_id: str, artifact_dir: str = "artifacts") -> Path:
    return Path(artifact_dir) / env_id


def policy_dir(env_id: str, artifact_dir: str = "artifacts") -> Path:
    return env_dir(env_id, artifact_dir) / "policies"


def reward_model_dir(env_id: str, artifact_dir: str = "artifacts") -> Path:
    return env_dir(env_id, artifact_dir) / "reward_model"


def baseline_dir(env_id: str, artifact_dir: str = "artifacts") -> Path:
    return env_dir(env_id, artifact_dir) / "baseline"


def reflection_dir(env_id: str, artifact_dir: str = "artifacts") -> Path:
    return env_dir(env_id, artifact_dir) / "reflection"


def plots_dir(env_id: str, artifact_dir: str = "artifacts") -> Path:
    return env_dir(env_id, artifact_dir) / "plots"


def metrics_path(env_id: str, artifact_dir: str = "artifacts") -> Path:
    return env_dir(env_id, artifact_dir) / "metrics.json"


def eval_metrics_path(env_id: str, artifact_dir: str = "artifacts") -> Path:
    return env_dir(env_id, artifact_dir) / "eval.json"


def bt_vs_env_path(env_id: str, artifact_dir: str = "artifacts") -> Path:
    return env_dir(env_id, artifact_dir) / "bt_vs_env.json"


def baseline_metrics_path(env_id: str, artifact_dir: str = "artifacts") -> Path:
    return baseline_dir(env_id, artifact_dir) / "metrics.json"
