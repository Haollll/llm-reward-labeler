"""Plotting suite for the LLM-reward pipeline. Every function reads the JSON
artifacts written during training (artifacts/<env>/metrics.json,
reflection/reflection_log.json, baseline/metrics.json) and saves a single,
cleanly-labelled figure as a PDF under artifacts/<env>/plots/.

Driven by make_plots.py; importable directly too.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import metrics_path, baseline_metrics_path, eval_metrics_path, plots_dir

plt.rcParams.update({
    "figure.autolayout": True,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


# ── loaders ───────────────────────────────────────────────────────────────────

def load_metrics(env_id: str, artifact_dir: str = "artifacts") -> dict:
    p = metrics_path(env_id, artifact_dir)
    if not p.exists():
        raise FileNotFoundError(f"No metrics at {p} (train {env_id} first)")
    return json.loads(p.read_text())


def load_baseline(env_id: str, artifact_dir: str = "artifacts") -> Optional[dict]:
    p = baseline_metrics_path(env_id, artifact_dir)
    return json.loads(p.read_text()) if p.exists() else None


def _save(fig_path: Path) -> Path:
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path)
    plt.close()
    return fig_path


# ── per-env plots ───────────────────────────────────────────────────────────

def plot_episode_return_per_round(env_id: str, artifact_dir: str = "artifacts") -> Path:
    """Eval episodic return (true env reward) vs training round."""
    m = load_metrics(env_id, artifact_dir)
    ev = m["eval"]
    rounds = [e["round"] for e in ev]
    rets = [e["episode_env_reward"] for e in ev]
    plt.figure(figsize=(7, 4.5))
    plt.plot(rounds, rets, marker="o", color="#4C72B0")
    plt.xlabel("Training round")
    plt.ylabel("Mean episodic return (true env reward)")
    plt.title(f"{env_id}: episodic return per round")
    return _save(plots_dir(env_id, artifact_dir) / "episode_return_per_round.pdf")


def plot_reward_model_loss(env_id: str, artifact_dir: str = "artifacts") -> Path:
    """Reward-model CE / smooth / total loss across training steps."""
    m = load_metrics(env_id, artifact_dir)
    rm = m["reward_model"]
    x = list(range(len(rm)))
    labels = [r["round"] for r in rm]
    plt.figure(figsize=(7, 4.5))
    plt.plot(x, [r["ce"] for r in rm], marker="o", label="CE loss")
    plt.plot(x, [r["smooth"] for r in rm], marker="s", label="smooth (unscaled)")
    plt.plot(x, [r["total"] for r in rm], marker="^", label="total")
    plt.xticks(x, labels)
    plt.xlabel("Reward-model training step (round)")
    plt.ylabel("Loss")
    plt.title(f"{env_id}: reward-model loss")
    plt.legend()
    return _save(plots_dir(env_id, artifact_dir) / "reward_model_loss.pdf")


def plot_component_contributions(env_id: str, artifact_dir: str = "artifacts") -> Path:
    """Stacked per-component reward sums across rounds (how the reward is assembled)."""
    m = load_metrics(env_id, artifact_dir)
    ev = m["eval"]
    rounds = [e["round"] for e in ev]
    names = sorted({k for e in ev for k in e.get("component_means", {})})
    plt.figure(figsize=(7.5, 4.5))
    for name in names:
        series = [e.get("component_means", {}).get(name, 0.0) for e in ev]
        plt.plot(rounds, series, marker="o", label=name)
    plt.xlabel("Training round")
    plt.ylabel("Mean per-trajectory component sum")
    plt.title(f"{env_id}: reward components across rounds")
    if names:
        plt.legend(fontsize=8)
    return _save(plots_dir(env_id, artifact_dir) / "reward_components.pdf")


PER_ENV_PLOTS = [
    plot_episode_return_per_round,
    plot_reward_model_loss,
    plot_component_contributions,
]


def make_all_per_env(env_id: str, artifact_dir: str = "artifacts") -> List[Path]:
    out = []
    for fn in PER_ENV_PLOTS:
        try:
            out.append(fn(env_id, artifact_dir))
        except Exception as e:
            print(f"  [skip {fn.__name__}] {type(e).__name__}: {e}")
    return out


# ── cross-env plots ───────────────────────────────────────────────────────────

def _pipeline_summary(env_id: str, artifact_dir: str):
    """Pipeline return/length/success from evaluate.py's eval.json (the dedicated
    evaluation of the trained policy). Returns None if it hasn't been run yet."""
    p = eval_metrics_path(env_id, artifact_dir)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return {
        "return":  d["mean_return"],
        "length":  d["mean_length"],
        "success": d.get("success_rate"),
    }


def plot_cross_env_bars(
    env_ids: List[str],
    artifact_dir: str = "artifacts",
    out_dir: str = "artifacts/cross_env",
) -> List[Path]:
    """Single figure with the three cross-env metrics as stacked panels (sharing
    the env axis): mean episodic return, mean episode length, and success rate
    (panel shown only if at least one env reports it). Pipeline vs RL-Zoo3 PPO
    baseline. Returns a one-element list with the saved PDF path.

    The metrics live on very different scales (return in thousands, length in
    hundreds, success in [0,1]), so they get their own y-axis panels rather than
    being forced onto one axis."""
    pipe = {e: _pipeline_summary(e, artifact_dir) for e in env_ids}
    base = {e: load_baseline(e, artifact_dir) for e in env_ids}
    envs = [e for e in env_ids if pipe.get(e) is not None]
    if not envs:
        raise RuntimeError(
            "No eval.json found for any requested env. Run evaluate.py per env "
            "first (e.g. `python evaluate.py --env HalfCheetah-v4`) so the cross-env "
            "plot has pipeline eval data."
        )

    # panels to draw: (metric key, baseline accessor, y-label, bar fmt, ylim)
    panels = [
        ("return", lambda b: b["mean_return"],
         "Episodic return", "%.0f", None),
        ("length", lambda b: b["mean_length"],
         "Episode length", "%.0f", None),
    ]
    if any(pipe[e]["success"] is not None for e in envs):
        panels.append(
            ("success", lambda b: float(np.mean(b["success"])) if b.get("success") else np.nan,
             "Success rate", "%.2f", (0, 1.05))
        )

    x = np.arange(len(envs))
    w = 0.38
    fig, axes = plt.subplots(
        len(panels), 1, sharex=True,
        figsize=(max(7, 1.5 * len(envs)), 2.6 * len(panels) + 0.6),
    )
    if len(panels) == 1:
        axes = [axes]

    for ax, (metric, base_get, ylabel, fmt, ylim) in zip(axes, panels):
        pvals = [pipe[e][metric] if pipe[e][metric] is not None else np.nan for e in envs]
        bvals = [base_get(base[e]) if base.get(e) else np.nan for e in envs]
        b1 = ax.bar(x - w / 2, pvals, w, label="Pipeline (LLM reward)", color="#4C72B0")
        b2 = ax.bar(x + w / 2, bvals, w, label="RL-Zoo3 PPO baseline", color="#DD8452")
        ax.bar_label(b1, fmt=fmt, padding=2, fontsize=8)
        ax.bar_label(b2, fmt=fmt, padding=2, fontsize=8)
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(envs, rotation=20, ha="right")
    axes[0].set_title("Pipeline vs RL-Zoo3 PPO baseline across environments")
    # one shared legend at the top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, fontsize=9,
               bbox_to_anchor=(1.0, 1.0))

    out = Path(out_dir) / "cross_env_comparison.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close(fig)
    out_paths = [out]

    return out_paths
