"""Plotting suite for v2. Each function reads the JSON artifacts written during
training / evaluation and saves one PDF under artifacts/<env>/plots/.

Plots:
  1. training_phases.pdf      — avg 100-ep return per round, Phase I & II on one
                                axis with a vertical line at the phase boundary.
  2. bt_loss_per_epoch.pdf    — BT reward-model loss at every epoch of every round
                                across both phases (one continuous time series).
  3. bt_vs_env.pdf            — per-step BT reward vs true env reward for a single
                                baseline-policy episode (twin y-axes).
  4. eval_metrics_bars.pdf    — 100-ep mean return / length / success-rate bars,
                                pipeline vs RL-Zoo3 baseline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import (
    metrics_path, baseline_metrics_path, eval_metrics_path,
    bt_vs_env_path, plots_dir,
)

plt.rcParams.update({
    "figure.autolayout": True,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

PIPE_C = "#4C72B0"
BASE_C = "#DD8452"
BT_C = "#55A868"


def _load(path: Path) -> Optional[dict]:
    return json.loads(path.read_text()) if path.exists() else None


def _save(fig_path: Path) -> Path:
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path)
    plt.close()
    return fig_path


# ── 1. training returns across phases ─────────────────────────────────────────

def plot_training_phases(env_id: str, artifact_dir: str = "artifacts") -> Path:
    m = _load(metrics_path(env_id, artifact_dir))
    if m is None:
        raise FileNotFoundError(f"No metrics for {env_id} (train it first)")
    rounds = m["rounds"]
    x = [r["global_round"] for r in rounds]
    y = [r["episode_env_reward"] for r in rounds]
    err = [r.get("episode_env_reward_std", 0.0) for r in rounds]
    k1 = m["k1"]

    plt.figure(figsize=(8, 4.8))
    plt.errorbar(x, y, yerr=err, marker="o", capsize=3, color=PIPE_C,
                 label="Mean 100-ep return")
    # vertical line between the last Phase-I round and the first Phase-II round
    boundary = k1 + 0.5
    plt.axvline(boundary, color="black", linestyle="--", linewidth=1.2)
    ymin, ymax = plt.ylim()
    plt.text(boundary - 0.1, ymax, "Phase I", ha="right", va="top", fontsize=10)
    plt.text(boundary + 0.1, ymax, "Phase II", ha="left", va="top", fontsize=10)
    plt.xlabel("Training round (global)")
    plt.ylabel("Mean episodic return (true env reward, 100 ep)")
    plt.title(f"{env_id}: episodic return per round (Phase I → II)")
    plt.legend(loc="lower right")
    return _save(plots_dir(env_id, artifact_dir) / "training_phases.pdf")


# ── 2. BT loss at every epoch of every round ──────────────────────────────────

def plot_bt_loss_per_epoch(env_id: str, artifact_dir: str = "artifacts") -> Path:
    m = _load(metrics_path(env_id, artifact_dir))
    if m is None:
        raise FileNotFoundError(f"No metrics for {env_id}")
    rounds = m["rounds"]
    k1 = m["k1"]

    losses: List[float] = []
    round_boundaries: List[int] = []   # x index where each round starts
    phase2_start_x = None
    for i, r in enumerate(rounds):
        round_boundaries.append(len(losses))
        if r["phase"] == "II" and phase2_start_x is None:
            phase2_start_x = len(losses)
        losses.extend(r.get("bt_loss_epochs", []))

    x = list(range(len(losses)))
    plt.figure(figsize=(9, 4.8))
    plt.plot(x, losses, color=BT_C, linewidth=1.0, label="BT cross-entropy loss")
    # light vertical guides at each round boundary
    for i, b in enumerate(round_boundaries):
        plt.axvline(b, color="grey", linestyle=":", linewidth=0.6, alpha=0.5)
    if phase2_start_x is not None:
        plt.axvline(phase2_start_x, color="black", linestyle="--", linewidth=1.2,
                    label="Phase I → II")
    plt.xlabel("BT training epoch (concatenated across rounds, both phases)")
    plt.ylabel("Cross-entropy loss")
    plt.title(f"{env_id}: BT reward-model loss per epoch "
              f"({len(rounds)} rounds × {m.get('reward_epochs', '?')} epochs)")
    plt.legend(loc="upper right")
    return _save(plots_dir(env_id, artifact_dir) / "bt_loss_per_epoch.pdf")


# ── 3. BT reward vs true env reward per timestep (single episode) ─────────────

def plot_bt_vs_env(env_id: str, artifact_dir: str = "artifacts") -> Path:
    d = _load(bt_vs_env_path(env_id, artifact_dir))
    if d is None:
        raise FileNotFoundError(f"No bt_vs_env data for {env_id} (run evaluate.py)")
    env_r = d["env_reward"]
    bt_r = d["bt_reward"]
    t = list(range(len(env_r)))

    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax2 = ax1.twinx()
    l1, = ax1.plot(t, env_r, color=BASE_C, linewidth=1.0, label="True env reward")
    l2, = ax2.plot(t, bt_r, color=BT_C, linewidth=1.0, label="BT model reward")
    ax1.set_xlabel("Timestep (single baseline-policy episode)")
    ax1.set_ylabel("True env reward / step", color=BASE_C)
    ax2.set_ylabel("BT model reward / step", color=BT_C)
    ax1.tick_params(axis="y", labelcolor=BASE_C)
    ax2.tick_params(axis="y", labelcolor=BT_C)
    ax2.grid(False)
    # Pearson correlation as a quick alignment summary
    if len(env_r) > 1 and np.std(env_r) > 0 and np.std(bt_r) > 0:
        corr = float(np.corrcoef(env_r, bt_r)[0, 1])
        title = f"{env_id}: BT reward vs true env reward per step (Pearson r={corr:.2f})"
    else:
        title = f"{env_id}: BT reward vs true env reward per step"
    ax1.set_title(title)
    ax1.legend(handles=[l1, l2], loc="upper right")
    return _save(plots_dir(env_id, artifact_dir) / "bt_vs_env.pdf")


# ── 4. eval metric bars (pipeline vs baseline) ────────────────────────────────

def plot_eval_metrics_bars(env_id: str, artifact_dir: str = "artifacts") -> Path:
    pipe = _load(eval_metrics_path(env_id, artifact_dir))
    if pipe is None:
        raise FileNotFoundError(f"No eval.json for {env_id} (run evaluate.py)")
    base = _load(baseline_metrics_path(env_id, artifact_dir))

    def _succ(d):
        if d is None:
            return None
        if "success_rate" in d:
            return d["success_rate"]
        s = d.get("success")
        return float(np.mean(s)) if s else None

    pipe_succ = _succ(pipe)
    base_succ = _succ(base)
    has_success = pipe_succ is not None or base_succ is not None

    panels = [
        ("Mean episodic return", pipe["mean_return"],
         base["mean_return"] if base else np.nan, "%.0f", None),
        ("Mean episode length", pipe["mean_length"],
         base["mean_length"] if base else np.nan, "%.0f", None),
    ]
    if has_success:
        panels.append(("Success rate",
                       pipe_succ if pipe_succ is not None else np.nan,
                       base_succ if base_succ is not None else np.nan,
                       "%.2f", (0, 1.05)))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, pv, bv, fmt, ylim) in zip(axes, panels):
        bars = ax.bar(["Pipeline", "Baseline"], [pv, bv], color=[PIPE_C, BASE_C])
        ax.bar_label(bars, fmt=fmt, padding=3)
        ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
    fig.suptitle(f"{env_id}: pipeline vs RL-Zoo3 baseline "
                 f"({pipe['n_episodes']} episodes)", fontsize=13)
    return _save(plots_dir(env_id, artifact_dir) / "eval_metrics_bars.pdf")


PER_ENV_PLOTS = [
    plot_training_phases,
    plot_bt_loss_per_epoch,
    plot_bt_vs_env,
    plot_eval_metrics_bars,
]


def make_all_per_env(env_id: str, artifact_dir: str = "artifacts") -> List[Path]:
    out = []
    for fn in PER_ENV_PLOTS:
        try:
            out.append(fn(env_id, artifact_dir))
        except Exception as e:
            print(f"  [skip {fn.__name__}] {type(e).__name__}: {e}")
    return out
