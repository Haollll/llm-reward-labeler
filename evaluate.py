"""Evaluate a trained pipeline policy on the TRUE environment reward and compare
it against the RL-Zoo3-trained PPO baseline.

Both policies are scored on the raw environment reward (what we actually care
about), over the same number of episodes, and a labelled comparison bar chart is
saved as a PDF under artifacts/<env>/plots/.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import helper  # noqa: F401  (applies env-warning suppression on import)
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from env_setup import eval_with_components
from helper import load_task, task_for_env, success_fn_for_env
import baseline as baseline_mod
from paths import policy_dir, plots_dir, eval_metrics_path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a pipeline policy vs the RL-Zoo3 baseline")
    parser.add_argument("--env", default="HalfCheetah-v4")
    parser.add_argument("--task", default=None)
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--baseline-dir", default="baselines")
    parser.add_argument("--policy-path", default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip the RL-Zoo3 baseline comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_name = args.task or task_for_env(args.env)
    _ = load_task(task_name)  # validates the task file exists

    policy_path = Path(args.policy_path) if args.policy_path else (
        policy_dir(args.env, args.artifact_dir) / "policy.zip"
    )
    if not policy_path.exists():
        raise FileNotFoundError(f"No pipeline policy at {policy_path} (train one with train.py)")

    sfn = success_fn_for_env(args.env)

    # ── pipeline policy on TRUE env reward ───────────────────
    env = gym.make(args.env)
    policy = PPO.load(str(policy_path), device="cpu")
    pipe = eval_with_components(env, lambda o: policy.predict(o, deterministic=True)[0],
                                reward_fn=None, n_episodes=args.episodes)
    env.close()
    ep_rewards = [float(x) for x in pipe["episode_env_rewards"]]
    ep_lengths = [int(x) for x in pipe["episode_lengths"]]
    pipe_ret = float(np.mean(ep_rewards))
    pipe_len = float(np.mean(ep_lengths))
    pipe_succ = (float(np.mean([1.0 if sfn(l, r) else 0.0
                                for l, r in zip(ep_lengths, ep_rewards)]))
                 if sfn is not None else None)

    print(f"Pipeline | {args.episodes} ep | return {pipe_ret:10.1f} | length {pipe_len:6.0f}"
          + (f" | success {pipe_succ:.0%}" if pipe_succ is not None else ""))

    # ── persist pipeline eval results (drives the cross-env plot) ──
    eval_result = {
        "env_id":              args.env,
        "n_episodes":          args.episodes,
        "episode_env_rewards": ep_rewards,
        "episode_lengths":     ep_lengths,
        "mean_return":         pipe_ret,
        "mean_length":         pipe_len,
        "success_rate":        pipe_succ,
    }
    eval_json = eval_metrics_path(args.env, args.artifact_dir)
    eval_json.parent.mkdir(parents=True, exist_ok=True)
    eval_json.write_text(json.dumps(eval_result, indent=2))

    # ── RL-Zoo3 baseline on TRUE env reward ──────────────────
    base = None
    if not args.no_baseline:
        try:
            base = baseline_mod.evaluate_baseline(
                args.env, n_episodes=args.episodes, baseline_dir=args.baseline_dir,
                artifact_dir=args.artifact_dir, success_fn=sfn,
            )
            base_succ = (np.mean(base["success"]) if base.get("success") else None)
            print(f"Baseline | {args.episodes} ep | return {base['mean_return']:10.1f} | "
                  f"length {base['mean_length']:6.0f}"
                  + (f" | success {base_succ:.0%}" if base_succ is not None else ""))
        except Exception as e:
            print(f"[baseline skipped] {type(e).__name__}: {e}")

    # ── comparison bar chart (PDF) ───────────────────────────
    out = plots_dir(args.env, args.artifact_dir) / "pipeline_vs_baseline.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    labels = ["Pipeline (LLM reward)"]
    values = [pipe_ret]
    if base is not None:
        labels.append("RL-Zoo3 PPO baseline")
        values.append(base["mean_return"])
    plt.figure(figsize=(6, 5))
    bars = plt.bar(labels, values, color=["#4C72B0", "#DD8452"][:len(labels)])
    plt.bar_label(bars, fmt="%.0f", padding=3)
    plt.ylabel("Mean episodic return (true env reward)")
    plt.title(f"{args.env}: pipeline vs RL-Zoo3 baseline\n({args.episodes} episodes)")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"Plot saved → {out}")


if __name__ == "__main__":
    main()
