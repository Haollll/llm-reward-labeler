#!/usr/bin/env python3
"""
Diagnose step-level vs segment-level reward model alignment.

Hypothesis: R_phi is trained on segment-level preferences but PPO uses it as
a step-level reward. If the hypothesis holds, segment-level correlation with
ground-truth will be significantly higher than step-level correlation.

Usage:  python diagnose_reward_model.py
"""

import sys
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

# ── config ────────────────────────────────────────────────────────────────────
ARTIFACT_KEY   = "HalfCheetah-v5_fb9336b92897"
REWARD_PATH    = f"artifacts/reward_models/{ARTIFACT_KEY}"
POLICY_PATH    = f"artifacts/policies/{ARTIFACT_KEY}/policy.zip"
N_EPISODES     = 5      # HalfCheetah episodes are 1000 steps each
SEG_LEN        = 50     # must match training size_segment


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r without scipy."""
    xm, ym = x - x.mean(), y - y.mean()
    denom = np.sqrt((xm ** 2).sum() * (ym ** 2).sum())
    return float((xm * ym).sum() / (denom + 1e-12))


# ── load reward model ─────────────────────────────────────────────────────────
from reward_model import RewardModel

meta = torch.load(f"{REWARD_PATH}/metadata.pt", map_location="cpu")
_env = gym.make("HalfCheetah-v5")
reward_model = RewardModel(
    env           = _env,
    ensemble_size = meta["ensemble_size"],
    hidden        = meta["hidden"],
    size_segment  = meta["size_segment"],
)
reward_model.load(REWARD_PATH)
_env.close()
print(f"Reward model loaded  (obs={meta['obs_dim']} act={meta['action_dim']})")

# ── load policy ───────────────────────────────────────────────────────────────
try:
    from stable_baselines3 import PPO
    policy = PPO.load(POLICY_PATH)
    policy_fn = lambda obs: policy.predict(obs, deterministic=True)[0]
    print(f"Policy loaded        ({POLICY_PATH})")
except Exception as e:
    policy_fn = None
    print(f"Policy load failed ({e}) — using random policy")

# ── collect rollout data ──────────────────────────────────────────────────────
# Collect complete episodes and slice into segments entirely within each episode.
# This matches training: traj_to_segment never crosses episode boundaries.

env = gym.make("HalfCheetah-v5")

all_step_r_phi: list[float] = []
all_step_r_env: list[float] = []
segments: list[dict] = []          # {"r_phi_mean": float, "r_env_sum": float}

print(f"\nCollecting {N_EPISODES} episodes × up to 1000 steps …")

for ep in range(N_EPISODES):
    ep_r_phi: list[float] = []
    ep_r_env: list[float] = []

    obs, _ = env.reset()
    step = 0
    done = False
    while not done:
        action = (
            policy_fn(obs)
            if policy_fn is not None
            else env.action_space.sample()
        )
        next_obs, env_reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        r_phi = reward_model.predict(obs, action)
        ep_r_phi.append(r_phi)
        ep_r_env.append(float(env_reward))

        obs = next_obs
        step += 1

    all_step_r_phi.extend(ep_r_phi)
    all_step_r_env.extend(ep_r_env)

    # slice into complete segments of SEG_LEN (no partial final segment)
    n_segs = len(ep_r_phi) // SEG_LEN
    for i in range(n_segs):
        s = i * SEG_LEN
        e = s + SEG_LEN
        segments.append({
            "r_phi_mean": float(np.mean(ep_r_phi[s:e])),
            "r_env_sum":  float(np.sum(ep_r_env[s:e])),
        })

    ep_env_total = sum(ep_r_env)
    ep_phi_mean  = float(np.mean(ep_r_phi))
    print(f"  ep {ep+1}: {step} steps | env_reward {ep_env_total:.1f}"
          f" | R_phi mean {ep_phi_mean:.4f} | segments {n_segs}")

env.close()

# ── step-level correlation ────────────────────────────────────────────────────
step_phi = np.array(all_step_r_phi)
step_env = np.array(all_step_r_env)
r_step   = pearsonr(step_phi, step_env)

# ── segment-level correlation ─────────────────────────────────────────────────
seg_phi = np.array([s["r_phi_mean"] for s in segments])
seg_env = np.array([s["r_env_sum"]  for s in segments])
r_seg   = pearsonr(seg_phi, seg_env)

# ── report ────────────────────────────────────────────────────────────────────
n_steps = len(step_phi)
n_segs  = len(segments)

print(f"\n{'='*62}")
print(f"  DIAGNOSIS  ({n_steps} steps | {n_segs} segments of {SEG_LEN})")
print('='*62)

print(f"\n  Step-level  Pearson r : {r_step:+.4f}")
print(f"  Segment-level Pearson r : {r_seg:+.4f}")

print(f"\n  R_phi  — mean {step_phi.mean():.4f}  std {step_phi.std():.4f}"
      f"  range [{step_phi.min():.4f}, {step_phi.max():.4f}]")
print(f"  r_env  — mean {step_env.mean():.4f}  std {step_env.std():.4f}"
      f"  range [{step_env.min():.4f}, {step_env.max():.4f}]")

print(f"\n  Seg R_phi  — mean {seg_phi.mean():.4f}  std {seg_phi.std():.4f}")
print(f"  Seg r_env  — mean {seg_env.mean():.4f}  std {seg_env.std():.4f}")

print()
abs_gap = abs(r_seg) - abs(r_step)   # positive = segment has stronger signal
inverted = r_seg < -0.2              # model assigns HIGH output to LOW-reward states

if abs_gap > 0.1 and inverted:
    verdict = (
        "SUPPORTED (with inversion) — |segment r| is substantially larger than\n"
        "  |step r|, confirming segment >> step signal. But BOTH correlations are\n"
        "  NEGATIVE: R_phi assigns high values to low-env-reward states. This means\n"
        "  R_phi learned the oracle's preferences correctly but oracle labels were\n"
        "  based on reward_fn (LLM-generated), which is anti-correlated with env\n"
        "  reward. As alpha drops, PPO maximises R_phi → drives env reward DOWN.\n"
        "  Root cause: reward_fn and env_reward are misaligned, not the segment/step gap."
    )
elif abs_gap > 0.1:
    verdict = (
        "SUPPORTED — |segment r| substantially larger than |step r|.\n"
        "R_phi has segment-level signal but per-step output is noisy.\n"
        "PPO is receiving a misleading per-step signal."
    )
elif r_step > 0.5:
    verdict = (
        "NOT SUPPORTED — step-level correlation is already strong.\n"
        "R_phi per-step signal is meaningful; the performance drop has a different root cause."
    )
elif abs(r_step) < 0.1 and abs(r_seg) < 0.2:
    verdict = (
        "INCONCLUSIVE — both correlations are near zero.\n"
        "The reward model may not have learned any useful signal at this stage."
    )
else:
    verdict = (
        f"MIXED — |segment r|={abs(r_seg):.3f} vs |step r|={abs(r_step):.3f} "
        f"(abs gap={abs_gap:+.3f}).\n"
        "Collect more data or train longer before concluding."
    )

print(f"  VERDICT: {verdict}")
print()
