#!/usr/bin/env python3
"""
Reward model debug: 4 controlled experiments in oracle mode.

Usage:  python run_experiments.py
Output: console comparison table + results/ablation.csv
"""

import csv, sys, time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

from trainer import Trainer, TrainerConfig

# ─── shared base ─────────────────────────────────────────────────────────────

COMMON = dict(
    use_oracle    = True,
    rounds        = 5,
    n_queries     = 10,
    ppo_steps     = 20_000,
    eval_every    = 99,       # skip slow env-reward eval during ablation
    progress_bar  = False,
    verbose       = True,
)

EXPERIMENTS = [
    ("1-baseline",     dict(reward_epochs=20,  lambda_smooth=1.0, dynamic_batch=False)),
    ("2-no_smooth",    dict(reward_epochs=20,  lambda_smooth=0.0, dynamic_batch=False)),
    ("3-more_epochs",  dict(reward_epochs=100, lambda_smooth=1.0, dynamic_batch=False)),
    ("4-dyn_batch",    dict(reward_epochs=20,  lambda_smooth=1.0, dynamic_batch=True)),
]

# ─── run ─────────────────────────────────────────────────────────────────────

all_results = {}   # name → {"losses": [...], "accs": [...]}

for name, overrides in EXPERIMENTS:
    print(f"\n{'='*68}")
    print(f"  {name}")
    print('='*68)

    cfg = TrainerConfig(**{**COMMON, **overrides})
    t   = Trainer(cfg)

    t0 = time.time()
    t.run(n_rounds=cfg.rounds)
    elapsed = time.time() - t0

    all_results[name] = {
        "losses": list(t.reward_losses),
        "accs":   list(t.label_accuracies),
    }
    print(f"  [{name}] done in {elapsed/60:.1f} min | "
          f"rounds captured: {len(t.reward_losses)}")

# ─── comparison table ─────────────────────────────────────────────────────────

n_rounds = max(len(v["losses"]) for v in all_results.values())
round_labels = ["cs"] + [str(i) for i in range(1, n_rounds)]  # cs = cold-start

print("\n\n" + "="*78)
print("  COMPARISON TABLE  — Rnd0=cold-start, Rnd1-5=training rounds")
print("  Columns: loss / acc")
print("="*78)

col_w = 12
hdr = f"{'Experiment':<22}" + "".join(f"  Rnd{r:<{col_w-4}}" for r in round_labels)
print(hdr)
print("-" * len(hdr))

for name, data in all_results.items():
    row = f"{name:<22}"
    for i in range(n_rounds):
        if i < len(data["losses"]):
            cell = f"{data['losses'][i]:.3f}/{data['accs'][i]:.0%}"
        else:
            cell = "-/-"
        row += f"  {cell:<{col_w}}"
    print(row)

print()
print(f"  Random-guess loss (3 members): 3×ln(2) ≈ {3*np.log(2):.3f}")
print()

# ─── trend summary ────────────────────────────────────────────────────────────

print("="*78)
print("  TREND SUMMARY")
print("="*78)
for name, data in all_results.items():
    ls, acs = data["losses"], data["accs"]
    if not ls:
        continue
    dl = ls[-1] - ls[0]
    da = acs[-1] - acs[0] if acs else 0
    loss_tag = "↓ drops" if dl < -0.05 else ("↑ rises" if dl > 0.05 else "→ flat")
    acc_tag  = "↑ learns" if da > 0.03 else ("↓ drops" if da < -0.03 else "→ flat")
    print(f"  {name:<22}  loss {ls[0]:.3f}→{ls[-1]:.3f} ({loss_tag})  "
          f"acc {acs[0]:.0%}→{acs[-1]:.0%} ({acc_tag})")

# ─── save CSV ────────────────────────────────────────────────────────────────

out_path = Path("results/ablation.csv")
out_path.parent.mkdir(exist_ok=True)

with open(out_path, "w", newline="") as f:
    writer = csv.writer(f)
    header = ["experiment"] + [f"rnd{i}_loss" for i in range(n_rounds)] \
                             + [f"rnd{i}_acc"  for i in range(n_rounds)]
    writer.writerow(header)
    for name, data in all_results.items():
        ls  = data["losses"] + [None] * (n_rounds - len(data["losses"]))
        acs = data["accs"]   + [None] * (n_rounds - len(data["accs"]))
        row = [name] + [f"{v:.4f}" if v is not None else "" for v in ls] \
                     + [f"{v:.4f}" if v is not None else "" for v in acs]
        writer.writerow(row)

print(f"\n  CSV saved → {out_path}")
