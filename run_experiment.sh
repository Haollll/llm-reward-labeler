#!/usr/bin/env bash
# Full v2 experiment pipeline:
#   1. Train the two-phase LLM-reward pipeline on all envs
#   2. Evaluate pipeline vs RL-Zoo3 baseline per env (writes eval.json,
#      baseline/metrics.json, bt_vs_env.json)
#   3. Generate the plot suite per env
#
# Baselines are reused from the v1 project via the `baselines/` symlink — no
# baseline retraining here. If the symlink is missing, train baselines with
# rl_zoo3 first (see the v1 train_baselines.py).
#
# Usage:  ./run_experiment.sh

set -euo pipefail

ENVS=(Pendulum-v1 Swimmer-v4 HalfCheetah-v4 Hopper-v4 Walker2d-v4 Ant-v4)

# Phase I = 5 rounds (coded reward + reflection), Phase II = 5 rounds (alpha blend)
TRAIN_FLAGS=(--k1 5 --k2 5 --ppo-steps 100000 --num-trajs 10 \
             --reward-epochs 50 --alpha 0.5 --eval-episodes 100 --no-progress-bar)

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/experiment_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging all output to: $LOG_FILE"

echo "=============================================================="
echo " 1/3  Training v2 pipeline on: ${ENVS[*]}"
echo "=============================================================="
python train_all.py --envs "${ENVS[@]}" "${TRAIN_FLAGS[@]}"

echo "=============================================================="
echo " 2/3  Evaluating pipeline vs baseline (per env)"
echo "=============================================================="
for ENV in "${ENVS[@]}"; do
    echo "----- evaluate $ENV -----"
    python evaluate.py --env "$ENV" --episodes 100
done

echo "=============================================================="
echo " 3/3  Generating plots"
echo "=============================================================="
python make_plots.py --env "${ENVS[@]}"

echo "=============================================================="
echo " Done. Artifacts in artifacts/<env>/ (plots under artifacts/<env>/plots/)"
echo "=============================================================="
