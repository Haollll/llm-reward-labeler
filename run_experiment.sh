#!/usr/bin/env bash
# Full experiment pipeline:
#   1. Train the LLM-reward pipeline on all envs
#   2. Train the RL-Zoo3 PPO baselines
#   3. Evaluate pipeline vs baseline per env (writes eval.json + comparison PDF)
#   4. Generate the plot suite (per-env + cross-env)
#
# Usage:
#   ./run_experiment.sh
#
# Notes:
#   - PPO steps/round is auto-set per env to (rl-zoo3 total timesteps)/(rounds+1).
#   - This is a long, API-billed run (LLM preference labels + reflection per round).

set -euo pipefail

ENVS=(Pendulum-v1 Swimmer-v4 HalfCheetah-v4 Hopper-v4 Walker2d-v4 Ant-v4)

PIPELINE_FLAGS=(--rounds 9 --queries 25 --reward-epochs 50 --lambda-smooth 0.05 --dynamic-batch)

# Mirror all terminal output (stdout + stderr) to a timestamped log file.
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/experiment_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging all output to: $LOG_FILE"

echo "=============================================================="
echo " 1/4  Training LLM-reward pipeline on: ${ENVS[*]}"
echo "=============================================================="
python train_all.py --envs "${ENVS[@]}" "${PIPELINE_FLAGS[@]}"

echo "=============================================================="
echo " 2/4  Training RL-Zoo3 PPO baselines"
echo "=============================================================="
python train_baselines.py --envs "${ENVS[@]}"

echo "=============================================================="
echo " 3/4  Evaluating pipeline vs baseline (per env)"
echo "=============================================================="
for ENV in "${ENVS[@]}"; do
    echo "----- evaluate $ENV -----"
    python evaluate.py --env "$ENV"
done

echo "=============================================================="
echo " 4/4  Generating plots"
echo "=============================================================="
python make_plots.py --env "${ENVS[@]}"
python make_plots.py --cross-env "${ENVS[@]}"

echo "=============================================================="
echo " Done. Artifacts in artifacts/<env>/ and artifacts/cross_env/"
echo "=============================================================="
