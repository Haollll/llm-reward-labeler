#!/usr/bin/env bash
# Render every env's trained pipeline policy next to the RL-Zoo3 baseline,
# writing a side-by-side GIF per env to artifacts/<env>/render_compare.gif.
#
# Rendering is OFFSCREEN (MuJoCo OSMesa) because a live window can't be created
# on this WSL/software-GL box; evaluate.py sets MUJOCO_GL/PYOPENGL_PLATFORM.
#
# Usage:  ./run_renders.sh                 # side-by-side pipeline vs baseline
#         RENDER_FLAGS="" ./run_renders.sh # pipeline only (render.gif)
#         EPISODES=2 ./run_renders.sh      # more episodes per env
#
# Each env gets a zoom tuned so the agent stays a sensible size while the wider
# checker scene is visible. Pendulum is classic-control (no MuJoCo scene/zoom)
# and needs `pip install "gymnasium[classic_control]"`, so it is rendered flat.

set -uo pipefail   # NOTE: no -e, so one env failing doesn't abort the rest

# env:zoom pairs (Ant moves slower, tolerates a wider view)
ENV_ZOOMS=(
    "HalfCheetah-v4:2"
    "Hopper-v4:2"
    "Walker2d-v4:2"
    "Ant-v4:3"
    "Swimmer-v4:2"
    "Pendulum-v1:1"
)

RENDER_FLAGS="${RENDER_FLAGS:---compare}"
EPISODES="${EPISODES:-1}"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/renders_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging all output to: $LOG_FILE"

echo "=============================================================="
echo " Rendering ${#ENV_ZOOMS[@]} envs (flags: '${RENDER_FLAGS}', $EPISODES ep each)"
echo "=============================================================="

failed=()
for PAIR in "${ENV_ZOOMS[@]}"; do
    ENV="${PAIR%%:*}"
    ZOOM="${PAIR##*:}"
    echo "----- render $ENV (zoom $ZOOM) -----"
    if ! python evaluate.py --env "$ENV" --render $RENDER_FLAGS \
            --render-episodes "$EPISODES" --zoom "$ZOOM"; then
        echo "[FAILED] $ENV"
        failed+=("$ENV")
    fi
done

echo "=============================================================="
if [ ${#failed[@]} -eq 0 ]; then
    echo " Done. GIFs in artifacts/<env>/ (render_compare.gif / render.gif)"
else
    echo " Done with failures: ${failed[*]}"
fi
echo "=============================================================="
