#!/usr/bin/env bash
# Run on the currently allocated compute node before LIBERO collection.
set -euo pipefail
EXP_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PROJECT_DIR=$(cd -- "$EXP_DIR/../.." && pwd)
CAPX_DIR="$PROJECT_DIR/third_party/geniac_prize_2026_baseline/geniac_prize_2026_code"
export TMPDIR="/scratch/exp001_${USER}_${SLURM_JOB_ID:?Run on an allocated compute node}"
mkdir -p "$TMPDIR"
export UV_CACHE_DIR="$(dirname -- "$PROJECT_DIR")/.cache/uv"
cd "$CAPX_DIR"
if [ ! -x .venv-libero/bin/python ]; then
  uv venv .venv-libero --python 3.12
fi
UV_PROJECT_ENVIRONMENT=.venv-libero uv sync --frozen --extra libero --extra contactgraspnet
uv pip install --python .venv-libero/bin/python nvidia-cuda-nvrtc-cu12==12.9.86
MACROS_DIR=capx/third_party/libero_dependencies/robosuite/robosuite
if [ ! -f "$MACROS_DIR/macros_private.py" ]; then
  cp "$MACROS_DIR/macros.py" "$MACROS_DIR/macros_private.py"
fi
export LIBERO_CONFIG_PATH="$EXP_DIR/configs/libero_runtime"
.venv-libero/bin/python -c 'import capx, robosuite; from libero import benchmark; print("robosuite", robosuite.__version__); assert robosuite.__version__ == "1.4.0"; print("LIBERO import OK")'
