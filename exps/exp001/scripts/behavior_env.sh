#!/usr/bin/env bash
# Shared by setup and collection; source this file.
EXP_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PROJECT_DIR=$(cd -- "$EXP_DIR/../.." && pwd)
BASELINE_DIR="$PROJECT_DIR/third_party/geniac_prize_2026_baseline"
CAPX_DIR="$BASELINE_DIR/geniac_prize_2026_code"
RUNTIME_DIR="$EXP_DIR/runtime"
BEHAVIOR_PY="$RUNTIME_DIR/venv/bin/python"
: "${SLURM_JOB_ID:?現在割り当てられたGPU計算ノードで実行してください}"
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
# Keep packages, assets and caches on group storage, temporary builds on scratch.
export TMPDIR="/scratch/exp001_behavior_${USER}_${SLURM_JOB_ID}"
export UV_CACHE_DIR="$(dirname -- "$PROJECT_DIR")/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$UV_CACHE_DIR/python"
export HF_HOME="$(dirname -- "$PROJECT_DIR")/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$(dirname -- "$PROJECT_DIR")/.cache/torch"
export XDG_CACHE_HOME="$(dirname -- "$PROJECT_DIR")/.cache/behavior"
export XDG_DATA_HOME="$RUNTIME_DIR/user-data"
export XDG_CONFIG_HOME="$RUNTIME_DIR/user-config"
export OMNI_CONFIG_PATH="$RUNTIME_DIR/omni-config"
export OMNIGIBSON_DATA_PATH="$RUNTIME_DIR/data"
export OMNI_KIT_ACCEPT_EULA=YES OMNIGIBSON_HEADLESS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$CAPX_DIR:$EXP_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_CUDA_ARCH_LIST=10.0 MAX_JOBS=8
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME=/opt/share/modules/cuda/12.9.1
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$RUNTIME_DIR/lib/usr/lib/x86_64-linux-gnu:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$TMPDIR" "$RUNTIME_DIR" "$EXP_DIR/logs" "$XDG_CACHE_HOME" "$OMNI_CONFIG_PATH" "$OMNIGIBSON_DATA_PATH"
cat > "$OMNI_CONFIG_PATH/omniverse.toml" <<EOF
[paths]
data_root = "$XDG_DATA_HOME/ov/data"
cache_root = "$XDG_CACHE_HOME/ov"
logs_root = "$EXP_DIR/logs/ov"
EOF
ulimit -c 0
