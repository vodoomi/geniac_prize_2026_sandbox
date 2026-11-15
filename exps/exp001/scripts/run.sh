#!/usr/bin/env bash
set -euo pipefail

# 起動したディレクトリに依存しない。
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXP_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
PROJECT_DIR=$(cd -- "$EXP_DIR/../.." && pwd)
BASELINE_DIR="$PROJECT_DIR/third_party/geniac_prize_2026_baseline"
cd "$PROJECT_DIR"

: "${SLURM_JOB_ID:?GPUを確保した計算ノード上で実行してください}"
export TMPDIR="/scratch/exp001_${USER}_${SLURM_JOB_ID}"
export HF_HUB_CACHE="$(dirname -- "$PROJECT_DIR")/.cache/huggingface/hub"
export UV_CACHE_DIR="$(dirname -- "$PROJECT_DIR")/.cache/uv"
export TORCH_HOME="$(dirname -- "$PROJECT_DIR")/.cache/torch"
mkdir -p "$TMPDIR" "$EXP_DIR/logs"

# 同じ出力先への二重起動を防ぐ。
exec 9>"$EXP_DIR/logs/collect_libero.lock"
if ! flock -n 9; then
  echo "[error] LIBERO収集スクリプトが既に実行中です。" >&2
  exit 1
fi

LOG="$EXP_DIR/logs/collect_libero-${SLURM_JOB_ID}-$(date +%Y%m%dT%H%M%S)-$$.log"
exec > >(tee -a "$LOG") 2>&1
echo "[run] log: $LOG"
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

module load cuda/12.8.1
export CUDA_HOME=/opt/share/modules/cuda/12.8.1

for executable in "$BASELINE_DIR/.venv/bin/geniac-baseline" \
  "$BASELINE_DIR/geniac_prize_2026_code/.venv-libero/bin/python" \
  "$BASELINE_DIR/geniac_prize_2026_code/.venv-serve/bin/python"; do
  if [[ ! -x "$executable" ]]; then
    echo "[error] 実行環境が見つかりません: $executable" >&2
    echo "[error] 先に bash $SCRIPT_DIR/setup_libero.sh を実行してください。" >&2
    exit 1
  fi
done

for part in 01 02 03 04; do
  name="collect_libero_part${part}"
  spec="$EXP_DIR/configs/${name}.yaml"
  run_dir="$EXP_DIR/runs/$name"
  done_file="$run_dir/.collection-complete"
  spec_digest=$(sha256sum "$spec")
  spec_digest=${spec_digest%% *}

  if [[ -f "$done_file" ]]; then
    if [[ "$(< "$done_file")" != "$spec_digest" ]]; then
      echo "[error] $name は実行済みですがYAMLが変更されています。別の出力先を使用してください。" >&2
      exit 1
    fi
    echo "[skip] $name: 前回の実行が正常終了しています。"
    continue
  fi

  if [[ -e "$run_dir" ]]; then
    echo "[error] $name の出力先が既に存在し、正常終了の記録がありません: $run_dir" >&2
    echo "[error] 重複収集を防ぐため停止します。未完了seedを確認し、別の出力先で収集してください。" >&2
    exit 1
  fi

  echo "[run] $name: 計画を生成します。"
  "$BASELINE_DIR/.venv/bin/geniac-baseline" collect plan --spec "$spec"

  echo "[run] $name: 収集を開始します。"
  time bash "$run_dir/run.sh"
  # エラーや中断ではここに到達しない。試行成功率とは区別する。
  printf '%s\n' "$spec_digest" > "$done_file"
  echo "[done] $name: 収集スクリプトが正常終了しました。"
done

echo "[done] 全4partの実行が正常終了しました。"
