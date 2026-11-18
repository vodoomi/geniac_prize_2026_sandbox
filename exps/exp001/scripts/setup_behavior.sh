#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/behavior_env.sh"
exec 9>"$EXP_DIR/logs/behavior_runtime.lock"
flock -n 9 || { echo '[error] BEHAVIOR setup or collection is already running' >&2; exit 1; }
LOG="$EXP_DIR/logs/setup_behavior-${SLURM_JOB_ID}-$(date +%Y%m%dT%H%M%S)-$$.log"
exec > >(tee -a "$LOG") 2>&1
echo "[setup] log: $LOG"
rm -f "$RUNTIME_DIR/.setup-complete"
cd "$PROJECT_DIR"
UPSTREAM="$RUNTIME_DIR/BEHAVIOR-1K"
COMMIT=b1979916ec1549b10a4e65e630bc6504a9af1b00
if [[ ! -d "$UPSTREAM/.git" ]]; then
  git clone --depth 1 --branch v3.9.2 https://github.com/StanfordVL/BEHAVIOR-1K.git "$UPSTREAM"
fi
[[ "$(git -C "$UPSTREAM" rev-parse HEAD)" == "$COMMIT" ]] || { echo '[error] Unexpected BEHAVIOR revision' >&2; exit 1; }
"$BASELINE_DIR/.venv/bin/python" "$EXP_DIR/src/behavior_collect/patch_runtime.py"
git -C "$UPSTREAM" diff > "$RUNTIME_DIR/compatibility.patch"
if [[ ! -x "$BEHAVIOR_PY" ]]; then
  uv venv "$RUNTIME_DIR/venv" --python 3.11 --seed
fi
uv pip install --python "$BEHAVIOR_PY" --no-config 'setuptools>=71,<81' wheel ninja 'numpy<2' \
  torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
uv pip install --python "$BEHAVIOR_PY" --no-config 'isaacsim[all,extscache]==5.1.0' \
  --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
# Preserve the simulator's torch/numpy while adding CaP-X and motion planning.
cat > "$RUNTIME_DIR/constraints.txt" <<'EOF'
torch==2.7.0
torchvision==0.22.0
torchaudio==2.7.0
numpy<2
setuptools<81
jax<0.7
jaxlib<0.7
EOF
printf '%s\n' 'nvidia-cuda-nvrtc-cu12==12.9.86' \
  'jaxls @ git+https://github.com/brentyi/jaxls.git@6fe7cf9d56223736b2c0272d080ec7346203c648' \
  > "$RUNTIME_DIR/overrides.txt"
uv pip install --python "$BEHAVIOR_PY" --no-config -c "$RUNTIME_DIR/constraints.txt" \
  --overrides "$RUNTIME_DIR/overrides.txt" \
  -e "$UPSTREAM/bddl3" -e "$UPSTREAM/OmniGibson[primitives]" --no-build-isolation
uv pip install --python "$BEHAVIOR_PY" --no-config -c "$RUNTIME_DIR/constraints.txt" \
  --overrides "$RUNTIME_DIR/overrides.txt" \
  gymnasium tyro 'pydantic>=2.7' rich scipy matplotlib 'imageio[ffmpeg]' requests tqdm h5py \
  einops decord pycocotools pyyaml omegaconf open3d robot_descriptions yourdfpy viser trimesh \
  msgpack 'ray==2.48.0' mediapy uvicorn fastapi 'transformers<5' openai pyrender \
  cloudpickle iopath timm av dm-tree hydra-core 'ftfy==6.1.1' nvidia-cuda-nvrtc-cu12==12.9.86 \
  'pyroki @ git+https://github.com/chungmin99/pyroki.git@95afccc22658c461ab1042a048ae4e9c24bc2a47'
uv pip install --python "$BEHAVIOR_PY" --no-config --no-deps --no-build-isolation \
  -e "$CAPX_DIR/capx/third_party/sam3" -e "$CAPX_DIR/capx/third_party/contact_graspnet_pytorch"
# cuRobo's wheel omits .h files required by its fallback runtime compiler.
"$BEHAVIOR_PY" - <<'PY'
import os
from pathlib import Path
import shutil
import sysconfig
target = Path(sysconfig.get_path("purelib")) / "curobo/curobolib/cpp"
headers = list(Path(os.environ["UV_CACHE_DIR"]).glob("git-v*/checkouts/*/78612f4/src/curobo/curobolib/cpp/*.h"))
for header in headers:
    shutil.copy2(header, target / header.name)
assert (target / "helper_math.h").is_file(), "cuRobo JIT headers missing"
PY
# libGLU is absent on the cluster. Extract the Ubuntu package locally without sudo.
if [[ ! -f "$RUNTIME_DIR/lib/usr/lib/x86_64-linux-gnu/libGLU.so.1" ]] && ! ldconfig -p | grep -q 'libGLU.so.1'; then
  curl --fail --location --retry 3 --connect-timeout 20 --max-time 180 \
    https://archive.ubuntu.com/ubuntu/pool/main/libg/libglu/libglu1-mesa_9.0.2-1_amd64.deb \
    -o "$TMPDIR/libglu.deb"
  dpkg-deb -x "$TMPDIR/libglu.deb" "$RUNTIME_DIR/lib"
fi
if [[ ! -f "$RUNTIME_DIR/lib/usr/lib/x86_64-linux-gnu/libvulkan.so.1" ]] && ! ldconfig -p | grep -q 'libvulkan.so.1'; then
  curl --fail --location --retry 3 --connect-timeout 20 --max-time 180 \
    https://archive.ubuntu.com/ubuntu/pool/main/v/vulkan-loader/libvulkan1_1.3.204.1-2_amd64.deb \
    -o "$TMPDIR/libvulkan.deb"
  dpkg-deb -x "$TMPDIR/libvulkan.deb" "$RUNTIME_DIR/lib"
fi
if [[ ! -f "$RUNTIME_DIR/lib/usr/lib/x86_64-linux-gnu/libOpenGL.so.0" ]] && ! ldconfig -p | grep -q 'libOpenGL.so.0'; then
  curl --fail --location --retry 3 --connect-timeout 20 --max-time 180 \
    https://archive.ubuntu.com/ubuntu/pool/main/libg/libglvnd/libopengl0_1.4.0-1_amd64.deb \
    -o "$TMPDIR/libopengl.deb"
  dpkg-deb -x "$TMPDIR/libopengl.deb" "$RUNTIME_DIR/lib"
fi
echo '[setup] Downloading robot assets, scene assets and public task instances'
"$BEHAVIOR_PY" - <<'PY'
from omnigibson.utils import asset_utils as a
a.download_omnigibson_robot_assets()
a.download_behavior_1k_assets(accept_license=True)
a.download_2026_challenge_task_instances()
a.download_2025_challenge_task_instances()
PY
"$BASELINE_DIR/.venv/bin/python" "$EXP_DIR/src/behavior_collect/prepare.py"
"$BASELINE_DIR/.venv/bin/python" "$EXP_DIR/src/behavior_collect/plan.py" --spec "$EXP_DIR/configs/collect_behavior.yaml" --check
"$BEHAVIOR_PY" -m pip freeze > "$RUNTIME_DIR/installed.txt"
timeout --kill-after=30 900 "$BEHAVIOR_PY" "$EXP_DIR/src/behavior_collect/check_setup.py"
printf '%s\n' "$COMMIT" > "$RUNTIME_DIR/.setup-complete"
echo '[setup] BEHAVIOR setup completed, including renderer and instance-0 checks'
