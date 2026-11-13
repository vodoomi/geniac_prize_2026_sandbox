# Geniac Prize 2026

Robosuite の環境構築から、`cube_restack` の少数データSFTまでの手順です（B300 × 1）。上から順に、 Bash セッションで実行してください。

## 事前準備

- SSH のユーザー名と秘密鍵のパスは、自分のものに置き換えてください。
- [facebook/sam3](https://huggingface.co/facebook/sam3) のアクセス承認を取得し、そのアカウントの読み取り用 Hugging Face トークンを用意してください。

## 環境構築

### 1. ログインノード・計算ノードへ接続

```bash
# ログインノードへ接続
ssh <ユーザー名>@157.7.234.55 -i ~/.ssh/id_ed25519_gmo -p 8822
```

ログインノードで実行します。以降の作業は、確保した計算ノード上で行ってください。

```bash
# ログインノードから計算ノードへ接続
srun --partition part-group_9fce22 --gres=gpu:1 --time=06:00:00 --pty bash
```

### 2. チーム共有領域に作業ディレクトリを作成

個人ホームはディスク容量が100GBしかないため、チーム共有領域に作業ディレクトリを作成する。

```bash
# 自分のユーザー名を確認する
id -un
# グループ名を確認する
id -gn
# グループ共有領域のパスを設定
group_dir="/home/$(id -gn)"
# 共有領域が存在することとアクセス権を確認
ls -ld "$group_dir"
# 共有領域に作業ディレクトリを作成
mkdir -p "$group_dir/$(id -un)"
# リポジトリをクローン
work_dir="$group_dir/$(id -un)/geniac_prize_2026"
git clone https://github.com/vodoomi/geniac_prize_2026_sandbox "$work_dir"
git submodule update --init
# 作業ディレクトリに移動
cd "$work_dir"
```

### 3. CLI ツールをインストール

以下のコマンドはダウンロードが含まれるため、必ず計算ノードで実行する。

```bash
# claude
curl -fsSL https://claude.ai/install.sh | bash
# codex
curl -fsSL https://chatgpt.com/codex/install.sh | sh
# uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 4. キャッシュ・CUDA の環境変数を設定

`work_dir` を設定した同じシェルで実行します。`~/.bashrc` への追記は初回のみ行ってください。

```bash
# 環境変数を設定
cat >> ~/.bashrc <<EOF

# Geniac cache settings
export GENIAC_WORK_ROOT="$(dirname -- "$work_dir")"
export HF_HUB_CACHE="\$GENIAC_WORK_ROOT/.cache/huggingface/hub"
export UV_CACHE_DIR="\$GENIAC_WORK_ROOT/.cache/uv"
export TORCH_HOME="\$GENIAC_WORK_ROOT/.cache/torch"

# CUDA settings
export CUDA_HOME=/opt/share/modules/cuda/12.8.1
EOF
source ~/.bashrc
```

### 5. ベースライン環境を構築・Hugging Face にログイン

`hf auth login` では SAM3 のアクセス承認を取得したアカウントのトークンを使用します。

```bash
uv pythoninstall 3.10 3.12
cd third_party/geniac_prize_2026_baseline
uv venv
uv pip install -e ".[dev]"
source .venv/bin/activate
uv tool install huggingface_hub
hf auth login
```

### 6. CaP-X・Qwen を準備

```bash
# SSHではなくHTTPSでクローンするように変更
sed -i 's|git@github.com:matsuolab/geniac_prize_2026_code\.git|https://github.com/matsuolab/geniac_prize_2026_code.git|' patches/geniac_prize/setup.sh
# Cap-Xを作る
bash patches/geniac_prize/setup.sh
# 指定バージョンをダウンロード
hf download Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
# ダウンロード先を指定してテンプレートを生成
geniac-baseline template \
  --model-dir "$HF_HUB_CACHE/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0" \
  --out runs/qwen38_low_reasoning.jinja
```

### 7. Robosuite・推論サーバーの仮想環境を構築

```bash
cd "$work_dir/third_party/geniac_prize_2026_baseline/geniac_prize_2026_code"
# Robosuite
uv venv .venv --python 3.10
source .venv/bin/activate
uv sync --active --frozen --extra robosuite
python capx/third_party/robosuite/robosuite/scripts/setup_macros.py
# B300で必要なNVRTCの修正
uv pip install "nvidia-cuda-nvrtc-cu12==12.9.86"
deactivate
```

```bash
# 推論サーバー
uv venv .venv-serve --python 3.12
source .venv-serve/bin/activate
uv pip install --no-config --torch-backend=cu130 \
  vllm==0.26.0 "transformers>=5" huggingface_hub
deactivate
cd ..
```

### 8. CUDA を読み込み・動作確認

```bash
module load cuda/12.8.1
export CUDA_HOME=/opt/share/modules/cuda/12.8.1
# 確認
# nvcc --version に release 12.8 が表示されればOK
echo "$CUDA_HOME"
nvcc --version
```

### 9. データ変換用の依存・Molmo を準備

収集後のタグ付けには、baseline側にもtokenizer用の依存が必要です。

```bash
uv pip install --python .venv/bin/python -e ".[tag]"
```

```bash
# Molmoをダウンロードする
geniac_prize_2026_code/.venv-serve/bin/hf download allenai/Molmo2-8B --revision e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b
```

### 10. 学習での依存を準備

SFTやDMPOを実行するには、baseline側に学習用の依存が必要です。

```bash
uv pip install -e ".[train]" -r requirements-train.txt \
  --extra-index-url https://download.pytorch.org/whl/cu130
```

## データ収集と動作確認

### 1. 自分用の収集設定を作成

`cube_restack` を4試行（学習用 seed 301–302、検証用 seed 391–392）で実行する設定です。

```bash
cp configs/collect.yaml configs/my_collect.yaml

# 少数データでの動作確認用に、`configs/my_collect.yaml` の内容を以下のように書き換える。
.venv/bin/python - <<'PY'
import os
from pathlib import Path
import yaml

hub = Path(os.environ["HF_HUB_CACHE"])
qwen = hub / "models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
molmo = hub / "models--allenai--Molmo2-8B/snapshots/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"

for model in (qwen, molmo):
    if not (model / "config.json").is_file():
        raise FileNotFoundError(f"モデルが見つかりません: {model}")

path = Path("configs/my_collect.yaml")
config = yaml.safe_load(path.read_text())

config.update({
    "model_path": str(qwen),
    "molmo_path": str(molmo),
    "b300_workarounds": True,
    "output_dir": "../runs/collect_smoke",
    "tasks": [{
        "task": "cube_restack",
        "config": "capx/geniac_prize/robosuite/cube_restack.yaml",
        "seeds": ["301-302", "391-392"],
    }],
})

extra_env = dict(config.get("extra_env") or {})
extra_env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
config["extra_env"] = extra_env

path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
print(path.read_text())
PY
```

### 2. 収集・変換を実行

収集後、成功・失敗データの変換、タグ付け、成功データの学習用・検証用分割と検証まで実行します。

```bash
SPEC=configs/my_collect.yaml \
DATA=runs/data_smoke \
RUN_GPU=1 \
TOKENIZER="$HF_HUB_CACHE/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0" \
bash examples/00_collect_data.sh
```

### 3. 評価・学習の設定を作成

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import copy
import yaml

def load(name):
    return yaml.safe_load(Path(f"configs/{name}.yaml").read_text())

def save(name, config):
    path = Path(f"configs/{name}.yaml")
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    print(f"保存: {path}")

collect = load("my_collect")

# 学習前評価：cube_restack、評価用seed 401–402
base = load("eval_base")
base.update({
    "model_path": collect["model_path"],
    "molmo_path": collect["molmo_path"],
    "b300_workarounds": True,
    "output_dir": "../runs/eval_base_smoke",
    "tasks": [{
        "task": "cube_restack",
        "config": "capx/geniac_prize/robosuite/cube_restack.yaml",
        "seeds": ["401-402"],
    }],
})
base.setdefault("extra_env", {})["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
save("my_eval_base_smoke", base)

# SFT
sft = load("train_sft")
sft.update({
    "data": "../runs/data_smoke/split",
    "model": collect["model_path"],
    "output_dir": "../runs/train_sft_smoke",
})
save("my_train_sft_smoke", sft)

# DMPO：SFTの検証lossが最小のepochを使用
dmpo = load("train_dmpo")
dmpo.update({
    "data": "../runs/data_smoke/pairs",
    "model": collect["model_path"],
    "init_adapter": "../runs/train_sft_smoke",
    "output_dir": "../runs/train_dmpo_smoke",
})
save("my_train_dmpo_smoke", dmpo)

# 学習後評価：学習前と同じタスク・seed
trained = copy.deepcopy(base)
trained.update({
    "output_dir": "../runs/eval_trained_smoke",
    "adapter": "../runs/train_dmpo_smoke/adapter",
    "lora_name": "dmpo-lora",
})
save("my_eval_trained_smoke", trained)
PY
```

### 4. 学習前の評価

```bash
SPEC=configs/my_eval_base_smoke.yaml \
RUN_GPU=1 \
bash examples/01_eval_base_cap_agent0.sh
```

終了コード0と、`n=2・missing=0`を確認する。

```bash
echo "終了コード: $?"
.venv/bin/python -m json.tool runs/eval_base_smoke/score.json
```

### 5. SFTの学習

```bash
CONFIG=configs/my_train_sft_smoke.yaml \
RUN_GPU=1 \
bash examples/02_train_sft.sh
```
