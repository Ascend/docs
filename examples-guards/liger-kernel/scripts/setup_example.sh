#!/usr/bin/env bash
# Prepare the CI environment for one supported Liger-Kernel example.
# $1 is the manifest profile. Unknown profiles fail before any install.
#
# Liger itself is installed from TARGET_ROOT - the release checkout the
# shared engine resolved - so the guarded kernel source and the examples
# are always the same release.
#
# Stack: liger's own setup.py and Huawei's Ascend-CI recipe pin
# torch 2.9.0 + torch_npu 2.9.0 plus triton-ascend 3.2.2 (the Ascend Triton
# fork is what the _ascend backend compiles against). The CANN base image
# does not ship a torch build, so the pair is installed explicitly and then
# verified; never let pip resolve a plain PyPI torch over torch_npu's match.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"
# Apply to every pip invocation, including transitive ModelScope dependencies.
export PIP_CONSTRAINT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/constraints-npu.txt"

ASCEND_PIP_INDEX=https://repo.huaweicloud.com/ascend/repos/pypi
TRITON_ASCEND_INDEX=https://triton-ascend.osinfra.cn/pypi/simple
FALLBACK_PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
CLUSTER_PIP_HOST=cache-service.nginx-pypi-cache.svc.cluster.local
export CLUSTER_PIP_INDEX="http://${CLUSTER_PIP_HOST}/pypi/simple"

select_pip_index() {
  # Runners live in mainland China: prefer the cluster pip cache, fall
  # back to the Tsinghua mirror.
  if python -c "
import urllib.error
import urllib.request
try:
    urllib.request.urlopen('${CLUSTER_PIP_INDEX}', timeout=3)
except urllib.error.HTTPError:
    pass
" 2>/dev/null; then
    export PIP_INDEX_URL="$CLUSTER_PIP_INDEX"
    export PIP_TRUSTED_HOST="$CLUSTER_PIP_HOST"
  else
    export PIP_INDEX_URL="$FALLBACK_PIP_INDEX"
    unset PIP_TRUSTED_HOST
  fi
  echo "pip index: $PIP_INDEX_URL"
}

verify_torch_stack() {
  # torch_npu's compiled op library is bound to one exact torch build; this
  # is the pair liger and Huawei's Ascend CI both declare.
  python - <<'PY'
import torch
import torch_npu

print("torch", torch.__version__, "torch_npu", torch_npu.__version__)
if not torch.__version__.startswith("2.9.0"):
    raise SystemExit(f"expected torch 2.9.0, got {torch.__version__}")
if not torch_npu.__version__.startswith("2.9.0"):
    raise SystemExit(f"expected torch_npu 2.9.0, got {torch_npu.__version__}")
if not torch.npu.is_available() or torch.npu.device_count() < 1:
    raise SystemExit("NPU unavailable; refusing to run examples on CPU")
print("NPU devices:", torch.npu.device_count())
PY
}

ensure_torch_stack() {
  # The CANN base image may ship no PyTorch at all, so probe for a matching
  # pair and install the two pinned wheels from the cluster pip cache plus
  # the Ascend index when it is missing. Same shape as the tensordict setup,
  # which hit this on its own first run.
  if python - <<'PY'
try:
    import torch
    import torch_npu
except Exception as exc:
    print(f"torch stack probe failed: {exc}")
    raise SystemExit(1)
print(f"found torch={torch.__version__} torch_npu={torch_npu.__version__}")
raise SystemExit(
    0 if torch.__version__.startswith("2.9.0")
    and torch_npu.__version__.startswith("2.9.0") else 1
)
PY
  then
    echo "reusing compatible torch/torch_npu stack"
  else
    echo "installing torch==2.9.0 torch_npu==2.9.0.post2"
    python -m pip install \
      --index-url "$PIP_INDEX_URL" \
      --extra-index-url "$ASCEND_PIP_INDEX" \
      torch==2.9.0 torch_npu==2.9.0.post2
  fi
  verify_torch_stack
}

ensure_triton_ascend() {
  # The _ascend backend lowers kernels through the Ascend Triton fork, which
  # installs the "triton" package itself (there is no triton_ascend module);
  # a PyPI triton wheel would shadow it. Not on default PyPI.
  if python -m pip show triton-ascend >/dev/null 2>&1; then
    echo "triton-ascend already present: $(python -m pip show triton-ascend | awk '/^Version:/ {print $2}')"
    return
  fi
  python -m pip uninstall -y triton triton-ascend 2>/dev/null || true
  python -m pip install "triton-ascend==3.2.2" \
    --extra-index-url "$TRITON_ASCEND_INDEX" \
    --trusted-host triton-ascend.osinfra.cn --no-cache-dir
  python -c "import triton; print('triton', triton.__version__)"
}

install_liger_from_checkout() {
  # --no-deps: torch/torch_npu come from ensure_torch_stack and liger's
  # runtime deps are installed explicitly per profile below, so the source
  # install cannot pull a plain PyPI torch over the torch_npu build.
  python -m pip install --no-deps -e "$TARGET_ROOT"
  python - <<'PY'
import os
from pathlib import Path

import liger_kernel

loaded = Path(liger_kernel.__file__).resolve()
target = Path(os.environ["TARGET_ROOT"]).resolve()
# liger_kernel exposes no __version__ attribute (the version lives in
# pyproject.toml); identity of the loaded tree is the check that matters.
print("liger_kernel source", loaded)
if not loaded.is_relative_to(target):
    raise SystemExit(f"liger_kernel is not loaded from the tested checkout: {loaded}")
print("liger_kernel resolves to the tested release checkout")
PY
}

ms_download_models() {
  # Private download staging avoids concurrent jobs writing the same cache.
  # Retain it until the run step finishes; export only validated snapshots.
  python -m pip install -q "modelscope==1.37.0" safetensors
  TQDM_MININTERVAL="${TQDM_MININTERVAL:-15}" \
    MODELSCOPE_DOWNLOAD_PARALLELS=1 MODELSCOPE_ENABLE_DEFAULT_HASH_VALIDATION=true \
    python - "$@" <<'PY'
import json
import os
from pathlib import Path
import sys
import tempfile
import time

from modelscope import snapshot_download
from safetensors import safe_open


def validate_snapshot(local):
    root = Path(local)
    json.loads((root / 'config.json').read_text())
    shards = sorted(root.glob('*.safetensors'))
    index = root / 'model.safetensors.index.json'
    if index.is_file():
        expected = set(json.loads(index.read_text())['weight_map'].values())
        if not expected.issubset({shard.name for shard in shards}):
            raise RuntimeError(f'model weight shards missing: {root}')
    for shard in shards:
        with safe_open(str(shard), framework='np') as weights:
            if not list(weights.keys()):
                raise RuntimeError(f'empty model weights: {shard}')
    if not shards and not list(root.glob('pytorch_model*.bin')):
        raise RuntimeError(f'model weights missing: {root}')
    print(f'validated model snapshot: {root}', flush=True)


cache = tempfile.mkdtemp(prefix='liger-modelscope-', dir='/tmp')
print('private ModelScope cache:', cache, flush=True)
for pair in sys.argv[1:]:
    env_name, model_id = pair.split("=", 1)
    for attempt in range(3):
        try:
            local = snapshot_download(model_id, cache_dir=cache,
                                      enable_file_lock=True, max_workers=2)
            validate_snapshot(local)
            break
        except Exception as exc:
            print(f'{model_id} attempt {attempt + 1}/3 failed: {exc}', flush=True)
            if attempt == 2:
                raise
            cache = tempfile.mkdtemp(prefix='liger-modelscope-retry-', dir='/tmp')
            time.sleep(2 * (attempt + 1))
    with open(os.environ["GITHUB_ENV"], "a") as handle:
        handle.write(f"{env_name}={local}\n")
    print(f"{env_name}={local}", flush=True)
PY
}

stage_fixtures() {
  # CI fixtures live in the workflows checkout; copy them under TARGET_ROOT
  # so the examples' local-path arguments resolve inside one tree.
  local src="${FIXTURE_DIR:?FIXTURE_DIR is required}"
  local dst="$TARGET_ROOT/fixtures"
  mkdir -p "$dst"
  cp "$src"/*.json "$src"/*.jsonl "$dst/"
  ls -la "$dst/"
}

# Shared HF stack for every example that trains through transformers/trl.
# transformers 5.3.0 is the 5.x pairing Huawei's Ascend-CI validates on torch
# 2.9.0 (v0.8.5's monkey_patch refuses anything below 5.0.0); trl 0.12.1 is
# the last release whose SFTTrainer/ORPOTrainer still accept the examples'
# max_seq_length=/tokenizer= kwargs.
install_hf_stack() {
  python -m pip install "transformers==5.3.0" "trl==0.12.1" \
    "datasets>=3.0.0" "accelerate>=1.1" "sentencepiece" "pillow"
}

# ----- profile: liger_hf_trainer (examples/huggingface/training.py) -----
# AutoLigerKernelForCausalLM monkey-patches Qwen2 with Liger kernels and
# trains through trl's SFTTrainer. The example itself is unchanged; only
# the model/dataset/step flags are overridden.
setup_liger_hf_trainer() {
  install_hf_stack
  verify_data_stack
  stage_fixtures
  # load_dataset(path) needs a DIRECTORY containing a train split file; a
  # bare .jsonl path raises FileNotFoundError. The staging dir gives that.
  mkdir -p "$TARGET_ROOT/fixtures/ci_alpaca_8"
  cp "$TARGET_ROOT/fixtures/ci_alpaca_8.jsonl" "$TARGET_ROOT/fixtures/ci_alpaca_8/train.jsonl"
  ms_download_models "LIGER_MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct"
  printf 'LIGER_DATASET_PATH=%s\n' "$TARGET_ROOT/fixtures/ci_alpaca_8" >> "$GITHUB_ENV"
}

# ----- profile: liger_hf_fsdp (examples/huggingface/run_qwen.sh) -----
# Use the same model and fixture as the single-card SFT job. The run script
# reproduces the upstream launcher's FSDP recipe on two NPU devices.
setup_liger_hf_fsdp() {
  setup_liger_hf_trainer
  python - <<'PY'
import torch
import torch_npu

count = torch.npu.device_count()
if count < 2:
    raise SystemExit(f"run_qwen.sh FSDP requires 2 NPU devices, found {count}")
print(f"FSDP runner has {count} NPU devices")
PY
}

# ----- profile: liger_medusa (examples/medusa/train.py) -----
# Medusa multi-head retraining on a frozen backbone; trains the heads with
# Liger's fused_linear_cross_entropy. Needs scikit-learn (train_test_split)
# and safetensors; the wrapper's nvidia-smi launcher is not used.
setup_liger_medusa() {
  # Same stack as the HF profile. The entry itself stays unsupported upstream:
  # medusa's train.py calls Trainer(tokenizer=...), removed in transformers 5.0,
  # so this profile is only kept for a future upstream fix.
  install_hf_stack
  python -m pip install "scikit-learn" "safetensors"
  python -c "import transformers, sklearn, safetensors; print('transformers', transformers.__version__)"
  verify_data_stack
  stage_fixtures
  ms_download_models "LIGER_MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct"
  printf 'LIGER_FIXTURE_JSON=%s\n' "$TARGET_ROOT/fixtures/ci_sharegpt_8.json" >> "$GITHUB_ENV"
}

# ----- profile: liger_multimodal (examples/huggingface/training_multimodal.py) -----
# Qwen2-VL SFT on image-text data: monkey-patches Qwen2-VL with Liger's
# multimodal RoPE + RMSNorm + SwiGLU + FLCE, trained through trl SFTTrainer.
# The upstream script wants the_cauldron (168 GB on ModelScope), so the CI
# fixture is a 4-row local directory reproducing the ai2d schema exactly
# (images: Sequence(Image()), texts: a one-element list holding a dict) plus
# a dataset card declaring the config name - without the card
# load_dataset(dir, "ai2d") raises "BuilderConfig 'ai2d' not found".
setup_liger_multimodal() {
  install_hf_stack
  verify_data_stack
  # AutoProcessor for Qwen2-VL pulls in the torchvision image backend, which
  # the plain text stack does not carry. 0.24.0 matches torch 2.9.0.
  python -m pip install "torchvision==0.24.0"
  python -c "import transformers, trl, torchvision; print('transformers', transformers.__version__, 'trl', trl.__version__, 'torchvision', torchvision.__version__)"
  ms_download_models "LIGER_VL_MODEL_PATH=Qwen/Qwen2-VL-2B-Instruct"
  python - <<'PY'
import os
from pathlib import Path

from datasets import Dataset
from datasets import Features
from datasets import Image as ImageFeature
from datasets import Sequence
from datasets import Value
from PIL import Image

root = Path(os.environ["TARGET_ROOT"]) / "fixtures" / "cauldron_ai2d"
config = root / "ai2d"
config.mkdir(parents=True, exist_ok=True)

colors = [(220, 60, 60), (60, 200, 90), (70, 120, 230), (230, 200, 60)]
rows = []
for index, color in enumerate(colors):
    image = Image.new("RGB", (112, 112), color)
    rows.append({
        "images": [image],
        "texts": [{
            "user": f"Describe image {index}.",
            "assistant": f"This is a solid colour image {index}.",
            "source": "ci",
        }],
    })

# the_cauldron declares texts as a dict schema; a Sequence-of-dict fails to
# encode ("'list' object has no attribute 'get'"), so the field is declared
# as the one-element list schema the loader actually produces.
features = Features({
    "images": Sequence(ImageFeature()),
    "texts": [{"user": Value("string"), "assistant": Value("string"), "source": Value("string")}],
})
Dataset.from_list(rows, features=features).to_parquet(str(config / "train.parquet"))

(root / "README.md").write_text(
    "---\n"
    "configs:\n"
    "  - config_name: ai2d\n"
    "    data_files:\n"
    "      - split: train\n"
    "        path: ai2d/train.parquet\n"
    "---\n\n"
    "# Liger-Kernel CI image-text fixture\n\n"
    "Four synthetic 112x112 solid-colour images paired with a one-turn\n"
    "user/assistant exchange, shaped like HuggingFaceM4/the_cauldron ai2d\n"
    "so the upstream multimodal example runs unchanged offline.\n",
    encoding="utf-8",
)
print(f"image-text fixture ready: {root} ({len(rows)} rows)")
PY
  printf 'LIGER_VL_DATASET_PATH=%s\n' "$TARGET_ROOT/fixtures/cauldron_ai2d" >> "$GITHUB_ENV"
}

# ----- profile: liger_orpo (examples/alignment/run_orpo.py) -----
# ORPO through LigerORPOTrainer (chunked ORPO loss). The example hardcodes its
# model/data/steps, so it always runs at upstream scale; it also hardcodes the
# gated repo id meta-llama/Llama-3.2-1B-Instruct, so the same snapshot is
# mirrored from ModelScope into the HuggingFace hub cache and the run step
# resolves model + preference dataset from cache while offline.
setup_liger_orpo() {
  install_hf_stack
  python -m pip install -q "modelscope==1.37.0"
  python - <<'PY'
import os
import shutil
from pathlib import Path

from modelscope import snapshot_download

repo = "meta-llama/Llama-3.2-1B-Instruct"
ms_id = "LLM-Research/Llama-3.2-1B-Instruct"
source = Path(snapshot_download(ms_id, cache_dir="/tmp/liger-modelscope-orpo"))

hub = Path(os.environ.get("HF_HUB_CACHE") or Path.home() / ".cache" / "huggingface" / "hub")
org, name = repo.split("/", 1)
commit = "0" * 40
root = hub / f"models--{org}--{name}"
snapshot = root / "snapshots" / commit
snapshot.mkdir(parents=True, exist_ok=True)
copied = 0
for path in sorted(source.rglob("*")):
    if not path.is_file() or path.name.startswith("."):
        continue
    target = snapshot / path.relative_to(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        continue
    try:
        os.link(path, target)
    except OSError:
        shutil.copy2(path, target)
    copied += 1
(root / "refs").mkdir(parents=True, exist_ok=True)
(root / "refs" / "main").write_text(commit)
print(f"staged {repo} from ModelScope {ms_id}: {copied} files -> {snapshot}")
PY
  # The run step is offline, so the preference dataset must already be cached.
  python - <<'PY'
from datasets import load_dataset

rows = load_dataset("trl-lib/tldr-preference", split="train")
print("cached trl-lib/tldr-preference rows:", len(rows))
PY
  python -c "import transformers, trl; print('transformers', transformers.__version__, 'trl', trl.__version__)"
}

# ----- profile: liger_launcher (examples/huggingface/run_*.sh) -----
# The upstream launcher scripts build their own torchrun command and pick
# their own 7-8B models/datasets; they take no CLI overrides, so they run at
# upstream scale. torchvision is only needed by the Qwen2-VL launcher.
setup_liger_launcher() {
  install_hf_stack
  python -m pip install "torchvision==0.24.0"
  python -c "import transformers, trl, torchvision; print('transformers', transformers.__version__, 'trl', trl.__version__, 'torchvision', torchvision.__version__)"
}

verify_data_stack() {
  python - <<'PY'
import tempfile
from pathlib import Path

import numpy
import pandas
import pyarrow
import datasets
from datasets import Dataset, load_dataset

print('data stack:', 'numpy', numpy.__version__, 'pandas', pandas.__version__,
      'pyarrow', pyarrow.__version__, 'datasets', datasets.__version__)
with tempfile.TemporaryDirectory(prefix='liger-data-probe-') as work:
    path = Path(work) / 'train.parquet'
    Dataset.from_list([{'text': 'NPU CI dependency probe'}]).to_parquet(str(path))
    restored = load_dataset('parquet', data_files={'train': str(path)}, split='train')
    if restored[0]['text'] != 'NPU CI dependency probe':
        raise SystemExit('data stack parquet round-trip failed')
PY
}

supported_profiles() {
  declare -F | awk '/^declare -f setup_/ { sub(/^declare -f setup_/, ""); print }' | paste -sd' ' -
}

if ! declare -F "setup_${PROFILE}" >/dev/null 2>&1; then
  echo "unknown profile: ${PROFILE} (supported: $(supported_profiles))" >&2
  exit 1
fi

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
GITHUB_WORKSPACE="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
GITHUB_ENV="${GITHUB_ENV:?GITHUB_ENV is required}"
FIXTURE_DIR="${FIXTURE_DIR:-$GITHUB_WORKSPACE/workflows/examples-guards/liger-kernel/fixtures}"
export FIXTURE_DIR

source /usr/local/Ascend/ascend-toolkit/set_env.sh

select_pip_index
python -m pip install -U pip setuptools wheel
ensure_torch_stack
ensure_triton_ascend
install_liger_from_checkout

"setup_${PROFILE}"
