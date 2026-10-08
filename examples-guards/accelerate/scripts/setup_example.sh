#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"

CLUSTER_PIP_HOST=cache-service.nginx-pypi-cache.svc.cluster.local
export CLUSTER_PIP_INDEX="http://${CLUSTER_PIP_HOST}/pypi/simple"
ASCEND_PIP_INDEX=https://repo.huaweicloud.com/ascend/repos/pypi
ALIYUN_PIP_INDEX=https://mirrors.aliyun.com/pypi/simple/

pip_ascend() {
  python -m pip install --extra-index-url "$ASCEND_PIP_INDEX" "$@"
}

select_pip_index() {
  # Runners live in mainland China: prefer the cluster pip cache, fall
  # back to the Aliyun mirror. The ascend index stays available via
  # PIP_EXTRA_INDEX_URL (set by the engine) for torch_npu wheels.
  if python -c "
import os
import urllib.error
import urllib.request
try:
    urllib.request.urlopen(os.environ['CLUSTER_PIP_INDEX'], timeout=3)
except urllib.error.HTTPError:
    pass
" 2>/dev/null; then
    export PIP_INDEX_URL="$CLUSTER_PIP_INDEX"
    export PIP_TRUSTED_HOST="$CLUSTER_PIP_HOST"
  else
    export PIP_INDEX_URL="$ALIYUN_PIP_INDEX"
    unset PIP_TRUSTED_HOST
  fi
  echo "pip index: $PIP_INDEX_URL"
}

ensure_torch_stack() {
  # Same torch line as accelerate quick-start (CANN 9.1.0 pairing):
  # torch 2.9.0 + torch_npu 2.9.0.post2. Reuse the image stack when it
  # already matches, otherwise install via the cluster cache + ascend
  # dual-source (peft's proven mechanism).
  if python -c "
import torch, torch_npu
raise SystemExit(
    0 if torch.__version__.startswith('2.9.0')
    and torch_npu.__version__.startswith('2.9.0') else 1)
"; then
    echo "reusing image torch stack (" \
      "$(python -c 'import torch; print(torch.__version__)'))"
    return
  fi
  echo "installing torch==2.9.0 torch_npu==2.9.0.post2"
  pip_ascend torch==2.9.0 torch_npu==2.9.0.post2
}

# Smoke-validate the planted NLP assets by actually loading them (all
# cache hits, no network): bert-base-cased via the bare hardcoded id,
# then MRPC through datasets. This is the pre-2026-09-16 behavior,
# now backed by the ModelScope plant instead of hf-mirror downloads.
validate_nlp_assets() {
  python - <<'PY'
import os
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained("bert-base-cased")
print("bert-base-cased tokenizer OK, vocab_size:", tok.vocab_size)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
print("bert-base-cased model OK, params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
from datasets import load_dataset
ds = load_dataset("nyu-mll/glue", "mrpc")
print("mrpc splits:", {k: len(v) for k, v in ds.items()})
PY
}

# Smoke-validate the SmolLM + wikitext plants (cache hits only).
validate_ar_assets() {
  python - <<'PY'
import os
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-360M")
AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-360M")
from datasets import load_dataset
ds = load_dataset("Salesforce/wikitext", "wikitext-2-v1")
print("smollm OK, wikitext-2 splits:", {k: len(v) for k, v in ds.items()})
PY
}

# Smoke-validate the planted inference models: resolve the snapshot via
# refs/main and check the weight files exist. A full from_pretrained
# here would just duplicate what the example does.
# NOTE: deliberately NOT snapshot_download(local_files_only=True) —
# huggingface_hub 1.x caches the full HF repo listing (trees/<sha>.json,
# written by any earlier networked call, e.g. from_pretrained during a
# previous leg's run step) and then *requires* every listed file to
# exist, including .gitattributes — which ModelScope snapshots never
# contain (verified), so the check would detonate on any warm runner
# (llava, run 35188975420). The plant's contract is "from_pretrained
# finds its files", not "complete vs the HF listing"; from_pretrained
# does per-file lookups and never needs .gitattributes.
validate_infer_assets() {
  # $1 = infer group name
  python - "$1" <<'PY'
import os
import sys
from pathlib import Path

CHECKS = {
    "infer-phi2": ("microsoft/phi-2",
                   ["model-00001-of-00002.safetensors",
                    "model-00002-of-00002.safetensors"]),
    "infer-sd": ("stable-diffusion-v1-5/stable-diffusion-v1-5",
                 ["model_index.json",
                  "unet/diffusion_pytorch_model.safetensors",
                  "vae/diffusion_pytorch_model.safetensors",
                  "text_encoder/model.safetensors",
                  "safety_checker/model.safetensors"]),
    "infer-tts": ("facebook/mms-tts-eng",
                  ["model.safetensors", "vocab.json"]),
    "infer-llava": ("llava-hf/LLaVA-NeXT-Video-7B-hf",
                    [f"model-0000{i}-of-00003.safetensors" for i in (1, 2, 3)]),
}
repo, must_have = CHECKS[sys.argv[1]]
hub_root = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))) / "hub"
repo_dir = f"models--{repo.replace('/', '--')}"
sha = (hub_root / repo_dir / "refs" / "main").read_text().strip()
snap = hub_root / repo_dir / "snapshots" / sha
missing = [f for f in must_have if not (snap / f).is_file()]
if missing:
    raise SystemExit(f"{repo}: seeded snapshot missing {missing}")
print(f"{repo} seeded snapshot OK: {snap}")

# tts / llava additionally need their datasets, which are NOT on
# ModelScope — delivered as repo bundles by the cache-seed workflow
# (spec + history: cache-seed/README.md). Warn only: the example will
# fail loudly on the Xet flake if they are absent.
DATASETS = {
    "infer-tts": ["datasets--svjack--pokemon-blip-captions-en-zh"],
    "infer-llava": ["datasets--malterei--LLaVA-Video-small-swift"],
}
for repo_dir in DATASETS.get(sys.argv[1], []):
    refs = hub_root / repo_dir / "refs" / "main"
    if not refs.is_file():
        print(f"WARN: {repo_dir} missing from shared cache root — "
              f"dispatch the cache-seed workflow (see cache-seed/README.md)",
              flush=True)
    else:
        snap = hub_root / repo_dir / "snapshots" / refs.read_text().strip()
        n = sum(1 for p in snap.rglob("*") if p.is_file())
        print(f"seeded {repo_dir} ({n} files)", flush=True)
PY
}

# Materialize the Oxford-IIT Pet Dataset jpg files used by cv_example.py
# + complete_cv_example.py. cv_example.py uses os.listdir(data_dir) +
# ".jpg" filter, so the data_dir must contain the .jpg files directly.
# The parquet now comes from the planted shared cache; the
# materialization loop itself is unchanged.
prepare_pets_data() {
  local dst="$TARGET_ROOT/fixtures/pets/images"
  if [[ -d "$dst" ]] \
     && [[ "$(ls -A "$dst" 2>/dev/null | wc -l)" -gt 1000 ]]; then
    echo "reusing pets dataset at $dst ($(ls "$dst" | wc -l) files)"
    return
  fi
  echo "materializing Oxford-IIT Pets from planted cache to $dst"
  mkdir -p "$dst"
  python - <<PY
import os
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
from datasets import load_dataset
DST = "$dst"
ds = load_dataset("timm/oxford-iiit-pet")
n = 0
for split in ("train", "test"):
    for r in ds[split]:
        path = os.path.join(DST, r["image_id"] + ".jpg")
        if not os.path.exists(path):
            img = r["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(path, "JPEG")
        n += 1
print(f"wrote {n} images to {DST}")
PY
  echo "pets dataset ready: $(ls "$dst" | wc -l) files"
}

setup_accelerate-nlp() {
  echo "installing accelerate from $TARGET_ROOT"
  # Pin torch explicitly. accelerate's setup.py requires `torchpippy>=0.2.0`,
  # which transitively pulls nvidia-cu13 metapackages and the resolver
  # then upgrades torch to 2.14.0+cu130 — breaking torch_npu 2.9.0 ABI.
  # `pip install -e` re-resolves all deps, so repeat the torch pin here
  # in the same command (constraints-npu.txt is already exported).
  python -m pip install -e "$TARGET_ROOT" "torch==2.9.0" "torch_npu==2.9.0.post2"
  # scikit-learn is needed by the `evaluate` library's glue metric (sklearn's
  # f1_score, matthews_corrcoef); not a direct dep of evaluate or transformers
  # so it must be listed explicitly. schedulefree is a pure-Python wheel
  # needed only by by_feature/schedule_free.py (tiny, kept in the base list).
  python -m pip install \
    transformers datasets evaluate safetensors scikit-learn schedulefree \
    "torch==2.9.0"
  python -c "
import torch, torch_npu
assert torch.__version__.startswith('2.9.0'), \
    f'torch drifted to {torch.__version__}'
import accelerate, transformers, datasets, evaluate, sklearn, schedulefree
print('accelerate', accelerate.__version__,
      '/ transformers', transformers.__version__,
      '/ datasets', datasets.__version__,
      '/ torch', torch.__version__,
      '/ sklearn', sklearn.__version__)
"
  validate_nlp_assets
}

setup_accelerate-nlp-ar() {
  # Autoregressive grad-accum variant = NLP base + SmolLM/wikitext.
  setup_accelerate-nlp
  validate_ar_assets
}

setup_accelerate-ds() {
  # DeepSpeed config-support profile = NLP base + upstream deepspeed.
  # 2026-09-18 coder npu-1 (2x910B4) verification: the plain PyPI
  # deepspeed wheel (>=0.18.2) ships its own NPU accelerator —
  # deepspeed.accelerator.get_accelerator() auto-detects torch_npu and
  # resolves to npu/hccl, no Ascend fork needed. ZeRO-2 bf16 2-card
  # full flow (train/eval/best-checkpoint/save_pretrained) passes.
  # torch is re-pinned in the same command: deepspeed's resolver may
  # otherwise pick a newer wheel that breaks the torch_npu 2.9.0 ABI
  # (same reason as the -e install pin in setup_accelerate-nlp).
  setup_accelerate-nlp
  python -m pip install "deepspeed>=0.18.2" "torch==2.9.0"
  python -c "
import torch, torch_npu
assert torch.__version__.startswith('2.9.0'), \
    f'torch drifted to {torch.__version__}'
import deepspeed
from deepspeed.accelerator import get_accelerator
acc = get_accelerator()
assert acc.device_name() == 'npu', \
    f'deepspeed did not detect NPU: {acc.device_name()}'
from deepspeed.runtime.sequence_parallel.ulysses_sp import (
    UlyssesSPAttentionHF,
)
print('deepspeed', deepspeed.__version__,
      '/ accelerator:', acc.device_name(),
      '/', acc.communication_backend_name(),
      '/ ulysses_sp import OK')
"
  # The DS entry reuses the SmolLM-360M + wikitext plants (no new seeds).
  validate_ar_assets
}

setup_accelerate-cv() {
  # CV profile = NLP profile + vision deps + Pets data.
  setup_accelerate-nlp
  # torchvision ≤ 0.28.0 (matches torch 2.9.0 ABI; v0.29+ requires Stable ABI
  # symbols that torch 2.9 lacks — see `torchvision-v29-stable-abi` memory).
  python -m pip install "torchvision==0.24.0" "torch==2.9.0" timm
  python -c "
import torch, torch_npu
assert torch.__version__.startswith('2.9.0'), \
    f'torch drifted to {torch.__version__}'
import timm, torchvision
print('timm', timm.__version__,
      '/ torchvision', torchvision.__version__)
"
  prepare_pets_data
}

# Shared pip stack for the four inference/distributed examples. Each
# example then gets its own model plant group so a job only pulls the
# weights it actually hardcodes (phi-2 5.5G / SD 5.5G / mms 145M /
# llava 14G — planting all four per job would move ~27G).
# - fire: speech_gen + llava CLI entry
# - av: llava video decode
# - diffusers: stable_diffusion pipeline
# - scipy: speech_gen wavfile output
# - torchvision: DiffusionPipeline / VitsModel import chains pull the
#   torchvision op registrations (same ABI pin as cv profile).
setup_infer_base() {
  setup_accelerate-nlp
  python -m pip install \
    fire av diffusers scipy "torchvision==0.24.0" "torch==2.9.0"
  python -c "
import torch, torch_npu
assert torch.__version__.startswith('2.9.0'), \
    f'torch drifted to {torch.__version__}'
import fire, av, diffusers, scipy, torchvision
print('av', av.__version__,
      '/ diffusers', diffusers.__version__,
      '/ torchvision', torchvision.__version__)
"
}

setup_accelerate-infer-phi2() {
  setup_infer_base
  validate_infer_assets infer-phi2
}

setup_accelerate-infer-sd() {
  setup_infer_base
  validate_infer_assets infer-sd
}

setup_accelerate-infer-tts() {
  setup_infer_base
  validate_infer_assets infer-tts
}

setup_accelerate-infer-llava() {
  setup_infer_base
  validate_infer_assets infer-llava
}

supported_profiles() {
  declare -F | awk '/^declare -f setup_/ { sub(/^declare -f setup_/, ""); print }' \
    | paste -sd' ' -
}

if ! declare -F "setup_${PROFILE}" >/dev/null 2>&1; then
  echo "unknown profile: ${PROFILE} (supported: $(supported_profiles))" >&2
  exit 1
fi

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
GITHUB_WORKSPACE="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
GITHUB_ENV="${GITHUB_ENV:?GITHUB_ENV is required}"

# Xet-backed HF repos 302 to cas-bridge.xethub.hf.co which hf-mirror
# cannot proxy. Planted caches make weight downloads unnecessary, but
# keep the legacy transfer path forced for any residual metadata/file
# fetch (README misses on dataset resolution etc.).
echo "HF_HUB_DISABLE_XET=1" >> "$GITHUB_ENV"
export HF_HUB_DISABLE_XET=1

HERE=$(cd "$(dirname "$0")" && pwd)
export PIP_CONSTRAINT="$(cd "$HERE/.." && pwd)/constraints-npu.txt"

source /usr/local/Ascend/ascend-toolkit/set_env.sh

select_pip_index
python -m pip install -U pip setuptools wheel
ensure_torch_stack

"setup_${PROFILE}"
