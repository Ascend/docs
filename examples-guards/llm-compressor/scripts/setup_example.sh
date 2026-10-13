#!/usr/bin/env bash
# Prepare the CI environment for one supported llm-compressor example.
# $1 is the manifest profile. $2 is the example path relative to the
# target repo (the engine passes it as the EXAMPLE_PATH env instead;
# either form works). Unknown profiles fail before any install.
set -euo pipefail

export PYTHONNOUSERSITE=1

if [[ $# -lt 1 ]] || [[ $# -gt 2 ]]; then
  echo "usage: $0 <profile> [example-relpath]" >&2
  exit 2
fi

PROFILE="$1"
EXAMPLE_REL="${2:-${EXAMPLE_PATH:?EXAMPLE_PATH env or argument 2 is required}}"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

ASCEND_PIP_INDEX=https://repo.huaweicloud.com/ascend/repos/pypi
ALIYUN_PIP_INDEX=https://mirrors.aliyun.com/pypi/simple/
CLUSTER_PIP_HOST=cache-service.nginx-pypi-cache.svc.cluster.local
export CLUSTER_PIP_INDEX="http://${CLUSTER_PIP_HOST}/pypi/simple"
export PIP_CONSTRAINT="$ROOT_DIR/constraints-npu.txt"

pip_locked() {
  python -m pip install \
    --extra-index-url "$ASCEND_PIP_INDEX" \
    -c "$PIP_CONSTRAINT" \
    "$@"
}

select_pip_index() {
  # Prefer the in-cluster pip cache when it answers. Otherwise Aliyun.
  # torch_npu wheels come from the Ascend index via --extra-index-url
  # and the engine's PIP_EXTRA_INDEX_URL.
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

ensure_torch_npu_stack() {
  # CANN 9.1.0 image paired with torch 2.12.0 + torch_npu 2.12.0.
  # llm-compressor's release floor is torch>=2.10.0,<=2.14.0, so 2.12
  # sits inside it. The CPU wheel is only on download.pytorch.org.
  if python -c "
import torch, torch_npu
print('found torch', torch.__version__, 'torch_npu', torch_npu.__version__)
raise SystemExit(
    0 if torch.__version__.startswith('2.12.0')
    and torch_npu.__version__.startswith('2.12.0')
    else 1)
"; then
    echo "reusing torch 2.12 / torch_npu 2.12 stack"
    return
  fi
  echo "installing torch==2.12.0+cpu (direct URL) + torch_npu==2.12.0"
  CP_ABI=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
  python -m pip install --no-deps \
    "https://download.pytorch.org/whl/cpu/torch-2.12.0%2Bcpu-${CP_ABI}-${CP_ABI}-manylinux_2_28_aarch64.whl"
  python -m pip install -c "$PIP_CONSTRAINT" \
    'filelock' 'typing-extensions>=4.10.0' 'setuptools<82' \
    'sympy>=1.13.3' 'networkx>=2.5.1' 'jinja2' 'fsspec>=0.8.5' \
    numpy pyyaml
  pip_locked torch_npu==2.12.0
}

install_llmcompressor() {
  # modelscope pinned to the last pre-hub-split line: >=1.38 depends on
  # the separate modelscope-hub package, and 1.40.1's loose ">=0.4.2"
  # floor breaks import when the pip mirror lags on hub 0.4.3. The
  # plant path (modelscope.hub.snapshot_download) exists in 1.37.0.
  # datasets stays at llm-compressor's own floor (>=4.8.4). Do not
  # downgrade it; audio examples that need torchcodec are unsupported.
  pip_locked "modelscope==1.37.0" huggingface_hub safetensors
  # setup.py defaults BUILD_TYPE to dev and then drops the release
  # upper bounds. The engine checks out a release tag, so install
  # that tag's capped set. Dev resolution on the first PR run pulled
  # transformers 5.19, datasets 5.1 and auto-round 0.16.
  BUILD_TYPE=release pip_locked -e "$TARGET_ROOT"
}

# from_pretrained on a planted id still asks the Hub for the real commit
# unless offline. Call this only after every prefetch that needs network.
emit_offline_hub() {
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  if [[ -n "${GITHUB_ENV:-}" ]]; then
    {
      echo 'HF_HUB_OFFLINE=1'
      echo 'TRANSFORMERS_OFFLINE=1'
    } >> "$GITHUB_ENV"
  fi
}

install_npu_stack() {
  export PATH="/usr/local/sbin:$PATH"
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  select_pip_index
  python -m pip install -U pip 'setuptools<82' wheel
  ensure_torch_npu_stack
  python -c "
import torch, torch_npu
assert torch.npu.is_available()
print(torch.__version__, torch_npu.__version__, torch.npu.device_count())
"
  install_llmcompressor
  python -c "
import llmcompressor
import torch
import torch_npu
assert torch.__version__.startswith('2.12.0'), torch.__version__
assert torch_npu.__version__.startswith('2.12.0'), torch_npu.__version__
assert torch.npu.is_available()
print('llmcompressor', llmcompressor.__version__)
"
}

install_cpu_torchvision() {
  # transformers 5 builds its vision and video processors on torchvision.
  # Like torch, the PyPI linux wheel links CUDA, and the +cpu wheel that
  # pairs with torch 2.12.0 is only on download.pytorch.org.
  local cp_abi
  cp_abi=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
  python -m pip install --no-deps \
    "https://download.pytorch.org/whl/cpu/torchvision-0.27.0%2Bcpu-${cp_abi}-${cp_abi}-manylinux_2_28_aarch64.whl"
  python -c "
import torch
assert torch.__version__.startswith('2.12.0'), torch.__version__
import torchvision
print('torchvision ok', torchvision.__version__, torch.__version__)
"
}

install_example_extras() {
  local example="$TARGET_ROOT/$EXAMPLE_REL"
  if grep -qE 'AutoProcessor|AutoImageProcessor|AutoVideoProcessor' "$example"; then
    echo "installing torchvision for ${EXAMPLE_REL}"
    install_cpu_torchvision
  fi
  if grep -q 'qwen_vl_utils' "$example"; then
    echo "installing qwen_vl_utils for ${EXAMPLE_REL}"
    pip_locked qwen_vl_utils
    python -c "
import torch
assert torch.__version__.startswith('2.12.0'), torch.__version__
import qwen_vl_utils
print('qwen_vl_utils ok', torch.__version__)
"
  fi
}

prefetch_for_example() {
  python "$SCRIPT_DIR/prefetch_example_assets.py" "$TARGET_ROOT/$EXAMPLE_REL"
  emit_offline_hub
}

setup_npu_oneshot() {
  install_npu_stack
  install_example_extras
  prefetch_for_example
}

setup_npu_ddp() {
  setup_npu_oneshot
}

supported_profiles() {
  declare -F | awk '/^declare -f setup_/ { sub(/^declare -f setup_/, ""); print }' | paste -sd' ' -
}

if ! declare -F "setup_${PROFILE}" >/dev/null 2>&1; then
  echo "unknown profile: ${PROFILE} (supported: $(supported_profiles))" >&2
  exit 1
fi

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"

"setup_${PROFILE}"
