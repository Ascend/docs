#!/usr/bin/env bash
# Prepare the CI environment for one supported bitsandbytes example.
# $1 is the manifest profile. Unknown profiles fail before any install.
set -euo pipefail

export PYTHONNOUSERSITE=1

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"

CLUSTER_PIP_HOST=cache-service.nginx-pypi-cache.svc.cluster.local
export CLUSTER_PIP_INDEX="http://${CLUSTER_PIP_HOST}/pypi/simple"
ASCEND_PIP_INDEX=https://repo.huaweicloud.com/ascend/repos/pypi
ALIYUN_PIP_INDEX=https://mirrors.aliyun.com/pypi/simple/
TRITON_ASCEND_LINKS=https://repo.huaweicloud.com/ascend/repos/pypi/triton-ascend/

pip_ascend() {
  python -m pip install --extra-index-url "$ASCEND_PIP_INDEX" "$@"
}

select_pip_index() {
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

install_cpu_torch() {
  # $1 is the torch version, without the +cpu local label.
  # The CPU wheel is only on download.pytorch.org. A plain torch== pin
  # resolves to a CUDA build whose metadata depends on cuda-toolkit,
  # which constraints-npu.txt rejects.
  local version="$1"
  local cp_abi
  cp_abi=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
  echo "installing torch==${version}+cpu"
  python -m pip install --no-deps \
    "https://download.pytorch.org/whl/cpu/torch-${version}%2Bcpu-${cp_abi}-${cp_abi}-manylinux_2_28_aarch64.whl"
  python -m pip install \
    'filelock' 'typing-extensions>=4.10.0' 'setuptools>=77,<82' \
    'sympy>=1.13.3' 'networkx>=2.5.1' 'jinja2' 'fsspec>=0.8.5' \
    numpy packaging pyyaml
}

install_bitsandbytes() {
  # Build against the torch just installed. Build isolation would
  # resolve torch>=2.4 again and can pick a CUDA wheel.
  python -m pip install 'scikit-build-core' 'setuptools>=77,<82' cmake ninja
  python -m pip install --no-build-isolation --no-deps -e "$TARGET_ROOT"
}

install_triton_ascend() {
  # Community CUDA Triton has no Ascend backend. triton-ascend provides
  # the triton module and also Requires-Dist triton==3.5.0, which would
  # replace the fork. Install the fork with --no-deps and add pybind11
  # itself.
  python -m pip uninstall -y triton >/dev/null 2>&1 || true
  python -m pip install --no-deps --force-reinstall \
    --find-links "$TRITON_ASCEND_LINKS" \
    --extra-index-url "$ASCEND_PIP_INDEX" \
    triton-ascend==3.2.2
  python -m pip install pybind11
}

emit_offline() {
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export HF_ENDPOINT=https://huggingface.co
  if [[ -n "${GITHUB_ENV:-}" ]]; then
    printf '%s\n' \
      'HF_HUB_OFFLINE=1' \
      'TRANSFORMERS_OFFLINE=1' \
      'HF_DATASETS_OFFLINE=1' \
      'HF_ENDPOINT=https://huggingface.co' \
      >> "$GITHUB_ENV"
  fi
}

prefetch() {
  python "$HERE/hub_cache.py" "$@"
}

setup_cpu() {
  # No torch.compile on this path. Upstream allows torch>=2.4,<3, and
  # the CANN 9.1.0 recommended pair is torch 2.12.0. CPU wheel only.
  install_cpu_torch 2.12.0
  install_bitsandbytes
  python -m pip install \
    transformers datasets sentencepiece protobuf huggingface_hub modelscope safetensors
  python -c "import torch; print('torch', torch.__version__); raise SystemExit(0 if torch.__version__.startswith('2.12.0') else 1)"
  python "$HERE/assert_native_lib.py"
  prefetch LLAMA_MODEL ALPACA_DATASET
  emit_offline
}

setup_npu() {
  # compile_inference.py calls torch.compile. triton-ascend 3.2.2 is the
  # release whose CANN 9.1.0 matrix includes Python 3.12, and the newest
  # torch_npu on that matrix is 2.10.0.post4. The headline CANN 9.1.0
  # recommendation is torch 2.12.0, which that triton release does not
  # list. bitsandbytes accepts torch>=2.4,<3, so 2.10.0+cpu is inside
  # the upstream floor and inside the CANN 9.1.0 table.
  install_cpu_torch 2.10.0
  echo "installing torch_npu==2.10.0.post4"
  # --no-deps: the wheel depends on torch==2.10.0, and a resolver will
  # replace the +cpu wheel with a CUDA build to satisfy that pin.
  pip_ascend --no-deps torch_npu==2.10.0.post4
  install_bitsandbytes
  python -m pip install \
    transformers accelerate huggingface_hub modelscope \
    sentencepiece protobuf safetensors
  install_triton_ascend
  python "$HERE/assert_npu_stack.py"
  python "$HERE/assert_native_lib.py"
  prefetch GEMMA_MODEL
  emit_offline
}

supported_profiles() {
  declare -F | awk '/^declare -f setup_/ { sub(/^declare -f setup_/, ""); print }' | paste -sd' ' -
}

if ! declare -F "setup_${PROFILE}" >/dev/null 2>&1; then
  echo "unknown profile: ${PROFILE} (supported: $(supported_profiles))" >&2
  exit 1
fi

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
HERE=$(cd "$(dirname "$0")" && pwd)
export PIP_CONSTRAINT="$(cd "$HERE/.." && pwd)/constraints-npu.txt"
export HF_ENDPOINT=https://huggingface.co
export PATH="/usr/local/sbin:$PATH"
# shellcheck disable=SC1091
source /usr/local/Ascend/ascend-toolkit/set_env.sh

select_pip_index
python -m pip install -U pip wheel
"setup_${PROFILE}"
