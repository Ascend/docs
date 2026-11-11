#!/usr/bin/env bash
# Build the checked-out kt-kernel with the Ascend callback path enabled.
set -eo pipefail

if [[ "${1:-}" != "npu" ]]; then
  echo "usage: $0 npu" >&2
  exit 2
fi

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
export PATH="/usr/local/sbin:$PATH"
export PYTHONNOUSERSITE=1
export PIP_CONSTRAINT="${PROJECT_ROOT}/constraints-npu.txt"

# CANN's environment script can reference unset shell variables.
# shellcheck disable=SC1091
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u

echo "ktransformers target $(git -C "$TARGET_ROOT" rev-parse HEAD)"

apt-get update -qq
apt-get install -y -qq --no-install-recommends \
  build-essential cmake git pkg-config libhwloc-dev libnuma-dev python3-dev patchelf

CLUSTER_PIP_HOST=cache-service.nginx-pypi-cache.svc.cluster.local
export CLUSTER_PIP_INDEX="http://${CLUSTER_PIP_HOST}/pypi/simple"
if python -c "
import os, urllib.error, urllib.request
try:
    urllib.request.urlopen(os.environ['CLUSTER_PIP_INDEX'], timeout=3)
except urllib.error.HTTPError:
    pass
" ; then
  export PIP_INDEX_URL="$CLUSTER_PIP_INDEX"
  export PIP_TRUSTED_HOST="$CLUSTER_PIP_HOST"
else
  export PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/"
  unset PIP_TRUSTED_HOST || true
fi
echo "pip index: $PIP_INDEX_URL"

python -m pip install -U pip setuptools wheel uv
# Same CANN 9.1 pair as the torchvision guard on this image. PyPI's
# aarch64 torch==2.14.0 wheel is the CUDA build, so the CPU wheel comes
# from the pytorch-wheels/cpu find-links index.
uv pip install --system -f https://mirrors.aliyun.com/pytorch-wheels/cpu 'torch==2.14.0+cpu'
uv pip install --system --no-deps 'torch_npu==2.14.0'

git -C "$TARGET_ROOT" submodule update --init third_party/pybind11 third_party/llama.cpp
test -f "$TARGET_ROOT/third_party/pybind11/CMakeLists.txt"
test -f "$TARGET_ROOT/third_party/llama.cpp/CMakeLists.txt"

export CPUINFER_USE_ASCEND_NPU=1
export CPUINFER_USE_CUDA=0
export CPUINFER_ENABLE_KML=OFF
export CPUINFER_ENABLE_AMX=OFF
export CPUINFER_ENABLE_BLIS=OFF
export CPUINFER_BUILD_TYPE=Release
export CPUINFER_PARALLEL=8
python -m pip install --no-build-isolation --no-deps "$TARGET_ROOT/kt-kernel"
python "$PROJECT_ROOT/scripts/check_npu_stack.py"
