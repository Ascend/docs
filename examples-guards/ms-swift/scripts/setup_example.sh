#!/usr/bin/env bash
# Prepare the CI environment for one supported ms-swift example.
# $1 is the manifest profile. Unknown profiles fail before any install.
# ms-swift itself is installed from TARGET_ROOT (the release checkout under
# test), so the guarded tag is exactly the code that runs.
#
# The runtime image is the vLLM-Ascend 0.23.0 image: torch, torch_npu,
# vllm and vllm-ascend come preinstalled as the one compatibility set that
# vllm-ascend pins and the upstream NPU guide installs. Nothing here
# reinstalls them; constraints-npu.txt only stops later installs from
# replacing them.
#
# Profiles are environment families:
#   swift      ms-swift alone (transformers, FSDP, vllm rollouts)
#   megatron   ms-swift plus Megatron-LM, MindSpeed and mcore-bridge
set -euo pipefail

export PYTHONNOUSERSITE=1

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$HERE/.." && pwd)

# The set of upstream pins for the Megatron stack
# (docs/source_en/BestPractices/NPU-support.md): Megatron-LM v0.16.0 with
# MindSpeed core_r0.16.0. MindSpeed is pinned to the commit its release
# branch pointed at when this guard was written, so a push to the branch
# cannot change what a run installs. ms-swift 4.6.0 requires
# mcore-bridge>=1.6.3.
MEGATRON_LM_REF=core_v0.16.0
MINDSPEED_URL=https://github.com/Ascend/MindSpeed.git
MINDSPEED_COMMIT=8787b1645be11e02a8d3f285a35fdd96e6c88b87
MCORE_BRIDGE_VERSION=1.6.5

CLUSTER_PIP_HOST=cache-service.nginx-pypi-cache.svc.cluster.local
export CLUSTER_PIP_INDEX="http://${CLUSTER_PIP_HOST}/pypi/simple"
export PIP_CONSTRAINT="$ROOT_DIR/constraints-npu.txt"
# Shard-by-shard download bars flood the job log.
export HF_HUB_DISABLE_PROGRESS_BARS=1
# MindSpeed patching walks transformers' lazy modules, and each access logs a
# warning: 33,000 lines in one process (run 38866705483). Errors still print.
if [[ "$PROFILE" == megatron ]]; then
  export TRANSFORMERS_VERBOSITY=error
fi

select_pip_index() {
  # Prefer the in-cluster pip cache when it answers; otherwise leave pip on
  # its default index. The Ascend index stays available through the
  # engine's PIP_EXTRA_INDEX_URL.
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
  fi
  echo "pip index: ${PIP_INDEX_URL:-default}"
}

verify_image_stack() {
  # The image must hold the pinned stack and the NPU must answer. A
  # different image tag fails here, not halfway through a training run.
  python -c "
from importlib.metadata import version
import torch
import torch_npu
from vllm.platforms import current_platform

for name, expected in {'torch': '2.10.0', 'torch-npu': '2.10.0.post4',
                       'vllm': '0.23.0', 'vllm-ascend': '0.23.0',
                       'transformers': '5.5.4'}.items():
    actual = version(name).split('+')[0]
    print(f'image stack: {name}={actual}')
    if actual != expected:
        raise SystemExit(f'{name}: expected {expected}, got {actual}')
if current_platform.device_type != 'npu' or not torch.npu.is_available():
    raise SystemExit(f'vLLM selected {current_platform.device_type}, no usable NPU')
print('NPU devices:', torch.npu.device_count())
"
}

install_ms_swift() {
  python -m pip install --no-deps -e "$TARGET_ROOT"
  python "$HERE/install_framework_deps.py" "$TARGET_ROOT"
  python -c "
from qwen_vl_utils import fetch_image, fetch_video
import swift
print('ms-swift', swift.__version__)
"
}

resolve_model() {
  # Make the example's model and datasets available in the shared cache,
  # so no rank downloads weights or data while the run is in progress.
  python "$HERE/hub_cache.py" "$TARGET_ROOT/$EXAMPLE_PATH"
}

install_megatron_stack() {
  local deps="$GITHUB_WORKSPACE/deps"
  mkdir -p "$deps"
  git -c advice.detachedHead=false clone --depth 1 --branch "$MEGATRON_LM_REF" \
    https://github.com/NVIDIA/Megatron-LM.git "$deps/Megatron-LM"
  git init --quiet "$deps/MindSpeed"
  git -C "$deps/MindSpeed" fetch --quiet --depth 1 "$MINDSPEED_URL" "$MINDSPEED_COMMIT"
  git -C "$deps/MindSpeed" checkout --quiet FETCH_HEAD
  python -m pip install "mcore-bridge==${MCORE_BRIDGE_VERSION}"
  python -m pip install -e "$deps/MindSpeed"
  echo "MEGATRON_LM_PATH=$deps/Megatron-LM" >> "$GITHUB_ENV"
  export MEGATRON_LM_PATH="$deps/Megatron-LM"
  export PYTHONPATH="$deps/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"
  echo "PYTHONPATH=$PYTHONPATH" >> "$GITHUB_ENV"
  python -c "
import mindspeed.megatron_adaptor
from swift.megatron.init import init_megatron_env
init_megatron_env()
print('Megatron-SWIFT environment ready')
"
}

prepare_stack() {
  # The a2 runner label is a mixed pool of Atlas A2 chips; print which chip
  # and how much HBM this job got, so a failure can be read against it.
  npu-smi info || true
  select_pip_index
  verify_image_stack
  install_ms_swift
}

setup_swift() {
  prepare_stack
  resolve_model
}

setup_megatron() {
  prepare_stack
  install_megatron_stack
  resolve_model
}

supported_profiles() {
  declare -F | awk '/^declare -f setup_/ { sub(/^declare -f setup_/, ""); print }' | paste -sd' ' -
}

if ! declare -F "setup_${PROFILE}" >/dev/null 2>&1; then
  echo "unknown profile: ${PROFILE} (supported: $(supported_profiles))" >&2
  exit 1
fi

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
EXAMPLE_PATH="${EXAMPLE_PATH:?EXAMPLE_PATH is required}"
GITHUB_WORKSPACE="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
GITHUB_ENV="${GITHUB_ENV:?GITHUB_ENV is required}"

# Later steps are a new shell; do not rely on a previous workflow step.
# shellcheck disable=SC1091
source /usr/local/Ascend/ascend-toolkit/set_env.sh

"setup_${PROFILE}"
