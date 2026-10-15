#!/bin/bash
# setup_example.sh - Prepare environment for cache-dit example based on profile
# Called from the examples engine's run-example job.
# Positional argument: $1 is the manifest profile (e.g., flux).

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <profile>" >&2
    exit 2
fi

PROFILE="$1"

# wan2.2 dropped: its upstream scripts moved to unsupported (broken
# `from utils import ...` imports, see the manifest); keeping the
# profile would silently run the flux setup for it.
supported_profiles="flux"

if echo "$supported_profiles" | grep -qw "$PROFILE"; then
    echo "Profile '$PROFILE' is supported"
else
    echo "Unsupported profile: $PROFILE"
    echo "Supported profiles: $supported_profiles"
    exit 1
fi

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
GITHUB_WORKSPACE="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
GITHUB_ENV="${GITHUB_ENV:?GITHUB_ENV is required}"

HERE=$(cd "$(dirname "$0")" && pwd)

# Install cache-dit and dependencies
echo "Installing cache-dit and dependencies..."

# CANN 9.1.0 base image ships without torch/torch_npu; install the
# official pairing (same torch line as the diffusers project).
ASCEND_PIP_INDEX=https://repo.huaweicloud.com/ascend/repos/pypi

pip_ascend() {
  python3 -m pip install --extra-index-url "$ASCEND_PIP_INDEX" "$@"
}

ensure_torch_stack() {
  if python3 -c "
import torch, torch_npu
raise SystemExit(
    0 if torch.__version__.startswith('2.9.0')
    and torch_npu.__version__.startswith('2.9.0') else 1)
"; then
    echo "reusing image torch stack ($(python3 -c 'import torch; print(torch.__version__)'))"
    return
  fi
  echo "installing torch==2.9.0 torch_npu==2.9.0.post2"
  pip_ascend torch==2.9.0 torch_npu==2.9.0.post2
}

ensure_torch_stack
pip3 install --no-deps torchvision==0.24.0  # torch 2.9.0 pairing

pip3 install -U cache-dit
pip3 install einops sentencepiece accelerate

# Install diffusers for parallel support
pip3 install -U diffusers  # 要求 >= 0.36.0（PyPI latest，避免走 github 代理）

# hub_cache.py's fallback source, installed here because hub_cache.py
# imports it at module load (after diffusers so the latter's
# huggingface_hub pin stands). FLUX.1-dev is an HF gated repo
# (gated: auto) and this repo carries no HF_TOKEN, so the HuggingFace
# leg answers 401 and the ModelScope mirror delivers the snapshot.
# Pinned to the version verified in this image (1.40.x fails to import
# next to huggingface_hub<1.0).
pip3 install "modelscope==1.37.0"

# Keep the engine-declared NPU visibility (matrix npu_devices exported
# via GITHUB_ENV); default to 0 only when running outside the engine.
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# Source CANN environment if available
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

# Resolve the example's model into GITHUB_ENV for the manifest's
# overlay_args expansion (${FLUX_MODEL_PATH} in the run-example step).
# scripts/hub_cache.py owns the whole path: a snapshot in the shared
# runner cache that holds every required file is used as is, with zero
# network; a miss is downloaded (HuggingFace first, ModelScope
# fallback) into this job's private staging dir on the same persistent
# volume and published into the cache, so every later run - on any pod
# of the pool - hits it. The fetch is trimmed by allow_patterns to the
# diffusers component tree (57.89 GB -> 33.74 GB). Same helper shape as
# the peft project.
resolve_hub_paths() {
  python3 "$HERE/hub_cache.py" "$@"
}

setup_flux() {
  echo "profile=flux: resolving FLUX.1-dev model"
  # HF id is the gated black-forest-labs repo; hub_cache.py trims the
  # download to the diffusers component tree FluxPipeline.from_pretrained
  # reads and falls back to the ModelScope mirror (same layout) when the
  # HuggingFace leg 401s for lack of an HF_TOKEN.
  resolve_hub_paths FLUX_MODEL_PATH
}

setup_flux

echo "Environment setup complete for profile: $PROFILE"
