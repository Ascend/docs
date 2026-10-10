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

# Keep the engine-declared NPU visibility (matrix npu_devices exported
# via GITHUB_ENV); default to 0 only when running outside the engine.
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# Source CANN environment if available
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

# Resolve one example model into $2 (shell var + GITHUB_ENV) from the
# shared runner cache. Warm path is zero-network: the HF hub-cache
# layout (refs/main = real HF sha, real files) is resolved in place.
# Cold-cache self-heal: huggingface_hub first via HF_ENDPOINT (mirror
# default). Unlike the other projects, the ModelScope fallback is kept
# on purpose: FLUX.1-dev is an HF gated repo and this repo carries no
# HF_TOKEN, so the HF fill cannot succeed for it - the fallback pulls
# the AI-ModelScope mirror into a per-run workspace dir (re-downloaded
# every run; the durable fix is a cache-seed plant or an HF_TOKEN
# secret). Same helper as the xllm project.
resolve_model() {
  local hf_id="$1" var="$2"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  local resolved
  resolved=$(python3 - "$hf_id" <<'PY' | tee /dev/stderr | sed -n "s/^RESOLVED=//p" | tail -n 1
import os
import subprocess
import sys

model_id = sys.argv[1]
hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
repo_dir = os.path.join(hf_home, "hub", "models--" + model_id.replace("/", "--"))
workspace_dir = os.path.join(
    os.environ["GITHUB_WORKSPACE"], "cache_dit_models", model_id.split("/")[-1]
)


def cached_snapshot():
    refs = os.path.join(repo_dir, "refs", "main")
    if not os.path.isfile(refs):
        return None
    with open(refs) as fh:
        sha = fh.read().strip()
    snap = os.path.join(repo_dir, "snapshots", sha)
    if os.path.isdir(snap) and os.listdir(snap):
        return snap
    return None


snapshot = cached_snapshot()
if snapshot is not None:
    print(f"{model_id}: shared-cache hit -> {snapshot}", flush=True)
else:
    print(f"{model_id}: not in the shared cache ({repo_dir}); filling it via HF",
          flush=True)
    try:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(model_id)
        print(f"filled shared cache: {snapshot}", flush=True)
    except Exception as e:
        print(f"HF fill failed ({type(e).__name__}: {e}); trying ModelScope",
              flush=True)
        snapshot = None

    if snapshot is None:
        try:
            from modelscope import snapshot_download as ms_snapshot
        except ImportError:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q",
                 "modelscope==1.37.0"],
                check=True,
            )
            from modelscope import snapshot_download as ms_snapshot
        os.makedirs(workspace_dir, exist_ok=True)
        ms_snapshot(model_id, local_dir=workspace_dir)
        snapshot = workspace_dir
        print(f"downloaded to ephemeral {snapshot}", flush=True)

print(f"RESOLVED={snapshot}", flush=True)
PY
)
  [[ -n "$resolved" ]] || { echo "resolve_model($hf_id) printed no path" >&2; exit 2; }
  # Shell var for same-script consumers; GITHUB_ENV for the run-example
  # step's overlay_args expansion (${FLUX_MODEL_PATH} in the manifest).
  export "$var"="$resolved"
  echo "$var=$resolved" >> "$GITHUB_ENV"
}

setup_flux() {
  echo "profile=flux: resolving FLUX.1-dev model"
  # HF id is the gated black-forest-labs repo; cold-cache fill falls
  # back to the AI-ModelScope mirror by resolve_model (diffusers
  # layout, same as the legacy in-setup modelscope download).
  resolve_model black-forest-labs/FLUX.1-dev FLUX_MODEL_PATH
}

setup_flux

echo "Environment setup complete for profile: $PROFILE"
