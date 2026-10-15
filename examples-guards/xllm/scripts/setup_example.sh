#!/usr/bin/env bash
# Prepare the CI environment for xllm examples on NPU.
# $1 is the manifest profile. Unknown profiles fail before any install.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"

ASCEND_PIP_INDEX=https://repo.huaweicloud.com/ascend/repos/pypi
CLUSTER_PIP_HOST=cache-service.nginx-pypi-cache.svc.cluster.local
export CLUSTER_PIP_INDEX="http://${CLUSTER_PIP_HOST}/pypi/simple"
FALLBACK_PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

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
    export PIP_INDEX_URL="$FALLBACK_PIP_INDEX"
    unset PIP_TRUSTED_HOST
  fi
  echo "pip index: $PIP_INDEX_URL"
}

ensure_torch_stack() {
  # The CANN base image usually ships torch/torch_npu; verify the versions
  # match what xllm expects (2.9.0 / 2.9.0.post2) and reinstall if not.
  if python -c "
import torch, torch_npu
print('found torch', torch.__version__, 'torch_npu', torch_npu.__version__)
raise SystemExit(0 if torch.__version__.startswith('2.9.0') and torch_npu.__version__.startswith('2.9.0') else 1)
"; then
    echo "reusing image torch stack"
    return
  fi
  echo "installing torch==2.9.0 torch_npu==2.9.0.post2"
  pip_ascend torch==2.9.0 torch_npu==2.9.0.post2
}

supported_profiles() {
  declare -F | awk '/^declare -f setup_/ { sub(/^declare -f setup_/, ""); print }' | paste -sd' ' -
}

# Resolve one example model into $2 (shell var + GITHUB_ENV) from the
# shared runner cache. Warm path is zero-network: the HF hub-cache
# layout (refs/main = real HF sha, real files) is resolved in place.
# Cold-cache self-heal (first run or a model added since): pull once
# with huggingface_hub via HF_ENDPOINT (defaults to the hf-mirror
# mirror) - no local_dir, so it lands back in the shared hub-cache
# layout and every later run is a hit. No ModelScope fallback: HF is
# the single download channel; a failure here fails the setup loudly
# so the cache gap stays visible.
# Replaces the legacy /data/ci-cache/modelscope bind-mount +
# in-setup modelscope download (the engine mounts no project volumes).
resolve_model() {
  local hf_id="$1" var="$2"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  local resolved
  resolved=$(python3 - "$hf_id" <<'PY' | tee /dev/stderr | sed -n "s/^RESOLVED=//p" | tail -n 1
import os
import sys

model_id = sys.argv[1]
hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
repo_dir = os.path.join(hf_home, "hub", "models--" + model_id.replace("/", "--"))


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
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(model_id)
    print(f"filled shared cache: {snapshot}", flush=True)

print(f"RESOLVED={snapshot}", flush=True)
PY
)
  [[ -n "$resolved" ]] || { echo "resolve_model($hf_id) printed no path" >&2; exit 2; }
  # Shell var for same-script consumers; GITHUB_ENV for the run-example
  # step's overlay_args expansion (${XLLM_MODEL_PATH} in the manifest).
  export "$var"="$resolved"
  echo "$var=$resolved" >> "$GITHUB_ENV"
}

# Default profile: resolve the LLM example model (xllm is pre-installed in
# the official release image, no source build needed).
setup_default() {
  echo "profile=default: resolving example model"
  resolve_model Qwen/Qwen2-7B-Instruct XLLM_MODEL_PATH
}

if ! declare -F "setup_${PROFILE}" >/dev/null 2>&1; then
  echo "unknown profile: ${PROFILE} (supported: $(supported_profiles))" >&2
  exit 1
fi

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
GITHUB_WORKSPACE="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
GITHUB_ENV="${GITHUB_ENV:?GITHUB_ENV is required}"

# Later steps are a new shell; do not rely on a previous workflow step.
source /usr/local/Ascend/ascend-toolkit/set_env.sh

select_pip_index
python -m pip install -U pip
ensure_torch_stack

# Verify NPU is available
python -c "import torch, torch_npu; print('torch:', torch.__version__, 'torch_npu:', torch_npu.__version__, 'npu_count:', torch.npu.device_count())"
npu-smi info

# Verify xllm from the official release image (pre-installed, no source build)
python -c "import xllm; print('xllm import ok:', xllm.__file__)"

"setup_${PROFILE}"
