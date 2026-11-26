#!/usr/bin/env bash
# Prepare the CI environment for one supported speculators example.
# $1 is the manifest profile (demo / evaluate). Unknown profiles fail before
# any install.
#
# speculators is installed from TARGET_ROOT (the release checkout under test),
# so the guarded tag is exactly the code that runs.
#
# The image already ships the whole vLLM stack:
#   torch 2.10.0+cpu / torch_npu 2.10.0.post4 / torchvision 0.25.0 /
#   torchaudio 2.10.0 / transformers 5.5.4 / vllm 0.23.0 (Ascend build) /
#   vllm-ascend 0.23.0 / triton-ascend 3.2.2 / CANN 9.1.0.
# Those versions are pinned in constraints-npu.txt: PyPI's vllm wheels are CUDA
# builds (they want libcudart.so.13) and a torch-2.11 vllm wheel breaks the
# image's torch 2.10 stack, so the only working vllm is the one already in the
# image - never let pip replace it.
#
# What the image is missing and this setup adds:
#   - datasets (speculators' data pipeline needs it; the image has none)
#   - modelscope (ModelScope is the reachable model host from the runners)
#   - setuptools-git-versioning (speculators' build backend)
#   - aiperf 0.13.0 + the local evaluate plugin (evaluate profile only)
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
EXAMPLE_PATH="${EXAMPLE_PATH:?EXAMPLE_PATH is required}"
GITHUB_WORKSPACE="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
GITHUB_ENV="${GITHUB_ENV:?GITHUB_ENV is required}"

HERE=$(cd "$(dirname "$0")" && pwd)
export PIP_CONSTRAINT="$(cd "$HERE/.." && pwd)/constraints-npu.txt"

# The image's console scripts (vllm, torchrun, pip) live next to its python,
# /usr/local/python3.12.13/bin. The image ENV usually has it on PATH; make it
# explicit so the run step finds the same interpreter this setup installed into.
if [[ -d /usr/local/python3.12.13/bin ]]; then
  export PATH="/usr/local/python3.12.13/bin:$PATH"
fi

# CANN must be sourced before any torch_npu import.
source /usr/local/Ascend/ascend-toolkit/set_env.sh

CLUSTER_PIP_HOST=cache-service.nginx-pypi-cache.svc.cluster.local
export CLUSTER_PIP_INDEX="http://${CLUSTER_PIP_HOST}/pypi/simple"
ALIYUN_PIP_INDEX=https://mirrors.aliyun.com/pypi/simple/

select_pip_index() {
  if python -c "
import urllib.request
try:
    urllib.request.urlopen('${CLUSTER_PIP_INDEX}', timeout=3)
except Exception:
    raise SystemExit(1)
" 2>/dev/null; then
    export PIP_INDEX_URL="$CLUSTER_PIP_INDEX"
    export PIP_TRUSTED_HOST="$CLUSTER_PIP_HOST"
  else
    export PIP_INDEX_URL="$ALIYUN_PIP_INDEX"
    unset PIP_TRUSTED_HOST
  fi
  echo "pip index: $PIP_INDEX_URL"
}

# speculators from the guarded checkout. --no-deps: the image's torch /
# transformers / vllm line must not be touched (see constraints-npu.txt).
# The runtime deps the image lacks are installed explicitly below.
install_speculators() {
  echo "installing speculators from $TARGET_ROOT"
  # The build backend is setuptools + setuptools-git-versioning; install them
  # into the target env and build with --no-build-isolation so pip does not
  # spin up a separate build env (which would re-download the pinned torch /
  # numpy / transformers line from the constraint file just to build a wheel).
  python -m pip install "setuptools" "wheel" "setuptools-git-versioning" \
    >/tmp/setup-build-tools.log 2>&1 \
    || { echo "setup: FAILED - pip install build tools:" >&2;
         tail -20 /tmp/setup-build-tools.log >&2; exit 1; }
  python -m pip install --no-deps --no-build-isolation "$TARGET_ROOT" \
    >/tmp/setup-speculators.log 2>&1 \
    || { echo "setup: FAILED - pip install speculators:" >&2;
         tail -20 /tmp/setup-speculators.log >&2; exit 1; }
  # hs_connectors is the hidden-states transfer backend the trainer and
  # scripts/launch_vllm.py import; upstream resolves it from PyPI.
  python -m pip install --no-deps "hs-connectors" \
    >/tmp/setup-hs-connectors.log 2>&1 \
    || { echo "setup: FAILED - pip install hs-connectors:" >&2;
         tail -20 /tmp/setup-hs-connectors.log >&2; exit 1; }
  # check the installed metadata (importing speculators pulls torch, which
  # lives in the image env; the metadata check works in any interpreter)
  python -c "import importlib.metadata as m; print('speculators', m.version('speculators'), '| hs-connectors', m.version('hs-connectors'))"
}

# datasets (speculators' prepare-data / regenerate-responses need it) plus the
# two transitive deps the image lacks. datasets 4.1.1 works with the image's
# numpy 1.26.4 / pandas / scipy.
install_data_deps() {
  echo "installing datasets + modelscope"
  # datasets with its own deps: the image has fsspec/dill/requests but not
  # pyarrow/xxhash/multiprocess. pyarrow and pandas are pinned to the last
  # releases that still work with the image's numpy 1.26.4 (newer ones
  # require numpy >= 2).
  python -m pip install "datasets==4.0.0" "pyarrow==18.1.0" "pandas==2.2.3" \
    >/tmp/setup-datasets.log 2>&1 \
    || { echo "setup: FAILED - pip install datasets:" >&2;
         tail -20 /tmp/setup-datasets.log >&2; exit 1; }
  python -m pip install "modelscope" \
    >/tmp/setup-modelscope.log 2>&1 \
    || { echo "setup: FAILED - pip install modelscope:" >&2;
         tail -20 /tmp/setup-modelscope.log >&2; exit 1; }
  python -c "import datasets, modelscope, numpy, pyarrow; print('datasets', datasets.__version__, '| numpy', numpy.__version__, '| pyarrow', pyarrow.__version__)"
}

# Pre-download the models from ModelScope (China-reachable; the runners fall
# back to ModelScope when huggingface.co is slow). The verifier is Qwen3-8B;
# the dflash draft is z-lab/Qwen3-8B-DFlash-b16.
download_modelscope_models() {
  python -m pip install --quiet modelscope >/dev/null 2>&1 || true
  python - <<'PY'
import os
from modelscope import snapshot_download

cache = os.environ.get("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope"))
verifier = snapshot_download("Qwen/Qwen3-8B", cache_dir=cache)
draft = snapshot_download("z-lab/Qwen3-8B-DFlash-b16", cache_dir=cache)
with open(os.environ["GITHUB_ENV"], "a") as fh:
    fh.write(f"SPECULATORS_VERIFIER_PATH={verifier}\n")
    fh.write(f"SPECULATORS_DRAFT_PATH={draft}\n")
print("SPECULATORS_VERIFIER_PATH=", verifier)
print("SPECULATORS_DRAFT_PATH=", draft)
PY
}

# evaluate profile: aiperf + the upstream evaluate plugin (which registers the
# spec_al_* datasets) and the HF draft checkpoint the example serves.
# NOTE: the evaluate entry is currently listed as unsupported in the manifest
# (vllm 0.23 has no --per-request-spec-decode-metrics), so this profile is
# not scheduled today. It is kept because the pieces it installs are exactly
# what the entry needs once the image ships a vllm with that flag.
setup_evaluate() {
  echo "installing aiperf 0.13.0 + the upstream evaluate plugin"
  python -m pip install "aiperf==0.13.0" \
    >/tmp/setup-aiperf.log 2>&1 \
    || { echo "setup: FAILED - pip install aiperf:" >&2;
         tail -20 /tmp/setup-aiperf.log >&2; exit 1; }
  python -m pip install --no-deps "$TARGET_ROOT/scripts/evaluate" \
    >/tmp/setup-aiperf-plugin.log 2>&1 \
    || { echo "setup: FAILED - pip install the evaluate plugin:" >&2;
         tail -20 /tmp/setup-aiperf-plugin.log >&2; exit 1; }
}

setup_demo() {
  echo "demo profile: the two servers share the verifier + the dflash draft"
}

supported_profiles() {
  declare -F | awk '/^declare -f setup_/ { sub(/^declare -f setup_/, ""); print }' | paste -sd' ' -
}

select_pip_index
python -m pip install -U pip setuptools wheel >/dev/null 2>&1 || true
install_data_deps
install_speculators

if ! declare -F "setup_${PROFILE}" >/dev/null 2>&1; then
  echo "unknown profile: ${PROFILE} (supported: $(supported_profiles))" >&2
  exit 1
fi

download_modelscope_models
"setup_${PROFILE}"
echo "setup done: profile=${PROFILE}"
