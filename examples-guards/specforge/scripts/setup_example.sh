#!/usr/bin/env bash
# Prepare the CI environment for one supported specforge example.
# $1 is the manifest profile. Unknown profiles fail before any install.
# specforge itself is installed from TARGET_ROOT (the release checkout under
# test), so the guarded tag is exactly the code that runs.
#
# The image already ships a CANN-9.0.0 + torch-2.10.0 + sglang-0.5.18 stack
# (sglang:v0.5.18-cann9.0.0-910b). We reuse it and only patch the parts that
# specforge needs but pip's pyproject pins to a different torch:
#
#   - specforge installs with --no-deps (pyproject pins torch==2.13.0 which
#     would clobber the image's 2.10.0 line; the typed CLI works against the
#     image's torch because specforge only touches it through specforge.* APIs).
#   - accelerate + mooncake + modelscope + click + etc. install on top; the
#     few new transitive deps (pyyaml, psutil, safetensors, ...) are pinned
#     loosely.
#   - mooncake-transfer-engine-npu==0.3.13.post1 is the only mooncake wheel
#     published for the ascend toolchain; we install via Aliyun (Aliyun has
#     it, the Huawei Cloud ascend index does not host the npu flavor).
#   - the spec-capture sglang patch set is applied here; Quick-start-Ascend
#     already does this in the smoke and we don't want the runner to
#     re-derive it for every matrix leg.
#
# Two notes vs. the Quick-start smoke:
#   - the smoke downloads a real Qwen3.5-4B from ModelScope (~8GB). For the
#     single supported example (qwen3.5-4b-dflash-online-npu.yaml) we do
#     the same download because the smoke config is the supported recipe.
#     Smaller qwen2.5-0.5b recipes are not currently in `supported`; they
#     would need a separate profile if added later.
#   - we do not run the smoke here. setup is environment-only; the runner
#     does the actual `specforge train -c <yaml>` invocation.
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
  # specforge Quick-start uses CANN 9.0.0 + torch 2.10.0 + sglang 0.5.18;
  # the image is expected to ship that line. Reuse when present.
  if python -c "
import torch, torch_npu
raise SystemExit(
    0 if torch.__version__.startswith('2.10.0')
    and torch_npu.__version__.startswith('2.10.0') else 1)
" 2>/dev/null; then
    echo "reusing image torch stack ($(python -c 'import torch; print(torch.__version__)'))"
    return
  fi
  echo "installing torch==2.10.0 torch_npu==2.10.0"
  pip_ascend torch==2.10.0 torch_npu==2.10.0
}

# Apply specforge's spec-capture patch set to the image's pre-installed sglang.
# Quick-start-Ascend does the same; we factor it here so the runner does not
# repeat the (large) git apply on every matrix leg. The patch is idempotent
# and writes a sentinel .spec_capture_patch.applied under sglang/srt/.
apply_sglang_patches() {
  if [[ ! -d "$TARGET_ROOT/scripts" ]]; then
    echo "smoke: FAILED - $TARGET_ROOT/scripts missing; specforge-install-source first" >&2
    exit 1
  fi
  pushd "$TARGET_ROOT" >/dev/null
  SGLANG_VER=$(python -c "from importlib.metadata import version; print(version('sglang'))")
  if [[ -f scripts/apply_sglang_spec_capture_patch.sh ]]; then
    bash scripts/apply_sglang_spec_capture_patch.sh --target "v${SGLANG_VER}" \
      >/tmp/setup-sglang-patch.log 2>&1 \
      || { echo "smoke: FAILED - apply_sglang_spec_capture_patch.sh:" >&2;
           tail -30 /tmp/setup-sglang-patch.log >&2; exit 1; }
  else
    echo "setup: WARNING - apply_sglang_spec_capture_patch.sh missing; assuming already patched"
  fi
  popd >/dev/null
}

setup_specforge() {
  # specforge from the guarded release checkout plus the minimal stack the
  # typed CLI actually imports (everything else is lazily loaded inside the
  # producer/consumer/benchmark subcommands, which we do not exercise in the
  # single-step smoke).
  echo "installing specforge from $TARGET_ROOT"
  python -m pip install --no-deps "$TARGET_ROOT"
  python -m pip install \
    "click>=8.0" "accelerate" "pyyaml" "tqdm" \
    "pydantic" "psutil" "safetensors" "requests" "typing-extensions"
  python -c "import specforge, click; print('specforge', getattr(specforge, '__version__', 'unknown'), '/ click', click.__version__)"

  # mooncake (PyPI NPU prebuilt wheel). Aliyun mirror keeps it in sync; the
  # Huawei Cloud ascend index does not host the npu flavor. apt deps below
  # are required because the wheel's bundled libs do not include
  # libibverbs / libcurl / libnuma.
  apt-get update -qq >/dev/null 2>&1
  apt-get install -qq -y --no-install-recommends \
    libibverbs1 libcurl4 libnuma1 >/dev/null 2>&1 \
    || { echo "setup: FAILED - apt install runtime deps" >&2; exit 1; }
  python -m pip install --upgrade \
    'mooncake-transfer-engine-npu==0.3.13.post1' \
    --index-url "$ALIYUN_PIP_INDEX" \
    --extra-index-url https://pypi.org/simple/ \
    >/tmp/setup-mooncake.log 2>&1 \
    || { echo "setup: FAILED - pip install mooncake:" >&2;
         tail -20 /tmp/setup-mooncake.log >&2; exit 1; }

  apply_sglang_patches

  # Pre-download the example model from ModelScope (China-reachable) because
  # runners cannot reach HuggingFace. The model id is read from the recipe's
  # model.target_model_path (its upstream id, e.g. Qwen/Qwen3-4B) so each
  # supported recipe pulls its own model instead of a fixed hard-coded one.
  # The local snapshot dir is exported as SPECFORGE_MODEL_PATH for
  # run_example.sh + overlay_args to reference.
  python -m pip install --quiet modelscope
  python - "$TARGET_ROOT/$EXAMPLE_PATH" <<'PY'
import os
import sys

import yaml

os.environ.setdefault("TQDM_MININTERVAL", "15")
recipe = sys.argv[1]
doc = yaml.safe_load(open(recipe, encoding="utf-8"))
model_id = doc["model"]["target_model_path"]
from modelscope import snapshot_download

MODEL_CACHE = os.environ.get(
    "MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope"))
local = snapshot_download(model_id, cache_dir=MODEL_CACHE)
with open(os.environ["GITHUB_ENV"], "a") as fh:
    fh.write(f"SPECFORGE_MODEL_PATH={local}\n")
print("SPECFORGE_MODEL_PATH=", local)
PY

  # Pre-download sharegpt fixture data (regenerated copy in CI workdir).
  # The training CLI expects a JSONL with at least one conversation row;
  # the smoke generates one inline and we keep a copy under fixtures/ for
  # the runner to copy into the workdir.
  mkdir -p "${FIXTURE_DIR:?FIXTURE_DIR is required}"
  cat > "${FIXTURE_DIR}/sharegpt_train.jsonl" <<'JSONL'
{"id": "smoke_1", "conversations": [{"role": "user", "content": "Explain the difference between supervised and unsupervised learning in machine learning."}, {"role": "assistant", "content": "Supervised learning uses labeled data to learn a mapping from inputs to outputs; unsupervised learning discovers structure in unlabeled data. Choose supervised when you have a well-defined prediction target and labeled examples; unsupervised when labels are scarce and the goal is exploratory or structural. Modern practice combines them: unsupervised pretraining learns representations, then supervised fine-tuning specializes them."}]}
JSONL
  echo "wrote ${FIXTURE_DIR}/sharegpt_train.jsonl"
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
FIXTURE_DIR="${FIXTURE_DIR:?FIXTURE_DIR is required}"

HERE=$(cd "$(dirname "$0")" && pwd)
export PIP_CONSTRAINT="$(cd "$HERE/.." && pwd)/constraints-npu.txt"

# CANN must be sourced before any torch_npu import; the image's
# ascend-toolkit is the one Quick-start-Ascend.md uses (CANN 9.0.0).
source /usr/local/Ascend/ascend-toolkit/set_env.sh

select_pip_index
python -m pip install -U pip setuptools wheel
ensure_torch_stack
"setup_${PROFILE}"
