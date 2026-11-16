#!/usr/bin/env bash
# Prepare the CI environment for one supported specforge example.
# $1 is the manifest profile. Unknown profiles fail before any install.
# specforge itself is installed from TARGET_ROOT (the release checkout under
# test), so the guarded tag is exactly the code that runs.
#
# The image already ships a CANN-9.0.0 + torch-2.10.0 + sglang-0.5.18 stack
# (sglang:v0.5.18-cann9.0.0-910b). We reuse it and only replace the parts that
# specforge needs but pip's pyproject pins to a different torch:
#
#   - specforge installs with --no-deps (pyproject pins torch==2.13.0 which
#     would clobber the image's 2.10.0 line; the typed CLI works against the
#     image's torch because specforge only touches it through specforge.* APIs).
#   - accelerate + mooncake + modelscope + click + etc. install on top; the
#     few new transitive deps (pyyaml, psutil, safetensors, ...) are pinned
#     loosely.
#   - mooncake-transfer-engine-npu==0.3.14 is the only mooncake wheel
#     published for the ascend toolchain; we install via Aliyun (Aliyun has
#     it, the Huawei Cloud ascend index does not host the npu flavor).
#   - upstream switched the server-side capture from a patch to a plugin
#     (specforge-sglang-capture). The plugin needs three SGLang extension
#     points that exist only in SGLang main (--enable-return-aux-hidden-states,
#     ModelRunner.forward_observer, Scheduler.defer_response), so we build
#     SGLang from source at the commit upstream's own CI pins
#     (SGLANG_COMMIT in specforge's .github/workflows/test.yaml), upgrade the
#     matching sgl-kernel-npu wheel, and install the upstream plugin.
#     Quick-start-Ascend.md does the same in the smoke; we factor it here so
#     the runner does not repeat it for every matrix leg.
#
# Two notes vs. the Quick-start smoke:
#   - the smoke downloads a real Qwen3.5-4B from ModelScope (~8GB). For the
#     single supported example (qwen3.5-4b-dflash-npu.yaml) we do
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
  # specforge Quick-start uses torch 2.10.0 + torch_npu 2.10.0.post6. The CANN
  # image ships neither, so install the pair when missing (torch and torch_npu
  # must match exactly - torch_npu.post6 requires torch==2.10.0). torch comes
  # from the CPU wheel index (the default PyPI aarch64 wheel would drag the
  # CUDA runtime in), torch_npu from the Ascend index.
  if python -c "
import torch, torch_npu
raise SystemExit(
    0 if torch.__version__.startswith('2.10.0')
    and torch_npu.__version__.startswith('2.10.0') else 1)
" 2>/dev/null; then
    echo "reusing image torch stack ($(python -c 'import torch; print(torch.__version__)'))"
    return
  fi
  echo "installing torch==2.10.0 torch_npu==2.10.0.post6"
  python -m pip install \
    -f https://mirrors.aliyun.com/pytorch-wheels/cpu \
    --extra-index-url "$ASCEND_PIP_INDEX" \
    "torch==2.10.0" "torch_npu==2.10.0.post6" >/tmp/setup-torch.log 2>&1 \
    || { echo "setup: FAILED - pip install torch stack:" >&2;
         tail -20 /tmp/setup-torch.log >&2; exit 1; }
  python -c "import torch, torch_npu; print('torch', torch.__version__, '/ torch_npu', torch_npu.__version__)"
}

# Build SGLang from source at the commit upstream specforge's CI pins.
#
# The capture plugin needs three extension points that only exist in SGLang
# main (PR #43449 / #43450): --enable-return-aux-hidden-states,
# ModelRunner.forward_observer and Scheduler.defer_response. No released
# wheel has them (0.5.23 still lacks them), so the image's sglang 0.5.18 is
# replaced by a source build of the pinned commit.
install_sglang_source() {
  if [[ ! -f "$TARGET_ROOT/.github/workflows/test.yaml" ]]; then
    echo "setup: FAILED - $TARGET_ROOT/.github/workflows/test.yaml missing" >&2
    exit 1
  fi
  local sglang_commit
  sglang_commit=$(sed -n 's/^[[:space:]]*SGLANG_COMMIT:[[:space:]]*\([0-9a-f]\{40\}\).*/\1/p' \
    "$TARGET_ROOT/.github/workflows/test.yaml" | head -1)
  if [[ -z "$sglang_commit" ]]; then
    echo "setup: FAILED - upstream SGLANG_COMMIT not found in test.yaml" >&2
    exit 1
  fi
  echo "installing sglang from source at ${sglang_commit}"
  local src="${GITHUB_WORKSPACE}/sglang-src"
  rm -rf "$src" && mkdir -p "$src"
  # HTTP/1.1 + 5 次重试：拉大仓库时偶发 HTTP2 framing / GnuTLS recv error。
  local git_opts=(-c http.version=HTTP/1.1 -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=60)
  (
    cd "$src"
    git init -q
    git remote add origin https://github.com/sgl-project/sglang.git
    for attempt in 1 2 3 4 5; do
      if git "${git_opts[@]}" fetch --depth 1 origin "$sglang_commit"; then
        break
      fi
      echo "sglang fetch attempt ${attempt} failed, retrying" >&2
      sleep 10
      [ "$attempt" = 5 ] && { echo "setup: FAILED - fetch sglang" >&2; exit 1; }
    done
    git checkout -q FETCH_HEAD
  ) >/tmp/setup-sglang-clone.log 2>&1 \
    || { echo "setup: FAILED - clone sglang:" >&2; tail -20 /tmp/setup-sglang-clone.log >&2; exit 1; }

  # NPU packaging: pyproject_npu.toml is the ascend flavour (all_npu extra).
  ( cd "$src/python" && rm -f pyproject.toml && mv pyproject_npu.toml pyproject.toml )

  # The matching sgl-kernel-npu wheel: the image's 2026.6.1 lacks symbols the
  # new SGLang imports (sgl_kernel_npu.activation.fused_sigmoid_mul), which
  # makes the model modules fail to register ("has no SGlang implementation").
  # CANN-side custom ops from the image are reused as-is.
  local kernel_tag="2026.9.0.post12"
  local py_tag="py$(python -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
  # CANN 版本取当前环境里的那个（sgl-kernel-npu 的 wheel 按 (torch, CANN, 设备,
  # python) 出包，写死会拼出不存在的文件名 -> 404）。上游目前只发到 CANN 9.1.0
  # 的 910b 包，可用 SGL_KERNEL_CANN_VER 覆盖。
  local cann_ver device="910b" arch
  cann_ver=$(ls -d /usr/local/Ascend/cann-* 2>/dev/null | head -1 | sed 's/.*cann-//')
  cann_ver="${SGL_KERNEL_CANN_VER:-$cann_ver}"
  arch=$(uname -m)
  local wheel="sgl-kernel-npu-${kernel_tag}-torch2.10.0-${py_tag}-cann${cann_ver}-${device}-${arch}.zip"
  mkdir -p "$src/npu-ops"
  (
    cd "$src/npu-ops"
    curl -fSL --retry 5 --retry-all-errors --http1.1 -o "$wheel" \
      "https://github.com/sgl-project/sgl-kernel-npu/releases/download/${kernel_tag}/${wheel}"
    python - "$wheel" <<'PYUNZIP'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as z:
    z.extractall(".")
PYUNZIP
    python -m pip install deep_ep*.whl sgl_kernel_npu*.whl torch_memory_saver*.whl
  ) >/tmp/setup-sglang-kernel.log 2>&1 \
    || { echo "setup: FAILED - install sgl-kernel-npu:" >&2; tail -20 /tmp/setup-sglang-kernel.log >&2; exit 1; }

  # triton must be the ascend fork: the community one has no
  # triton.language.extra.cann, so the NPU model modules fail to import.
  if ! python -c "import triton.language.extra.cann" 2>/dev/null; then
    python -m pip install --force-reinstall --no-deps "triton-ascend==3.6.0" pybind11 \
      --extra-index-url "$ASCEND_PIP_INDEX" >/tmp/setup-triton.log 2>&1 \
      || { echo "setup: FAILED - install triton-ascend:" >&2; tail -20 /tmp/setup-triton.log >&2; exit 1; }
  fi

  # SGLang itself, with upstream's own dependency set (pyproject_npu.toml).
  # torch stays pinned at 2.10.0 (the image's torch_npu requires it) - without
  # the constraint the resolver would pull the newest torch and break the pair.
  printf 'torch==2.10.0\ntorchvision==0.25.0\ntorchaudio==2.10.0\n' >/tmp/torch-constraint.txt
  python -m pip install -e "$src/python" --constraint /tmp/torch-constraint.txt \
    >/tmp/setup-sglang-pip.log 2>&1 \
    || { echo "setup: FAILED - pip install sglang:" >&2; tail -20 /tmp/setup-sglang-pip.log >&2; exit 1; }
  python -c "from importlib.metadata import version; print('sglang', version('sglang'), '/ sgl_kernel_npu', version('sgl_kernel_npu'))"
}

# The upstream capture plugin (specforge-sglang-capture) lives in the guarded
# checkout; it registers itself through the sglang.srt.plugins entry point, so
# it must be installed into the same python as SGLang.
install_capture_plugin() {
  local plugin_dir="$TARGET_ROOT/plugins/sglang-spec-capture"
  if [[ ! -d "$plugin_dir" ]]; then
    echo "setup: FAILED - $plugin_dir missing (upstream capture plugin)" >&2
    exit 1
  fi
  python -m pip install --no-deps "$plugin_dir" >/tmp/setup-capture-plugin.log 2>&1 \
    || { echo "setup: FAILED - pip install capture plugin:" >&2; tail -20 /tmp/setup-capture-plugin.log >&2; exit 1; }
  python -c "
from importlib.metadata import entry_points
eps = [e.name for e in entry_points(group='sglang.srt.plugins')]
assert 'specforge_spec_capture' in eps, eps
print('capture plugin registered: specforge_spec_capture')
"
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
    "pydantic" "psutil" "safetensors" "requests" "typing-extensions" \
    "transformers==5.19.0" "datasets" "huggingface-hub" "numpy"
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
    'mooncake-transfer-engine-npu==0.3.14' \
    --index-url "$ALIYUN_PIP_INDEX" \
    --extra-index-url https://pypi.org/simple/ \
    >/tmp/setup-mooncake.log 2>&1 \
    || { echo "setup: FAILED - pip install mooncake:" >&2;
         tail -20 /tmp/setup-mooncake.log >&2; exit 1; }

  install_sglang_source
  install_capture_plugin

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
