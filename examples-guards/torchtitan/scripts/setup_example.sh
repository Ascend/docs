#!/usr/bin/env bash
# Prepare the environment for the torchtitan examples.
#
# Usage: setup_example.sh <profile>
#
# Env (provided by the examples engine):
#   TARGET_ROOT      checkout of pytorch/torchtitan at the tested ref
#   GITHUB_WORKSPACE workspace root (writable)
#   GITHUB_ENV       file where exported env vars are persisted
#
# Profile:
#   torchtitan       llama3 debugmodel smoke (1-card / 2-card)
set -euo pipefail

[[ $# -lt 1 ]] && { echo "usage: $0 <profile>" >&2; exit 2; }
PROFILE="$1"

# The examples engine ships a pip cache; fall back to the public mirrors.
CLUSTER_PIP_HOST=cache-service.nginx-pypi-cache.svc.cluster.local
if curl -sm 5 -o /dev/null "http://${CLUSTER_PIP_HOST}/pypi/simple/"; then
    PIP_INDEX_URL="http://${CLUSTER_PIP_HOST}/pypi/simple"
    PIP_EXTRA_INDEX_URL="https://mirrors.aliyun.com/pypi/simple"
else
    PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple"
    PIP_EXTRA_INDEX_URL="https://repo.huaweicloud.com/ascend/repos/pypi"
fi

export PIP_INDEX_URL PIP_EXTRA_INDEX_URL
export PIP_TRUSTED_HOST="cache-service.nginx-pypi-cache.svc.cluster.local,mirrors.aliyun.com,repo.huaweicloud.com,pypi.org"

HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"

# Keep the community CUDA wheels and the community `triton` package out of
# the environment: the community triton has CUDA-only backends and would
# shadow the Ascend fork's `triton/` tree.
export PIP_CONSTRAINT="$PROJECT_ROOT/constraints-npu.txt"

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
GITHUB_WORKSPACE="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
GITHUB_ENV="${GITHUB_ENV:?GITHUB_ENV is required}"

# Versions pinned to the stack this project is verified against:
#   CANN 9.2.0 (container image) / torch 2.15.0 / torch_npu 2.15.0.dev20261116
#   triton-ascend 3.6.0+dev20261112 (Ascend fork of the triton 3.6 baseline)
#
# torch_npu and triton-ascend come from Ascend's *nightly* channel: torchtitan
# v0.4.0 needs torch >= 2.15 (torch.distributed.pipelining's
# analyze_pipeline_activation_liveness / PipelineStageInfo and FSDP2's
# MixedPrecisionPolicy(param_dtype_override_fn=...)), and the latest published
# torch_npu is 2.14.0, which pins torch==2.14.0.*.  The nightly channel is
# the only source of a torch_npu that matches torch 2.15.
TORCH_VERSION="2.15.0"
TORCH_NPU_VERSION="2.15.0.dev20261116"
TRITON_ASCEND_VERSION="3.6.0+dev20261112"
MODELSCOPE_VERSION="1.40.2"

# aliyun hosts the aarch64 torch CPU wheels under pytorch-wheels/cpu.
TORCH_WHEEL_INDEX="https://mirrors.aliyun.com/pytorch-wheels/cpu"
# Ascend's nightly channel.  Both directories are flat wheel listings (not
# PEP 503 indexes), so they are used as find-links, not as index urls.
ASCEND_NIGHTLY_ROOT="https://repo.huaweicloud.com/ascend/repos/pypi/nightly"
TORCH_NPU_INDEX="${ASCEND_NIGHTLY_ROOT}/torch-npu/"
TRITON_ASCEND_INDEX="${ASCEND_NIGHTLY_ROOT}/triton-ascend/"

supported_profiles() {
    declare -F | awk '/^declare -f setup_/ { sub(/^declare -f setup_/,""); print }' | paste -sd' ' -
}

setup_torchtitan() {
    echo "== setup: torchtitan profile =="
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    python -m pip install -U pip setuptools wheel

    # torch + torch_npu (aarch64 wheels).
    python -m pip install -f "$TORCH_WHEEL_INDEX" "torch==$TORCH_VERSION"
    # torch_npu from Ascend's nightly channel.  --no-deps: the wheel pins
    # torch==2.15.0.*, and without the pin pip would pull the PyPI (CUDA)
    # torch build on top of the CPU one installed above.  numpy (the
    # wheel's other dependency) is installed separately.
    python -m pip install --no-deps --find-links "$TORCH_NPU_INDEX" \
        "torch_npu==$TORCH_NPU_VERSION"
    python -m pip install numpy

    # Ascend Triton fork (the compiled regions need it). --no-deps keeps the
    # community `triton` out of the environment; pybind11 is the only skipped
    # dependency the Ascend backend imports at runtime (it compiles
    # npu_utils.cpp on first use).
    python -m pip install --no-deps \
        --find-links "$TRITON_ASCEND_INDEX" \
        "triton-ascend==$TRITON_ASCEND_VERSION"
    python -m pip install pybind11

    # modelscope: the tokenizer download demo in the quick start.
    python -m pip install "modelscope==$MODELSCOPE_VERSION"

    # torchtitan itself, editable, from the checkout under test.
    #
    # v0.4.0's requirements.txt pins `attn-gym[linear,cudnn]==0.0.17`.  The
    # extras pull CUDA-only wheels (nvidia-cudnn & co.), which have no aarch64
    # build -- and constraints-npu.txt already blocks them, so the resolver
    # fails outright on `pip install -e .`.  attn-gym only ships CUDA
    # attention kernels; torchtitan degrades officially without it
    # (torchtitan/models/common/attn_gym.py falls back to placeholder
    # kernels), so the dependency is dropped and the remaining requirements
    # are installed explicitly.  Same reason as the quick-start doc's
    # `uv pip install --excludes`.
    REQUIREMENTS_FILE="$GITHUB_WORKSPACE/torchtitan-requirements.txt"
    grep -v '^attn-gym' "$TARGET_ROOT/requirements.txt" > "$REQUIREMENTS_FILE"
    python -m pip install -r "$REQUIREMENTS_FILE"
    python -m pip install --no-deps -e "$TARGET_ROOT"

    # Tokenizer assets for the size-tier recipes (qwen3_0_6b): those configs
    # point hf_assets_path at ./assets/hf/<Model> instead of the local
    # tests/assets/tokenizer the debugmodel recipes use.  The cluster cannot
    # reach huggingface.co, so the files come from modelscope.  They land in
    # the CI working copy (never committed) and the modelscope cache on the
    # runner keeps them warm across jobs.
    python - <<'PY'
import os
from modelscope import snapshot_download

ASSETS = os.path.join(os.environ["TARGET_ROOT"], "assets", "hf")
TOKENIZER_FILES = [
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "special_tokens_map.json", "config.json", "generation_config.json",
]
for repo, dest in [
    ("Qwen/Qwen3-0.6B", "Qwen3-0.6B"),
    ("Qwen/Qwen3-1.7B", "Qwen3-1.7B"),
    ("Qwen/Qwen3-8B", "Qwen3-8B"),
    ("Qwen/Qwen3-14B", "Qwen3-14B"),
    ("deepseek-ai/deepseek-moe-16b-base", "deepseek-moe-16b-base"),
]:
    target = os.path.join(ASSETS, dest)
    if os.path.isdir(target) and os.listdir(target):
        print(f"assets: {dest} already present")
        continue
    os.makedirs(target, exist_ok=True)
    snapshot_download(
        repo, local_dir=target, allow_patterns=TOKENIZER_FILES
    )
    print(f"assets: {dest} -> {sorted(os.listdir(target))}")
PY

    # Sanity: NPU visible, stack importable, override module registered.
    python - <<'PY'
import torch, torch_npu
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("npu available:", torch.npu.is_available(), "devices:", torch.npu.device_count())
assert torch.npu.is_available(), "no NPU device visible"
PY

    # Sanity: the three NPU override modules register their overrides.
    # One module per model family, because the override mechanism matches
    # config classes by subclass: a single generic override would also
    # claim muse_glimmer's / deepseek_v3's own attention classes and
    # silently drop the fields those families add.
    PYTHONPATH="$PROJECT_ROOT/example${PYTHONPATH:+:$PYTHONPATH}" python - <<'PY'
import npu_compat
import npu_compat_deepseek
import npu_compat_muse
from torchtitan.config.override import _REGISTRY
registered = sorted(_REGISTRY)
print("registered overrides:", registered)
assert registered == [
    "npu_compat.npu_attention",
    "npu_compat_deepseek.deepseek_attention",
    "npu_compat_muse.muse_attention",
], registered
PY

    python -c "import torchtitan; print('torchtitan', torchtitan.__version__)"
}

# Unknown profile must fail before anything is installed.
if ! declare -F "setup_${PROFILE}" >/dev/null 2>&1; then
    echo "unknown profile: ${PROFILE} (supported: $(supported_profiles))" >&2
    exit 1
fi

"setup_${PROFILE}"
