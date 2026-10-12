#!/usr/bin/env bash
# Run one bitsandbytes example from the CI checkout of the target tree.
# Overlay CLI args come from OVERLAY_ARGS, a JSON array.
set -euo pipefail

export PYTHONNOUSERSITE=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <example-relpath>" >&2
  exit 2
fi

EXAMPLE_REL="$1"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
CI_OUTPUT_DIR="${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"
HERE=$(cd "$(dirname "$0")" && pwd)

case "$EXAMPLE_REL" in
  examples/cpu/cpu_training.py)
    KIND=cpu
    ;;
  examples/compile_inference.py)
    KIND=npu_compile
    ;;
  *)
    echo "unsupported bitsandbytes guard path: $EXAMPLE_REL" >&2
    exit 1
    ;;
esac

EXAMPLE_PATH="$TARGET_ROOT/$EXAMPLE_REL"
if [[ ! -f "$EXAMPLE_PATH" ]]; then
  echo "example not found: $EXAMPLE_PATH" >&2
  exit 1
fi

mkdir -p "$CI_OUTPUT_DIR"
RUN_LOG="$CI_OUTPUT_DIR/run.log"

eval "EXTRA_ARGS=( $(python "$HERE/expand_overlay.py") )"

export PATH="/usr/local/sbin:$PATH"
# shellcheck disable=SC1091
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# setup_example.sh already filled the default hub cache and recorded
# these in GITHUB_ENV. Keep the run off the network so a cached
# snapshot is not revalidated online.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

if [[ "$KIND" == "npu_compile" ]]; then
  export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${GITHUB_WORKSPACE:-$CI_OUTPUT_DIR}/.triton}"
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${GITHUB_WORKSPACE:-$CI_OUTPUT_DIR}/.inductor}"
  mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
  # The allocator warning is the device anchor. A filtered warning
  # list would hide it and leave exit 0 looking like a real NPU run.
  export PYTHONWARNINGS="${PYTHONWARNINGS:-default}"
fi

echo "running upstream bitsandbytes example: $EXAMPLE_REL"
cd "$TARGET_ROOT"
python "$EXAMPLE_PATH" "${EXTRA_ARGS[@]}" 2>&1 | tee "$RUN_LOG"

if [[ "$KIND" == "cpu" ]]; then
  # cpu_training.py still exits 0 when loss does not fall; it only
  # prints a warning. This profile does not use the NPU.
  if ! grep -q 'OK: Loss decreased as expected' "$RUN_LOG"; then
    echo "cpu_training did not report a loss decrease; exit 0 is fake green" >&2
    exit 1
  fi
elif [[ "$KIND" == "npu_compile" ]]; then
  if ! grep -q 'NPUCachingAllocator' "$RUN_LOG"; then
    echo "compile_inference did not allocate via torch_npu; exit 0 is fake green" >&2
    exit 1
  fi
  if ! grep -q 'Write me a poem about Machine Learning' "$RUN_LOG"; then
    echo "compile_inference did not print the generated prompt/output" >&2
    exit 1
  fi
fi
