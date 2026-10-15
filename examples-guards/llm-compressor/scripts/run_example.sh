#!/usr/bin/env bash
# Run one llm-compressor example from a CI working copy of the target tree.
# The example file is not modified. torch_npu is imported in-process so a
# script that would use CUDA on an NVIDIA machine uses npu here.
set -euo pipefail

export PYTHONNOUSERSITE=1

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <example-relpath>" >&2
  exit 2
fi

EXAMPLE_REL="$1"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
CI_OUTPUT_DIR="${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"
EXAMPLE_PATH="$TARGET_ROOT/$EXAMPLE_REL"
if [[ ! -f "$EXAMPLE_PATH" ]]; then
  echo "upstream example not found: $EXAMPLE_PATH" >&2
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$CI_OUTPUT_DIR"
RUN_LOG="$CI_OUTPUT_DIR/run.log"

export PATH="/usr/local/sbin:$PATH"
# shellcheck disable=SC1091
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
# Output goes through a pipe to tee. Buffered prints hid where a hung
# example stopped, so flush every line.
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export EXAMPLE_PATH

# benchmark_smoothquant_ddp.py defaults to one process and only calls
# init_dist when --num_gpus is greater than 1. Other init_dist scripts
# refuse to start unless torchrun provided the elastic env.
if [[ "$EXAMPLE_REL" == "examples/quantization_w8a8_int8/benchmark_smoothquant_ddp.py" ]]; then
  GUARD_KIND=npu_oneshot
elif grep -q 'init_dist()' "$EXAMPLE_PATH"; then
  GUARD_KIND=npu_ddp
else
  GUARD_KIND=npu_oneshot
fi

if [[ "$GUARD_KIND" == "npu_ddp" ]]; then
  IFS=',' read -ra DEVICES <<< "$ASCEND_RT_VISIBLE_DEVICES"
  NPROC="${#DEVICES[@]}"
  if [[ "$NPROC" -lt 2 ]]; then
    echo "ddp example needs at least 2 devices, got ${ASCEND_RT_VISIBLE_DEVICES}" >&2
    exit 1
  fi
  python -m torch.distributed.run \
    --nproc_per_node="$NPROC" \
    --master_port="${MASTER_PORT:-29500}" \
    "$SCRIPT_DIR/run_guard.py" 2>&1 | tee "$RUN_LOG"
else
  python "$SCRIPT_DIR/run_guard.py" 2>&1 | tee "$RUN_LOG"
fi

if ! grep -q 'LLM_COMPRESSOR_WORKLOAD_DEVICE=npu:0' "$RUN_LOG"; then
  echo "missing NPU workload device anchor" >&2
  exit 1
fi
