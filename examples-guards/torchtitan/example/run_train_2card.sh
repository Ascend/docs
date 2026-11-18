#!/usr/bin/env bash
# Two-card torchtitan smoke on Ascend NPU (FSDP shard=2).
#
# Same entry point as the single-card variant (module / config come from
# the manifest's overlay_args); two ranks share one container, so HCCL's
# per-rank listening socket has to be moved off the default port 16666 or
# both ranks race for it (EI0020).
#
# Env:
#   TARGET_ROOT     checkout of pytorch/torchtitan (required)
#   CI_OUTPUT_DIR   where --dump-folder points (optional)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"

export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-60000-60200}"

cd "$TARGET_ROOT"

exec torchrun \
    --nproc_per_node=2 \
    --rdzv_backend c10d \
    --rdzv_endpoint="localhost:0" \
    --local-ranks-filter 0 \
    --tee 3 \
    -m torchtitan.train \
    "$@"
