#!/usr/bin/env bash
# Eight-card torchtitan smoke on Ascend NPU (FSDP shard=8 + expert
# parallelism for the deepseek recipe).
#
# Same entry point as the single- and two-card variants (module / config come
# from the manifest's overlay_args); the ranks share one container, so HCCL's
# per-rank listening sockets have to be moved off the default port 16666 or
# they race for it (EI0020).
#
# Env:
#   TARGET_ROOT     checkout of pytorch/torchtitan (required)
#   CI_OUTPUT_DIR   where --dump-folder points (optional)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"

# The override module lives next to this script (project side, outside the
# upstream checkout) and is imported by name via --override.
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"

# Serial inductor compilation: parallel compile workers spike host memory
# past the runner's cgroup limit and get OOM-killed mid-run.
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"

# Eight ranks in one container -> move the HCCL listener off the default port.
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-60000-60200}"

# torchtitan resolves the tokenizer from ./tests/assets/tokenizer, i.e.
# relative to the checkout root.
cd "$TARGET_ROOT"

exec torchrun \
    --nproc_per_node=8 \
    --rdzv_backend c10d \
    --rdzv_endpoint="localhost:0" \
    --local-ranks-filter 0 \
    --tee 3 \
    -m torchtitan.train \
    "$@"
