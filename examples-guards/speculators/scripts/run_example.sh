#!/usr/bin/env bash
# Run one speculators example from a CI working copy of the target tree.
# $1 is the manifest entry path. The upstream file is copied into a scratch
# work dir and the Ascend adaptations are applied to the COPY - the upstream
# checkout stays untouched (never git add/commit/push).
#
# Why a copy instead of running the upstream file as-is: the upstream example
# scripts hardcode CUDA spellings and CUDA-only server flags. The adaptations
# below are the ones measured on Ascend A2 (2026-11-25, coder hdc-stable-npu-1,
# 2x910B4, vllm 0.23.0 + vllm-ascend 0.23.0):
#
#   CUDA_VISIBLE_DEVICES -> ASCEND_RT_VISIBLE_DEVICES
#       torch_npu does not read CUDA_VISIBLE_DEVICES; the two servers would
#       otherwise both land on card 0.
#   --attention-backend flash_attn  -> dropped
#       flash_attn is a CUDA backend. Ascend uses the default
#       AscendAttentionBackend.
#   + --block-size 128
#       The Ascend attention kernel's block size is 128. vllm's default cache
#       block size is 16 and has no common size with 128, so a plain server
#       aborts with "No common block size for 16.".
#   + --enforce-eager
#       The upstream demo pins --max-num-seqs 1. With speculative decoding the
#       cudagraph capture sizes must be multiples of (num_speculative_tokens+1),
#       which leaves no valid size under max_cudagraph_capture_size=1:
#       "No valid cudagraph sizes after rounding to multiple of 9".
#   demo_race.py reads its prompts from stdin
#       CI feeds one prompt and closes stdin; an empty line/EOF ends the loop.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <example-relpath>" >&2
  exit 2
fi

EXAMPLE_REL="$1"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
CI_OUTPUT_DIR="${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"
SPECULATORS_VERIFIER_PATH="${SPECULATORS_VERIFIER_PATH:?SPECULATORS_VERIFIER_PATH is required}"

EXAMPLE_PATH="$TARGET_ROOT/$EXAMPLE_REL"
[[ -e "$EXAMPLE_PATH" ]] || { echo "example not found: $EXAMPLE_PATH" >&2; exit 1; }

mkdir -p "$CI_OUTPUT_DIR"

# OVERLAY_ARGS is a JSON array of strings (the engine serializes the
# manifest's overlay_args). Each item is ONE argument; no shell splitting.
read_overlay() {
  "$PYTHON" - <<'PY'
import json
import os

raw = os.environ.get("OVERLAY_ARGS", "").strip()
if not raw or raw in ("null", '""'):
    raise SystemExit(0)
items = json.loads(raw)
if not isinstance(items, list):
    raise SystemExit("OVERLAY_ARGS must be a JSON array")
print("\n".join(str(item) for item in items))
PY
}

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  PYTHON=python
fi

# The image's console scripts (vllm, torchrun) live next to its python.
if [[ -d /usr/local/python3.12.13/bin ]]; then
  export PATH="/usr/local/python3.12.13/bin:$PATH"
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh
"$PYTHON" -c "import torch, torch_npu; print('NPU available:', torch.npu.is_available(), 'devices:', torch.npu.device_count())"

WORK_DIR="$CI_OUTPUT_DIR/work"
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

# ---------------------------------------------------------------------------
# demo: examples/demo/demo_side_by_side.sh + examples/demo/demo_race.py
# ---------------------------------------------------------------------------
run_demo() {
  local demo_dir="$WORK_DIR/examples/demo"
  mkdir -p "$demo_dir"
  cp "$TARGET_ROOT/examples/demo/demo_side_by_side.sh" "$demo_dir/"
  cp "$TARGET_ROOT/examples/demo/demo_race.py" "$demo_dir/"
  chmod +x "$demo_dir/demo_side_by_side.sh"

  local script="$demo_dir/demo_side_by_side.sh"
  # device selection
  sed -i 's/CUDA_VISIBLE_DEVICES=/ASCEND_RT_VISIBLE_DEVICES=/g' "$script"
  # CUDA-only attention backend
  sed -i 's/--attention-backend flash_attn //g' "$script"
  # Ascend kernel block size + no cudagraph capture (see header comment)
  sed -i 's|--gpu-memory-utilization 0.85|--gpu-memory-utilization 0.85 --block-size 128 --enforce-eager|' "$script"
  # a 32G card holds the 8B verifier plus the draft with less KV room than
  # the upstream 10240-token window asks for
  sed -i 's/--max-model-len 10240/--max-model-len 4096/' "$script"
  sed -i 's/--max-num-batched-tokens 10240/--max-num-batched-tokens 4096/' "$script"
  # the draft / model paths: the upstream demo takes them as CLI arguments,
  # so they are passed below instead of being patched in.

  local overlay=()
  while IFS= read -r line; do
    [[ -n "$line" ]] && overlay+=("$line")
  done < <(read_overlay)

  local extra=()
  local arg
  for arg in "${overlay[@]:-}"; do
    [[ -n "$arg" ]] && extra+=("$arg")
  done

  cd "$demo_dir"
  # demo_race.py is interactive: feed one prompt, then EOF ends the loop.
  printf 'Briefly describe AI.\n' | \
    bash "$script" \
      --model "$SPECULATORS_VERIFIER_PATH" \
      --draft "${SPECULATORS_DRAFT_PATH:?SPECULATORS_DRAFT_PATH is required}" \
      --method dflash \
      --gpus 0,1 \
      "${extra[@]}" \
    2>&1 | tee "$CI_OUTPUT_DIR/demo.log"

  # The demo's own assertion: both engines must produce the same text.
  if ! grep -qE "TEXT IDENTICAL|\[32msame" "$CI_OUTPUT_DIR/demo.log"; then
    echo "FAILED - the demo did not report identical outputs" >&2
    tail -40 "$CI_OUTPUT_DIR/demo.log" >&2
    exit 1
  fi
}

case "$EXAMPLE_REL" in
  examples/demo/demo_side_by_side.sh) run_demo ;;
  *)
    echo "no run recipe for $EXAMPLE_REL" >&2
    exit 2
    ;;
esac
