#!/usr/bin/env bash
# Run one speculators example from a CI working copy of the target tree.
# Overlay CLI args come from OVERLAY_ARGS (JSON array, possibly []).
# Never git add/commit/push the target tree.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <example-relpath>" >&2
  exit 2
fi

EXAMPLE_REL="$1"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
CI_OUTPUT_DIR="${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"
PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"

EXAMPLE_PATH="$TARGET_ROOT/$EXAMPLE_REL"
if [[ ! -f "$EXAMPLE_PATH" ]]; then
  echo "upstream example not found: $EXAMPLE_PATH" >&2
  exit 1
fi

mkdir -p "$CI_OUTPUT_DIR"
RUN_LOG="$CI_OUTPUT_DIR/run.log"

export PATH="$PROJECT_ROOT/scripts/shims:/usr/local/bin:/usr/local/sbin:$PATH"
export PYTHONPATH="$PROJECT_ROOT/scripts/sitecustomize_dir${PYTHONPATH:+:$PYTHONPATH}"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
  set +u
  source /usr/local/Ascend/nnal/atb/set_env.sh
  set -u
fi
if [[ -d /usr/local/Ascend/driver/lib64/driver ]]; then
  export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}"
fi

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
# Leave HF_ENDPOINT and Xet at their defaults so downloads go straight
# to huggingface.co. A local snapshot is passed as MODEL below; do not
# send vLLM down the ModelScope file-listing path.
unset VLLM_USE_MODELSCOPE
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

eval "EXTRA_ARGS=( $(python "$PROJECT_ROOT/scripts/overlay_args.py") )"
echo "running $EXAMPLE_PATH with ${#EXTRA_ARGS[@]} overlay args"
if ((${#EXTRA_ARGS[@]})); then
  printf 'overlay arg: %q\n' "${EXTRA_ARGS[@]}"
fi

# The upstream script is not edited. Extra args only arrive if that
# script already forwards "$@".

# Hub ids go through snapshot_download so vLLM 0.23 never hits
# modelscope_list_repo_files (KeyError: Type on current modelscope_hub).
# snapshot_download may print progress on stdout; only keep a path line.
RESOLVED_MODEL="$(python "$PROJECT_ROOT/scripts/resolve_model.py" "$EXAMPLE_PATH" | grep '^/' | tail -n 1)"
if [[ -z "$RESOLVED_MODEL" ]]; then
  echo "resolve_model.py printed no filesystem path" >&2
  exit 1
fi
export MODEL="$RESOLVED_MODEL"
echo "resolved MODEL=$MODEL"
# The snapshot is in the shared cache. Stay offline so a later
# from_pretrained / load_dataset does not revalidate over the network.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "$TARGET_ROOT"
set +e
bash "$EXAMPLE_PATH" "${EXTRA_ARGS[@]}" 2>&1 | tee "$RUN_LOG"
RC=${PIPESTATUS[0]}
set -e

if [[ "$RC" -ne 0 ]]; then
  echo "example exited $RC" >&2
  tail -80 "$RUN_LOG" >&2
  exit "$RC"
fi

# Workload device anchor: vLLM-ascend / trainer only print these when
# the process is on NPU. Setup-time torch.npu.is_available() is not enough.
if ! grep -Eiq 'device_config=npu|backend=hccl|Platform plugin ascend|npu:0|NPUCachingAllocator|current platform: npu' "$RUN_LOG"; then
  echo "missing NPU device anchor in $RUN_LOG after exit 0" >&2
  echo "---- last 80 lines ----" >&2
  tail -80 "$RUN_LOG" >&2
  exit 1
fi

echo "example ok with NPU device anchor"
