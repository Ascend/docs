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
# Hong Kong runners reach huggingface.co directly. A mirror 308-redirects
# namespace-less ids, and huggingface_hub does not follow that redirect.
# Xet stays on. vLLM is given a local snapshot below, so it must not
# take the ModelScope file-listing path.
export HF_ENDPOINT=https://huggingface.co
unset VLLM_USE_MODELSCOPE
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

eval "EXTRA_ARGS=( $(python "$PROJECT_ROOT/scripts/overlay_args.py") )"
echo "running $EXAMPLE_PATH with ${#EXTRA_ARGS[@]} overlay args"
if ((${#EXTRA_ARGS[@]})); then
  printf 'overlay arg: %q\n' "${EXTRA_ARGS[@]}"
fi

# CI-scale defaults for bash config vars. overlay_args still go to the
# train/evaluate CLI via "$@".
export MAX_SAMPLES="${MAX_SAMPLES:-8}"
export SEQ_LENGTH="${SEQ_LENGTH:-1024}"
export EPOCHS="${EPOCHS:-1}"
export CONCURRENCY="${CONCURRENCY:-4}"
export MAX_ANCHORS="${MAX_ANCHORS:-32}"

case "$EXAMPLE_REL" in
  examples/train/peagle_qwen3_8b_ultrachat_online_5k.sh)
    # Upstream uses cards 2,3 and 4,5. Remap onto the 4-device slice.
    export VLLM_GPUS="${VLLM_GPUS:-0,1}"
    export TRAIN_GPUS="${TRAIN_GPUS:-2,3}"
    ;;
esac

python "$PROJECT_ROOT/scripts/patch_example.py" "$EXAMPLE_PATH"

LAUNCH_VLLM="$TARGET_ROOT/scripts/launch_vllm.py"
if [[ -f "$LAUNCH_VLLM" ]]; then
  python "$PROJECT_ROOT/scripts/patch_launch_vllm.py" "$LAUNCH_VLLM"
elif [[ "$EXAMPLE_REL" == examples/train/* ]]; then
  echo "missing $LAUNCH_VLLM for train example" >&2
  exit 1
else
  echo "skipping launch_vllm patch (not present)"
fi

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
