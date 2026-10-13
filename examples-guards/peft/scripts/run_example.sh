#!/usr/bin/env bash
# Run one peft example from a CI working copy of the target tree.
# $1 is the manifest entry path. EXEC, when set, names the launchable
# file relative to the target root; otherwise path itself must be a
# launchable file. Overlay CLI args come from OVERLAY_ARGS (JSON array,
# possibly []). Shell examples that do not pass "$@" get it attached in
# this working copy only (last command line). Never git add/commit/push.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <example-relpath>" >&2
  exit 2
fi

EXAMPLE_REL="$1"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
CI_OUTPUT_DIR="${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"

HERE=$(cd "$(dirname "$0")" && pwd)

EXAMPLE_PATH="$TARGET_ROOT/$EXAMPLE_REL"
[[ -e "$EXAMPLE_PATH" ]] || { echo "example not found: $EXAMPLE_PATH" >&2; exit 1; }

# Resolve the launchable file: EXEC (relative to the target root) when
# set, otherwise path itself.
if [[ -n "${EXEC:-}" ]]; then
  LAUNCH_PATH="$TARGET_ROOT/$EXEC"
else
  LAUNCH_PATH="$EXAMPLE_PATH"
fi
if [[ ! -f "$LAUNCH_PATH" ]]; then
  echo "launchable file not found: $LAUNCH_PATH (directory examples need an exec field)" >&2
  exit 1
fi

mkdir -p "$CI_OUTPUT_DIR"

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  PYTHON=python
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh
# bnb 0.50.2 的 dequantize 函数会触发 torch 2.12 dynamo 反复重编译
# （73s/step → 3s/step）；其余条目不依赖 dynamo 编译路径，全局关闭
# 无副作用。
export TORCHDYNAMO_DISABLE=1
python -c "import torch, torch_npu; print('NPU available:', torch.npu.is_available(), 'devices:', torch.npu.device_count())"

expand_overlay() {
  # The workflow serializes manifest.overlay_args as JSON. Expand each
  # item with shell quoting intact, then allow CI paths such as
  # ${CI_OUTPUT_DIR} / ${SFT_MODEL_PATH} to resolve only in this job's
  # environment.
  "$PYTHON" "$HERE/expand_overlay.py"
}

eval "EXTRA_ARGS=( $(expand_overlay) )"

echo "running $LAUNCH_PATH with ${#EXTRA_ARGS[@]} overlay args"
if ((${#EXTRA_ARGS[@]})); then
  printf 'overlay arg: %q\n' "${EXTRA_ARGS[@]}"
fi

ensure_passthrough() {
  # Shell examples that already forward "$@" need no patch; otherwise
  # attach it in this CI working copy only, on the last non-comment
  # line (the tail of the example's main command). Python entry points
  # take the overlay args directly on their own command line.
  local script="$1"
  [[ "$script" == *.sh ]] || return 0
  if grep -qE '"\$@"' "$script"; then
    echo "example already has \"\$@\"; skipping patch"
    return
  fi
  "$PYTHON" "$HERE/ensure_passthrough.py" "$script"
}

ensure_passthrough "$LAUNCH_PATH"


# Install ci_sitecustomize.py as sitecustomize.py so every Python
# process the example starts picks up the four CI patches (CUDA->NPU,
# local-file load_dataset, fp16 neutralizer, wandb neutralizer; see the
# docstring of ci_sitecustomize.py). SKIP_TRANSFER_TO_NPU, set below for
# the bitsandbytes examples, turns off the first one.
prepare_shims() {
  local shim_dir="$GITHUB_WORKSPACE/ci_patch"
  mkdir -p "$shim_dir"
  cp "$HERE/ci_sitecustomize.py" "$shim_dir/sitecustomize.py"
  export PYTHONPATH="$shim_dir:${PYTHONPATH:-}"
}

# bitsandbytes examples must run without transfer_to_npu: bnb 0.50.2's
# 4-bit/8-bit work via the default CPU backend, but transfer_to_npu makes
# torch.cuda.is_available()=True which routes bnb to its CUDA backend and
# crashes on torch._C._cuda_getCurrentRawStream (torch is +cpu-built).
case "$EXAMPLE_REL" in
  examples/fp4_finetuning/*|examples/int8_training/*|examples/loftq_finetuning/*|examples/KappaTune/*|examples/arrow_multitask/*)
    export SKIP_TRANSFER_TO_NPU=1
    ;;
esac

prepare_shims

# run_peft.sh invokes `python train.py` with a path relative to its own
# directory, so shell examples run with cwd = the example's directory;
# python entry points run with cwd = the target root.
case "$LAUNCH_PATH" in
  *.sh)
    cd "$(dirname "$LAUNCH_PATH")"
    bash "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
    ;;
  *)
    cd "$TARGET_ROOT"
    python "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
    ;;
esac
