#!/usr/bin/env bash
# Run one ms-swift example from a CI working copy of the target tree.
# The example is run exactly as published, at the scale its author chose:
# no overlay args, no edit to the file. Never git add, commit, or push the
# working copy.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <example-relpath>" >&2
  exit 2
fi

EXAMPLE_REL="$1"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
CI_OUTPUT_DIR="${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"

# The upstream scripts end their command line without "$@", so overlay args
# could never reach them. Refuse them instead of dropping them silently.
overlay_args="${OVERLAY_ARGS:-[]}"
overlay_args="${overlay_args//[[:space:]]/}"
case "$overlay_args" in
  ''|'[]'|'null') ;;
  *)
    echo "ms-swift examples run unmodified; overlay_args are not supported: $overlay_args" >&2
    exit 2
    ;;
esac

EXAMPLE_PATH="$TARGET_ROOT/$EXAMPLE_REL"
if [[ ! -f "$EXAMPLE_PATH" ]]; then
  echo "example not found: $EXAMPLE_PATH" >&2
  exit 1
fi
mkdir -p "$CI_OUTPUT_DIR"
RUN_LOG="$CI_OUTPUT_DIR/run.log"

# A new shell per workflow step: the CANN environment from the setup step
# is gone. Without it the Megatron ranks fail with "No module named 'acl'"
# and vllm-ascend cannot load its custom op libraries.
set +u
# shellcheck disable=SC1091
source /usr/local/Ascend/ascend-toolkit/set_env.sh
if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/nnal/atb/set_env.sh
fi
set -u

export PYTHONNOUSERSITE=1
# Output goes through a pipe to tee. Flush every line so a hang shows
# where it stopped.
export PYTHONUNBUFFERED=1
# The model and datasets were put into the shared cache during setup.
# USE_HF makes ms-swift resolve ids against the HuggingFace cache layout
# (setup plants even ModelScope-only dataset ids there), and the offline
# switches keep the run from revalidating against the hub: several of
# the dataset ids do not exist on HuggingFace at all, and a slow download
# by every rank at once outlasts the collective timeout.
export USE_HF=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
# Same reason as in setup_example.sh: MindSpeed makes transformers log tens of
# thousands of warnings in every rank. Errors still print.
if [[ "${PROFILE:-}" == megatron ]]; then
  export TRANSFORMERS_VERBOSITY=error
fi

echo "running $EXAMPLE_PATH"

# The examples address their config files and swift/cli/*.py relative to
# the repository root.
cd "$TARGET_ROOT"
bash "$EXAMPLE_PATH" 2>&1 | tee "$RUN_LOG"

# Device anchor: ms-swift logs these lines only when it selected the NPU
# backend (swift/model/npu_patch/env.py, imported only when
# is_torch_npu_available() is true). A run that silently fell back to CPU
# would not print them.
if ! grep -qE 'Set HCCL_CONNECT_TIMEOUT=600 by default for NPU|Patched torch_npu getenv' "$RUN_LOG"; then
  echo "run printed no ms-swift NPU backend line; the NPU was not used" >&2
  exit 1
fi
