#!/usr/bin/env bash
# Run one llama.cpp example from the CI working copy.
# Overlay CLI args come from OVERLAY_ARGS, a JSON array. Never
# git add, commit, or push the working copy.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <example-relpath>" >&2
  exit 2
fi

EXAMPLE_REL="$1"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
CI_OUTPUT_DIR="${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"
EXEC_REL="${EXEC:-}"
PROFILE="${PROFILE:-}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

EXAMPLE_PATH="$TARGET_ROOT/$EXAMPLE_REL"

if [[ ! -e "$EXAMPLE_PATH" ]]; then
  echo "example not found: $EXAMPLE_PATH" >&2
  exit 1
fi
if [[ -n "$EXEC_REL" ]]; then
  EXEC_PATH="$TARGET_ROOT/$EXEC_REL"
  if [[ ! -f "$EXEC_PATH" ]]; then
    echo "exec not found: $EXEC_PATH" >&2
    exit 1
  fi
elif [[ -d "$EXAMPLE_PATH" ]]; then
  echo "path is a directory; set exec on the manifest entry: $EXAMPLE_REL" >&2
  exit 1
fi

mkdir -p "$CI_OUTPUT_DIR"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export HF_ENDPOINT=https://huggingface.co
# Weights were resolved during setup. Do not let the run step re-check
# them against the network.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if ! overlay=$(python3 "$HERE/expand_overlay.py"); then
  exit 1
fi
eval "EXTRA_ARGS=( $overlay )"

if [[ -n "$EXEC_REL" ]]; then
  echo "running $EXEC_PATH for $EXAMPLE_REL with ${#EXTRA_ARGS[@]} overlay args"
else
  echo "running $EXAMPLE_PATH with ${#EXTRA_ARGS[@]} overlay args"
fi
if ((${#EXTRA_ARGS[@]})); then
  printf 'overlay arg: %q\n' "${EXTRA_ARGS[@]}"
fi

cd "$TARGET_ROOT"
export CI_OUTPUT_DIR ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

assert_cann_used() {
  local run_log="$1"
  # cpu builds are not an NPU result.
  if [[ "$PROFILE" == "cpu" ]]; then
    return 0
  fi
  if ! grep -q 'CANN[0-9]' "$run_log"; then
    echo "run used no CANN device; check the container card mounts" >&2
    exit 1
  fi
}

if [[ -n "$EXEC_REL" ]]; then
  RUN_LOG="$CI_OUTPUT_DIR/$(basename "$EXEC_REL").log"
  case "$EXAMPLE_REL" in
    examples/gguf)
      "$EXEC_PATH" "$CI_OUTPUT_DIR/ci-demo.gguf" w 2>&1 | tee "$RUN_LOG"
      "$EXEC_PATH" "$CI_OUTPUT_DIR/ci-demo.gguf" r 2>&1 | tee -a "$RUN_LOG"
      ;;
    *)
      "$EXEC_PATH" "${EXTRA_ARGS[@]}" 2>&1 | tee "$RUN_LOG"
      ;;
  esac
  assert_cann_used "$RUN_LOG"
  exit 0
fi

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  PYTHON=python
fi
case "$EXAMPLE_PATH" in
  *.sh)
    bash "$EXAMPLE_PATH" "${EXTRA_ARGS[@]}"
    ;;
  *.py)
    "$PYTHON" "$EXAMPLE_PATH" "${EXTRA_ARGS[@]}"
    ;;
  *)
    echo "unsupported example type: $EXAMPLE_PATH" >&2
    exit 1
    ;;
esac
