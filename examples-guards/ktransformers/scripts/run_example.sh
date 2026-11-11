#!/usr/bin/env bash
# Run the project-owned llamafile forward and require its NPU anchor line.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 example/llamafile_npu_forward.py" >&2
  exit 2
fi

if [[ "${EXAMPLE_SOURCE:-}" != "project" ]]; then
  echo "ktransformers guarded example is project-owned; EXAMPLE_SOURCE must be project" >&2
  exit 1
fi

EXAMPLE_REL="$1"
case "$EXAMPLE_REL" in
  example/llamafile_npu_forward.py) ;;
  *)
    echo "unsupported ktransformers example: $EXAMPLE_REL" >&2
    exit 1
    ;;
esac

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
export PATH="/usr/local/sbin:$PATH"
export PYTHONNOUSERSITE=1
# shellcheck disable=SC1091
source /usr/local/Ascend/ascend-toolkit/set_env.sh

SCRIPT="$PROJECT_ROOT/$EXAMPLE_REL"
if [[ ! -f "$SCRIPT" ]]; then
  echo "example script not found: $SCRIPT" >&2
  exit 1
fi

LOG="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}/output/example.log"
mkdir -p "$(dirname "$LOG")"
python "$SCRIPT" | tee "$LOG"
grep -F 'kt-kernel ascend callback forward device=npu:' "$LOG"
