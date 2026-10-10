#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <example-relpath>" >&2
  exit 2
fi

EXAMPLE_REL="$1"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
CI_OUTPUT_DIR="${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"
EXAMPLE_PATH="$TARGET_ROOT/$EXAMPLE_REL"

[[ -f "$EXAMPLE_PATH" ]] || { echo "example not found: $EXAMPLE_PATH" >&2; exit 1; }
mkdir -p "$CI_OUTPUT_DIR"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -c "import torch, torch_npu; print('NPU available:', torch.npu.is_available(), 'devices:', torch.npu.device_count())"

expand_overlay() {
  # The workflow serializes manifest.overlay_args as JSON. Expand each item
  # with shell quoting intact, then allow CI paths such as ${CI_OUTPUT_DIR}
  # to resolve only in this job's environment.
  python - <<'PY'
import json
import os
import shlex

raw = os.environ.get('OVERLAY_ARGS', '').strip()
if not raw or raw in ('null', '""'):
    raise SystemExit(0)
items = json.loads(raw)
if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
    raise SystemExit('OVERLAY_ARGS must be a JSON array of strings')
tokens = []
for item in items:
    tokens.extend(shlex.split(os.path.expandvars(item), posix=True))
print(' '.join(shlex.quote(token) for token in tokens))
PY
}

eval "EXTRA_ARGS=( $(expand_overlay) )"
cd "$TARGET_ROOT"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

case "$EXAMPLE_REL" in
  */run_*_no_trainer.py)
    # No-trainer examples use Accelerate to initialize their training process.
    accelerate launch "$EXAMPLE_PATH" "${EXTRA_ARGS[@]}"
    ;;
  *)
    python "$EXAMPLE_PATH" "${EXTRA_ARGS[@]}"
    ;;
esac
