#!/usr/bin/env bash
# Run one cache-dit example from a CI working copy of the target tree.
# Overlay CLI args come from OVERLAY_ARGS (JSON array, as emitted by
# toJSON(matrix.example.overlay_args)); cache-dit examples parse sys.argv
# themselves (argparse), so args are passed straight through.
# EXEC overrides the launcher command when set (e.g. torchrun).
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <example-relpath>" >&2
  exit 2
fi

EXAMPLE_REL="$1"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
CI_OUTPUT_DIR="${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"

EXAMPLE_PATH="$TARGET_ROOT/$EXAMPLE_REL"

if [[ ! -f "$EXAMPLE_PATH" ]]; then
  echo "example not found: $EXAMPLE_PATH" >&2
  exit 1
fi

mkdir -p "$CI_OUTPUT_DIR"

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  PYTHON=python
fi

expand_overlay() {
  "$PYTHON" - <<'PY'
import json
import os
import shlex

raw = os.environ.get('OVERLAY_ARGS', '').strip()
if not raw or raw in ('null', '""'):
    raise SystemExit(0)
try:
    items = json.loads(raw)
except json.JSONDecodeError as exc:
    raise SystemExit(f'OVERLAY_ARGS is not valid JSON: {exc}') from exc
if items in (None, ''):
    raise SystemExit(0)
if not isinstance(items, list):
    raise SystemExit(
        f'OVERLAY_ARGS must be a JSON array, got {type(items).__name__}')
tokens = []
for item in items:
    if not isinstance(item, str):
        raise SystemExit(
            f'OVERLAY_ARGS items must be strings, got {type(item).__name__}')
    tokens.extend(shlex.split(os.path.expandvars(item), posix=True))
print(' '.join(shlex.quote(token) for token in tokens))
PY
}

eval "EXTRA_ARGS=( $(expand_overlay) )"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

# Run from the target repo root so relative paths (generated image
# outputs, example data) resolve consistently.
cd "$TARGET_ROOT"

echo "Running example: $EXAMPLE_PATH"
echo "With overlay args:"
if ((${#EXTRA_ARGS[@]})); then
  printf '  %q\n' "${EXTRA_ARGS[@]}"
fi
echo "Using NPU devices: ${ASCEND_RT_VISIBLE_DEVICES}"

if [[ -n "${EXEC:-}" ]]; then
  eval "${EXEC} ${EXTRA_ARGS[*]}"
else
  "$PYTHON" "$EXAMPLE_PATH" "${EXTRA_ARGS[@]}"
fi
