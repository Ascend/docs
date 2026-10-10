#!/usr/bin/env bash
# Run one xllm example from a CI working copy of the target tree.
# Overlay CLI args come from OVERLAY_ARGS (JSON array).
# The example scripts don't have "$@" passthrough, so we patch them.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <example-relpath>" >&2
  exit 2
fi

EXAMPLE_REL="$1"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
FIXTURE_DIR="${FIXTURE_DIR:?FIXTURE_DIR is required}"
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

ensure_passthrough() {
  local script="$1"
  if grep -qE '"\$@"' "$script"; then
    echo "example already has \"\$@\"; skipping patch"
    return
  fi
  # xllm example scripts call ArgumentParser().parse_args() themselves, so they
  # already consume sys.argv (incl. the overlay args). Patching "$@" onto the
  # last Python statement would produce invalid syntax — skip them.
  if grep -qE 'parse_args\(' "$script"; then
    echo "example uses argparse (reads sys.argv); skipping patch"
    return
  fi
  "$PYTHON" - "$script" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding='utf-8')

# Find the last non-comment, non-empty line (the actual command)
lines = text.splitlines(keepends=True)
for i in range(len(lines) - 1, -1, -1):
    stripped = lines[i].strip()
    if stripped and not stripped.startswith('#'):
        raw = lines[i]
        newline = ''
        if raw.endswith('\r\n'):
            newline = '\r\n'
            raw = raw[:-2]
        elif raw.endswith('\n'):
            newline = '\n'
            raw = raw[:-1]
        # Append "$@" before the newline
        lines[i] = raw.rstrip() + ' "$@"' + newline
        path.write_text(''.join(lines), encoding='utf-8')
        print(f'patched {path} to pass "$@" on last command line')
        raise SystemExit(0)

raise SystemExit(f'{path}: cannot find a command line to attach "$@"')
PY
}

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

echo "running $EXAMPLE_PATH with ${#EXTRA_ARGS[@]} overlay args"
if ((${#EXTRA_ARGS[@]})); then
  printf 'overlay arg: %q\n' "${EXTRA_ARGS[@]}"
  ensure_passthrough "$EXAMPLE_PATH"
fi

cd /tmp
# Export so the example script sees the same cache/output contract.
export FIXTURE_DIR CI_OUTPUT_DIR ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
# Source CANN env in the same shell that runs the example
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python "$EXAMPLE_PATH" "${EXTRA_ARGS[@]}"
