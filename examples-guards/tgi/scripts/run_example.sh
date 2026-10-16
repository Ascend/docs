#!/usr/bin/env bash
# TGI examples guard: run one serving example.
#
# $1 is the manifest entry path (the scenario script itself, relative
# to the target root). The script boots the service via the fork's
# start-tgi.sh, sends requests, asserts the replies and tears the
# service down; its exit code is the result. Never git add/commit/push.
#
# Env (provided by the engine, per the project script contract):
#   TARGET_ROOT / CI_OUTPUT_DIR / OVERLAY_ARGS (JSON array, possibly [])
#   ASCEND_RT_VISIBLE_DEVICES (manifest npu_devices)
set -euo pipefail

if [[ $# -ne 1 ]]; then
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

# The manifest never sets EXEC / LAUNCHER for tgi (the path is the
# entry script itself); reject them loudly instead of silently ignoring.
if [[ -n "${EXEC:-}" ]]; then
    echo "EXEC is not supported by the tgi run script (path is the script)" >&2
    exit 2
fi
if [[ -n "${LAUNCHER:-}" ]]; then
    echo "LAUNCHER is not supported by the tgi run script" >&2
    exit 2
fi

if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
else
    PYTHON=python
fi

# CANN toolkit env: the scenario's start-tgi.sh / run-npu.sh need
# ASCEND_* and the toolkit libs on LD_LIBRARY_PATH.
for candidate in /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/cann/set_env.sh; do
    if [[ -f "$candidate" ]]; then
        source "$candidate"
        break
    fi
done

mkdir -p "$CI_OUTPUT_DIR"

# The workflow serializes manifest.overlay_args as JSON. Expand each
# item with shell quoting intact, then allow CI paths such as
# ${CI_OUTPUT_DIR} to resolve only in this job's environment (same
# expander as the accelerate run script).
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
    if not isinstance(item, str) or not item.strip():
        raise SystemExit('OVERLAY_ARGS items must be non-empty strings')
    tokens.extend(shlex.split(os.path.expandvars(item), posix=True))
print(' '.join(shlex.quote(token) for token in tokens))
PY
}

eval "EXTRA_ARGS=( $(expand_overlay) )"

echo "running $EXAMPLE_PATH with ${#EXTRA_ARGS[@]} overlay args"
if ((${#EXTRA_ARGS[@]})); then
    printf 'overlay arg: %q\n' "${EXTRA_ARGS[@]}"
fi

cd "$TARGET_ROOT"
CI_OUTPUT_DIR="$CI_OUTPUT_DIR" bash "$EXAMPLE_PATH" "${EXTRA_ARGS[@]}"
