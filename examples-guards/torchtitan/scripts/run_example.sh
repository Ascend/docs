#!/usr/bin/env bash
# Run one torchtitan example from a CI working copy of the target tree.
#
# Usage: run_example.sh <example-relpath>
#
# Env (provided by the examples engine):
#   EXAMPLE_PATH     the manifest entry path (same as $1)
#   EXAMPLE_SOURCE   "project" (paths live in examples-guards/torchtitan/) or
#                    "upstream" (paths live in the target checkout)
#   TARGET_ROOT      checkout of pytorch/torchtitan
#   PROJECT_ROOT     examples-guards/torchtitan in this repo
#   OVERLAY_ARGS     JSON array of CLI args for the example
#   CI_OUTPUT_DIR    writable output dir
#
# Only the CI working copy is touched; never git add/commit/push.
set -euo pipefail

[[ $# -ne 1 ]] && { echo "usage: $0 <example-relpath>" >&2; exit 2; }

EXAMPLE_REL="$1"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
CI_OUTPUT_DIR="${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"
mkdir -p "$CI_OUTPUT_DIR"

case "${EXAMPLE_SOURCE:-upstream}" in
    project) EXAMPLE_PATH="$PROJECT_ROOT/$EXAMPLE_REL" ;;
    upstream) EXAMPLE_PATH="$TARGET_ROOT/$EXAMPLE_REL" ;;
    *)
        echo "unknown EXAMPLE_SOURCE: ${EXAMPLE_SOURCE}" >&2
        exit 2
        ;;
esac
[[ -f "$EXAMPLE_PATH" ]] || { echo "example not found: $EXAMPLE_PATH" >&2; exit 1; }

# OVERLAY_ARGS (JSON array) -> shell array; each item is split with
# shlex and expanded for ${CI_OUTPUT_DIR} / ${FIXTURE_DIR} style vars.
expand_overlay() {
    python - <<'PY'
import json
import os
import shlex

raw = os.environ.get('OVERLAY_ARGS', '').strip()
if not raw or raw in ('null', '""'):
    raise SystemExit(0)
items = json.loads(raw)
if not isinstance(items, list):
    raise SystemExit('OVERLAY_ARGS must be a JSON array')
tokens = []
for item in items:
    if not isinstance(item, str) or not item.strip():
        raise SystemExit('items must be non-empty strings')
    tokens.extend(shlex.split(os.path.expandvars(item), posix=True))
print(' '.join(shlex.quote(t) for t in tokens))
PY
}
eval "EXTRA_ARGS=( $(expand_overlay) )"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -c "import torch, torch_npu; print('NPU available:', torch.npu.is_available(), 'devices:', torch.npu.device_count())"

# The launchers are self-contained (they set PYTHONPATH for the override
# module and cd into the checkout, since torchtitan resolves
# ./tests/assets/tokenizer relative to it).
bash "$EXAMPLE_PATH" "${EXTRA_ARGS[@]}"
