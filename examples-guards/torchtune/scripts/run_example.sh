#!/usr/bin/env bash
# Run one torchtune example from a CI working copy of the target tree.
# $1 is the manifest entry path. EXEC, when set, names the launchable
# file relative to the target root; otherwise path itself must be a
# launchable file. Overlay CLI args come from OVERLAY_ARGS (JSON array,
# possibly []). torchtune recipes are pure Python with @config.parse
# decorator, so each gets a single `python <path> "$@"` invocation
# (cwd = target root, so `--config recipes/configs/<x>.yaml` resolves
# relative to the recipe's repo root, matching the engine contract).
# Never git add/commit/push.
set -o pipefail
# NOTE: do NOT use `set -u` here; /usr/local/Ascend/nnal/atb/set_env.sh
# references `$ZSH_VERSION` unquoted, which fails under bash's nounset
# (empirically reproduced on hdc-stable-npu-3 with CANN 9.1.0 —
# `ZSH_VERSION: unbound variable` aborts env.sh source). Ascend env
# handling stays unconditional but errors get piped to /dev/null.
set -e

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <example-relpath>" >&2
  exit 2
fi

EXAMPLE_REL="$1"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
CI_OUTPUT_DIR="${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"

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
python -c "import torch, torch_npu; print('NPU available:', torch.npu.is_available(), 'devices:', torch.npu.device_count())"

expand_overlay() {
  # The workflow serializes manifest.overlay_args as JSON. Expand each
  # item with shell quoting intact, then allow CI paths such as
  # ${CI_OUTPUT_DIR} / ${TT_MODEL_PATH} / ${TARGET_ROOT} to resolve
  # only in this job's environment.
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

# Distributed recipes gate on is_distributed() (RANK env) and their
# init_process_group rendezvous via env://. Without these exports the
# leg dies either with
#   ValueError: ... environment variable RANK expected, but not set
# (full_finetune_distributed, init_process_group directly) or with
#   RuntimeError: Distributed finetune recipe should be run via a
#   distributed launcher ...
# (is_distributed() guard). Export a 1-rank world. Single-device
# recipes must NOT see RANK: they would take the distributed branch
# too.
if [[ "$EXAMPLE_REL" == *distributed* ]]; then
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  export MASTER_PORT="${MASTER_PORT:-29500}"
  export RANK="${RANK:-0}"
  export WORLD_SIZE="${WORLD_SIZE:-1}"
  export LOCAL_RANK="${LOCAL_RANK:-0}"
fi

echo "running $LAUNCH_PATH with ${#EXTRA_ARGS[@]} overlay args"
if ((${#EXTRA_ARGS[@]})); then
  printf 'overlay arg: %q\n' "${EXTRA_ARGS[@]}"
fi

# torchtune recipes use @config.parse (TuneRecipeArgumentParser) which
# already does its own argv parsing inside the recipe's recipe_main.
cd "$TARGET_ROOT"
python "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
