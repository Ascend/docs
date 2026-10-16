#!/usr/bin/env bash
# Run one accelerate example from a CI working copy of the target tree.
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
  # ${CI_OUTPUT_DIR} / ${TARGET_ROOT}/fixtures/... to resolve only in
  # this job's environment.
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

echo "running $LAUNCH_PATH with ${#EXTRA_ARGS[@]} overlay args"
if ((${#EXTRA_ARGS[@]})); then
  printf 'overlay arg: %q\n' "${EXTRA_ARGS[@]}"
fi

# Launcher modes for python entry points. LAUNCHER comes from the
# manifest entry (optional field, empty = default bare run):
#   torchrun — torch.distributed.run with --nproc_per_node = visible
#     NPU count; activates the multi-card paths (DDP wrap, comm hooks,
#     gather_for_metrics cross-rank aggregation) that bare `python`
#     silently skips on multi-card runners.
#   accelerate-deepspeed — exercises the DeepSpeed config branch by
#     launching via `accelerate launch --config_file` with a ZeRO-2
#     bf16 DeepSpeed json (fixtures/ds_zero2.json). The launch yml is
#     materialized in the CI working copy because num_processes follows
#     the visible NPU count (a2-1/a2-2 both use this entry).
count_npus() {
  "$PYTHON" -c "import torch, torch_npu; print(torch.npu.device_count())"
}

prepare_deepspeed_configs() {
  local ds_json="${FIXTURE_DIR:?FIXTURE_DIR is required}/ds_zero2.json"
  local cfg_dir="$GITHUB_WORKSPACE/ci_patch"
  mkdir -p "$cfg_dir"
  local npus
  npus=$(count_npus)
  cat > "$cfg_dir/accelerate-deepspeed.yml" <<EOF
compute_environment: LOCAL_MACHINE
deepspeed_config:
  deepspeed_config_file: $ds_json
  zero3_init_flag: false
distributed_type: DEEPSPEED
machine_rank: 0
main_training_function: main
num_machines: 1
num_processes: $npus
rdzv_backend: static
same_network: true
use_cpu: false
EOF
  echo "deepspeed launch config: $cfg_dir/accelerate-deepspeed.yml (ds: $ds_json, $npus npu(s))"
}

# Shell examples that invoke `python train.py` with a path relative to
# their own directory run with cwd = the example's directory; python
# entry points run with cwd = the target root.
case "$LAUNCH_PATH" in
  *.sh)
    cd "$(dirname "$LAUNCH_PATH")"
    bash "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
    ;;
  *)
    cd "$TARGET_ROOT"
    case "${LAUNCHER:-}" in
      accelerate-deepspeed)
        prepare_deepspeed_configs
        "$PYTHON" -m accelerate.commands.launch \
          --config_file "$GITHUB_WORKSPACE/ci_patch/accelerate-deepspeed.yml" \
          "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
        ;;
      torchrun)
        npus=$(count_npus)
        echo "torchrun launcher: --nproc_per_node $npus"
        "$PYTHON" -m torch.distributed.run \
          --nproc_per_node "$npus" \
          "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
        ;;
      "")
        "$PYTHON" "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
        ;;
      *)
        echo "unknown LAUNCHER: $LAUNCHER (supported: torchrun, accelerate-deepspeed)" >&2
        exit 2
        ;;
    esac
    ;;
esac