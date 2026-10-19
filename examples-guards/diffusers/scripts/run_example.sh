#!/usr/bin/env bash
# Run one diffusers example from a CI working copy of the target tree.
# $1 is the manifest entry path. EXEC, when set, names the launchable
# file relative to the target root; otherwise path itself must be a
# launchable file. Overlay CLI args come from OVERLAY_ARGS (JSON array,
# possibly []).
#
# Python examples are launched with `accelerate launch` (the upstream
# entrypoint for the training scripts). Never git add/commit/push.
#
# Contract: examples-guards/README.md "项目数据" (scripts/run_example.sh).
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
  # ${CI_OUTPUT_DIR} / ${SD15_MODEL_PATH} to resolve only in this job's
  # environment.
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

# Visible NPU count for the multi-card launcher.
count_npus() {
  "$PYTHON" -c "import torch, torch_npu; print(torch.npu.device_count())"
}

# Materialize an accelerate+DeepSpeed launch config (num_processes = visible
# NPU count) from the project fixture fixtures/configs/ds_zero2.json.
prepare_deepspeed_configs() {
  local ds_name="$1"
  local ds_json="${PROJECT_ROOT:?PROJECT_ROOT is required}/fixtures/configs/$ds_name"
  local cfg_dir="$GITHUB_WORKSPACE/ci_patch"
  local cfg_file="$cfg_dir/${ds_name%.json}.yml"
  mkdir -p "$cfg_dir"
  local npus
  npus=$(count_npus)
  cat > "$cfg_file" <<EOF
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
  echo "deepspeed launch config: $cfg_file (ds: $ds_json, $npus npu(s))"
}

# Python entry points run with cwd = the target root (diffusers examples
# resolve their own paths relative to the repo root).
case "$LAUNCH_PATH" in
  *)
    cd "$TARGET_ROOT"
    # Examples import helper packages two ways: the module next to them (e.g.
    # examples/research_projects/ip_adapter imports `ip_adapter.ip_adapter`) and
    # the repo-root package namespace (e.g. the pixart scripts import
    # `examples.research_projects.pixart.controlnet_pixart_alpha`). Put both the
    # example's own directory and the repo root on PYTHONPATH (`python script.py`
    # does not add the cwd, only the script's directory).
    export PYTHONPATH="$(dirname "$LAUNCH_PATH"):$TARGET_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    case "${LAUNCHER:-}" in
      accelerate-deepspeed)
        prepare_deepspeed_configs ds_zero2.json
        "$PYTHON" -m accelerate.commands.launch \
          --config_file "$GITHUB_WORKSPACE/ci_patch/ds_zero2.yml" \
          "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
        ;;
      "")
        # Training examples expect the upstream entrypoint. Single process;
        # the script's own --mixed_precision controls the precision, so the
        # launcher's is pinned to `no` to avoid a conflict.
        accelerate launch --num_processes 1 --num_machines 1 \
          --mixed_precision no "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
        ;;
      *)
        echo "unknown LAUNCHER: $LAUNCHER (supported: accelerate-deepspeed)" >&2
        exit 2
        ;;
    esac
    ;;
esac
