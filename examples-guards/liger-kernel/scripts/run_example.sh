#!/usr/bin/env bash
# Run one supported Liger-Kernel example, unmodified.
#
# $1 is the manifest entry path (relative to EXAMPLES_ROOT/TARGET_ROOT).
# Python entries run from their example directory for sibling imports.
# run_qwen.sh fixes a 7B model and four processes, so we reproduce its
# torchrun/FSDP recipe with the CI-sized overlay on two NPU devices.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <example-relpath>" >&2
  exit 2
fi

EXAMPLE_REL="$1"
export EXAMPLE_REL
: "${TARGET_ROOT:?TARGET_ROOT is required}"
: "${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"
EXAMPLES_ROOT="${EXAMPLES_ROOT:-$TARGET_ROOT}"
export EXAMPLES_ROOT

case "$EXAMPLE_REL" in
  examples/huggingface/training.py|examples/huggingface/run_qwen.sh|examples/medusa/train.py|examples/huggingface/training_multimodal.py) ;;
  *)
    echo "unsupported Liger-Kernel example entry: $EXAMPLE_REL" >&2
    exit 2
    ;;
esac

entry_path="$EXAMPLES_ROOT/$EXAMPLE_REL"
if [[ ! -f "$entry_path" ]]; then
  echo "example not found: $entry_path" >&2
  exit 1
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh
if [[ "$EXAMPLE_REL" == examples/huggingface/run_qwen.sh ]]; then
  export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"
else
  export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
fi
mkdir -p "$CI_OUTPUT_DIR"

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  PYTHON=python
fi

# Expand OVERLAY_ARGS (JSON array of CLI strings) with env vars applied.
expand_overlay() {
  "$PYTHON" - <<'PY'
import json
import os
import shlex

raw = os.environ.get("OVERLAY_ARGS", "").strip()
if not raw or raw in ("null", '""'):
    raise SystemExit(0)
try:
    items = json.loads(raw)
except json.JSONDecodeError as exc:
    raise SystemExit(f"OVERLAY_ARGS is not valid JSON: {exc}") from exc
if items in (None, ""):
    raise SystemExit(0)
if not isinstance(items, list):
    raise SystemExit(f"OVERLAY_ARGS must be a JSON array, got {type(items).__name__}")
tokens = []
for item in items:
    if not isinstance(item, str):
        raise SystemExit(f"OVERLAY_ARGS items must be strings, got {type(item).__name__}")
    tokens.extend(shlex.split(os.path.expandvars(item), posix=True))
print(" ".join(shlex.quote(token) for token in tokens))
PY
}

eval "EXTRA_ARGS=( $(expand_overlay) )"
echo "running $EXAMPLE_REL with ${#EXTRA_ARGS[@]} overlay args"
if ((${#EXTRA_ARGS[@]})); then
  printf 'overlay arg: %q\n' "${EXTRA_ARGS[@]}"
fi

# Fail before the run if the stack cannot see an NPU; a silent CPU run
# would still "pass" the example's own asserts.
"$PYTHON" - <<'PY'
import torch
import torch_npu

import os

required = 2 if os.environ["EXAMPLE_REL"] == "examples/huggingface/run_qwen.sh" else 1
count = torch.npu.device_count()
if not torch.npu.is_available() or count < required:
    raise SystemExit(f"{os.environ['EXAMPLE_REL']} requires {required} NPU devices, found {count}")
print(f"NPU devices visible: {count} (required: {required})")
PY

entry_dir="$(dirname "$entry_path")"
cd "$entry_dir"

export PYTHONPATH="$entry_dir:${PYTHONPATH:-}"

# Capture the example's own output to a file we own and replay it to stdout.
# The engine's outer tee writes train.log through a block-buffered pipe, so
# reading train.log from inside this script would race; a redirect closes
# the file before we inspect it.
run_log="$CI_OUTPUT_DIR/example_run.log"
if [[ "$EXAMPLE_REL" == examples/huggingface/run_qwen.sh ]]; then
  command=("$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 training.py)
else
  command=("$PYTHON" "$(basename "$entry_path")")
fi
set +e
"${command[@]}" "${EXTRA_ARGS[@]}" >"$run_log" 2>&1
run_status=$?
set -e
cat "$run_log"
if [[ $run_status -ne 0 ]]; then
  echo "example exited with status $run_status" >&2
  exit $run_status
fi

# Post-run proof that the example really trained instead of exiting early:
# the trainer must have created its output_dir, and the captured output must
# contain a training loss line. medusa rewrites --output_dir by appending a
# header summary, so match on the prefix rather than an exact path.
"$PYTHON" - "$EXAMPLE_REL" "$run_log" <<'PY'
import os
import re
import sys
from pathlib import Path

entry, run_log = sys.argv[1], sys.argv[2]
out = Path(os.environ["CI_OUTPUT_DIR"])
prefix = {
    "examples/huggingface/training.py": "hf_trainer",
    "examples/huggingface/run_qwen.sh": "hf_fsdp",
    "examples/medusa/train.py": "medusa",
    "examples/huggingface/training_multimodal.py": "multimodal",
}[entry]
matches = sorted(p for p in out.glob(f"{prefix}*") if p.is_dir())
if not matches:
    raise SystemExit(
        f"{entry}: no trainer output directory matching {out}/{prefix}* - the example did not train")
output_dir = matches[0]

body = Path(run_log).read_text(encoding="utf-8", errors="replace")
loss_lines = [line for line in body.splitlines() if re.search(r"'loss'|train_loss|\bloss=", line)]
if not loss_lines:
    raise SystemExit(f"{entry}: no training loss line in {run_log} - the trainer did not log a step")

files = sorted(str(path.relative_to(out)) for path in output_dir.rglob("*") if path.is_file())
print(f"{entry}: output dir {output_dir.name}, {len(loss_lines)} loss line(s)")
print(f"{entry}: last loss line: {loss_lines[-1].strip()[:140]}")
print(f"{entry}: artifacts: {files[:8]}")
print(f"NPU example passed: {entry}")
PY
