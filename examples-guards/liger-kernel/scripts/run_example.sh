#!/usr/bin/env bash
# Run one supported Liger-Kernel example, unmodified.
# $1 is the manifest entry path (relative to EXAMPLES_ROOT/TARGET_ROOT).
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
  examples/huggingface/training.py|\
  examples/huggingface/training_multimodal.py|\
  examples/huggingface/run_qwen.sh|\
  examples/huggingface/run_gemma.sh|\
  examples/huggingface/run_llama.sh|\
  examples/huggingface/run_qwen2_vl.sh|\
  examples/huggingface/run_benchmarks.sh|\
  examples/medusa/train.py|\
  examples/alignment/run_orpo.py) ;;
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

# NPU budget per entry: shell launchers need 4 devices, run_qwen.sh 2, the rest 1.
case "$EXAMPLE_REL" in
  examples/huggingface/run_qwen.sh)
    default_devices="0,1"
    required_npus=2
    ;;
  examples/huggingface/run_gemma.sh|examples/huggingface/run_llama.sh|examples/huggingface/run_qwen2_vl.sh|examples/huggingface/run_benchmarks.sh)
    default_devices="0,1,2,3"
    required_npus=4
    ;;
  *)
    default_devices="0"
    required_npus=1
    ;;
esac
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-$default_devices}"
mkdir -p "$CI_OUTPUT_DIR"

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  PYTHON=python
fi

# Expand OVERLAY_ARGS (JSON array of CLI strings).
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

# Fail before the run if the stack cannot see enough NPUs.
REQUIRED_NPUS="$required_npus" "$PYTHON" - <<'PY'
import os

import torch
import torch_npu

required = int(os.environ["REQUIRED_NPUS"])
count = torch.npu.device_count()
if not torch.npu.is_available() or count < required:
    raise SystemExit(f"{os.environ['EXAMPLE_REL']} requires {required} NPU devices, found {count}")
print(f"NPU devices visible: {count} (required: {required})")
PY

entry_dir="$(dirname "$entry_path")"
cd "$entry_dir"

export PYTHONPATH="$entry_dir:${PYTHONPATH:-}"

# run_orpo.py hardcodes a gated HF repo id; setup mirrored it into the caches.
if [[ "$EXAMPLE_REL" == examples/alignment/run_orpo.py ]]; then
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
fi

# Capture the example's output to a file, then replay it to stdout.
run_log="$CI_OUTPUT_DIR/example_run.log"
case "$EXAMPLE_REL" in
  examples/huggingface/run_qwen.sh)
    command=("$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 training.py)
    ;;
  *.sh)
    command=(bash "$(basename "$entry_path")")
    ;;
  *)
    command=("$PYTHON" "$(basename "$entry_path")")
    ;;
esac
set +e
"${command[@]}" "${EXTRA_ARGS[@]}" >"$run_log" 2>&1
run_status=$?
set -e
cat "$run_log"
if [[ $run_status -ne 0 ]]; then
  echo "example exited with status $run_status" >&2
  exit $run_status
fi

# Post-run proof: the output directory exists and a loss line was logged.
case "$EXAMPLE_REL" in
  examples/huggingface/training.py)
    out_root_kind=ci; out_name=hf_trainer; need_loss=1 ;;
  examples/huggingface/run_qwen.sh)
    out_root_kind=ci; out_name=hf_fsdp; need_loss=1 ;;
  examples/medusa/train.py)
    out_root_kind=ci; out_name=medusa; need_loss=1 ;;
  examples/huggingface/training_multimodal.py)
    out_root_kind=ci; out_name=multimodal; need_loss=1 ;;
  examples/alignment/run_orpo.py)
    out_root_kind=example; out_name=Llama3.2_1B_Instruct; need_loss=1 ;;
  examples/huggingface/run_gemma.sh|examples/huggingface/run_llama.sh)
    out_root_kind=example; out_name=alpaca_finetuning; need_loss=1 ;;
  examples/huggingface/run_qwen2_vl.sh)
    out_root_kind=example; out_name=multimodal_finetuning; need_loss=1 ;;
  examples/huggingface/run_benchmarks.sh)
    # The benchmark launcher logs each run under results/, so only the output dir is checked.
    out_root_kind=example; out_name=model_output_dir; need_loss=0 ;;
esac
OUT_ROOT_KIND="$out_root_kind" OUT_NAME="$out_name" NEED_LOSS="$need_loss" ENTRY_DIR="$entry_dir" "$PYTHON" - "$EXAMPLE_REL" "$run_log" <<'PY'
import os
import re
import sys
from pathlib import Path

entry, run_log = sys.argv[1], sys.argv[2]
kind = os.environ["OUT_ROOT_KIND"]
root = Path(os.environ["CI_OUTPUT_DIR"]) if kind == "ci" else Path(os.environ["ENTRY_DIR"])
name = os.environ["OUT_NAME"]
need_loss = os.environ["NEED_LOSS"] == "1"

matches = sorted(p for p in root.glob(name + "*") if p.is_dir())
if not matches:
    raise SystemExit(
        f"{entry}: no trainer output directory matching {root}/{name}* - the example did not train")
output_dir = matches[0]

body = Path(run_log).read_text(encoding="utf-8", errors="replace")
loss_lines = [line for line in body.splitlines() if re.search(r"'loss'|train_loss|\bloss=", line)]
if need_loss and not loss_lines:
    raise SystemExit(f"{entry}: no training loss line in {run_log} - the trainer did not log a step")

files = sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file())
print(f"{entry}: output dir {output_dir}, {len(loss_lines)} loss line(s)")
if loss_lines:
    print(f"{entry}: last loss line: {loss_lines[-1].strip()[:140]}")
print(f"{entry}: artifacts: {files[:8]}")
print(f"NPU example passed: {entry}")
PY
