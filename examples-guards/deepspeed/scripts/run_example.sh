#!/usr/bin/env bash
# Run one example from a CI working copy of the examples tree.
# $1 is the manifest entry path. EXEC, when set, names the launchable file
# relative to the examples root; otherwise path itself must be launchable.
# Overlay CLI args come from OVERLAY_ARGS (JSON array). Never git add/commit/push.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <example-relpath>" >&2
  exit 2
fi

EXAMPLE_REL="$1"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
CI_OUTPUT_DIR="${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"

# Examples tree root: split mode sets EXAMPLES_ROOT (the DeepSpeedExamples
# checkout); non-split it falls back to TARGET_ROOT.
EXAMPLES_ROOT="${EXAMPLES_ROOT:-$TARGET_ROOT}"

EXAMPLE_PATH="$EXAMPLES_ROOT/$EXAMPLE_REL"
if [[ ! -e "$EXAMPLE_PATH" ]]; then
  echo "example not found: $EXAMPLE_PATH" >&2
  exit 1
fi

# Resolve the launchable file: EXEC (relative to the examples root) when set,
# otherwise path itself.
if [[ -n "${EXEC:-}" ]]; then
  LAUNCH_PATH="$EXAMPLES_ROOT/$EXEC"
else
  LAUNCH_PATH="$EXAMPLE_PATH"
fi
if [[ ! -f "$LAUNCH_PATH" ]]; then
  echo "launchable file not found: $LAUNCH_PATH" >&2
  exit 1
fi

mkdir -p "$CI_OUTPUT_DIR"

source /usr/local/Ascend/ascend-toolkit/set_env.sh

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

echo "running $LAUNCH_PATH with ${#EXTRA_ARGS[@]} overlay args"
if ((${#EXTRA_ARGS[@]})); then
  printf 'overlay arg: %q\n' "${EXTRA_ARGS[@]}"
fi

cd "$(dirname "$LAUNCH_PATH")"
export CI_OUTPUT_DIR
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

entry_key="${EXEC:-$EXAMPLE_REL}"
if [[ "$entry_key" == benchmarks/opsd/benchmark_hybrid_engine_rollout.py ]]; then
  echo 'unsupported: native NPU softmax_context interface mismatch (Run #30); upstream fix required' >&2
  exit 2
fi

require_visible_devices() {
  local required="$1"
  local default_devices="$2"
  local -a devices

  if [[ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
    export ASCEND_RT_VISIBLE_DEVICES="$default_devices"
  fi
  IFS=',' read -r -a devices <<< "$ASCEND_RT_VISIBLE_DEVICES"
  if ((${#devices[@]} < required)); then
    echo "insufficient NPU devices for $entry_key: required=$required visible=$ASCEND_RT_VISIBLE_DEVICES" >&2
    exit 1
  fi
  echo "NPU devices for $entry_key: required=$required visible=$ASCEND_RT_VISIBLE_DEVICES"
}

first_visible_devices() {
  local count="$1"
  local -a devices
  IFS=',' read -r -a devices <<< "$ASCEND_RT_VISIBLE_DEVICES"
  local selected="${devices[0]}"
  local index
  for ((index = 1; index < count; index++)); do
    selected+=",${devices[index]}"
  done
  printf '%s\n' "$selected"
}

master_port_for() {
  local offset="$1"
  local run_id="${GITHUB_RUN_ID:-0}"
  printf '%s\n' "$((20000 + run_id % 20000 + offset))"
}

require_overlay_path() {
  local flag="$1"
  local kind="$2"
  local index value=''
  for ((index = 0; index < ${#EXTRA_ARGS[@]}; index++)); do
    if [[ "${EXTRA_ARGS[index]}" == "$flag" ]]; then
      value="${EXTRA_ARGS[index + 1]:-}"
      break
    fi
  done
  if [[ -z "$value" ]] ||
     { [[ "$kind" == directory ]] && [[ ! -d "$value" ]]; } ||
     { [[ "$kind" == file ]] && [[ ! -f "$value" ]]; }; then
    echo "missing local $kind for $entry_key: $flag=${value:-<unset>}" >&2
    exit 1
  fi
}

run_cifar_moe() {
  require_visible_devices 2 '0,1'
  local devices
  devices="$(first_visible_devices 2)"
  echo "reproducing upstream run_ds_moe.sh on devices $devices"
  echo "disabling optional torch.compile for DeepSpeed MoE helpers; eager MoE/EP execution is preserved"
  TORCH_COMPILE_DISABLE=1 \
  ASCEND_RT_VISIBLE_DEVICES="$devices" \
  deepspeed \
    --master_port "$(master_port_for 2)" \
    --num_nodes 1 \
    --num_gpus 2 \
    --bind_cores_to_rank \
    "$EXAMPLES_ROOT/training/cifar/cifar10_deepspeed.py" \
    --deepspeed \
    --moe \
    --ep-world-size 2 \
    --num-experts 2 \
    --top-k 1 \
    --noisy-gate-policy RSample \
    --moe-param-group \
    "${EXTRA_ARGS[@]}"
}

run_cifar_prmoe() {
  require_visible_devices 2 '0,1'
  local devices
  devices="$(first_visible_devices 2)"
  echo "reproducing upstream run_ds_prmoe.sh on devices $devices: EP=2, experts=2/4, residual MoE"
  echo "disabling optional torch.compile for DeepSpeed MoE helpers; eager PR-MoE/EP execution is preserved"
  TORCH_COMPILE_DISABLE=1 \
  ASCEND_RT_VISIBLE_DEVICES="$devices" \
  deepspeed \
    --master_port "$(master_port_for 3)" \
    --num_nodes 1 \
    --num_gpus 2 \
    "$EXAMPLES_ROOT/training/cifar/cifar10_deepspeed.py" \
    --deepspeed \
    --moe \
    --ep-world-size 2 \
    --num-experts 2 4 \
    --top-k 1 \
    --mlp-type residual \
    --noisy-gate-policy RSample \
    --moe-param-group \
    "${EXTRA_ARGS[@]}"
}

run_hf_autotp() {
  require_visible_devices 8 '0,1,2,3,4,5,6,7'
  require_overlay_path --model_name_or_path directory
  require_overlay_path --data_path file
  local devices config_path
  devices="$(first_visible_devices 8)"
  config_path="$CI_OUTPUT_DIR/hf-autotp-config.json"
  "$PYTHON" - "$EXAMPLES_ROOT/training/tensor_parallel/hf_integration/configs/ds_config_temp.json" "$config_path" <<'PY'
import json
import sys
from pathlib import Path

template = Path(sys.argv[1]).read_text()
config = json.loads(template.replace('${zero_stage}', '0').replace('${autotp_size}', '8'))
if config['zero_optimization']['stage'] != 0 or config['tensor_parallel']['autotp_size'] != 8:
    raise SystemExit('HF AutoTP CI requires ZeRO=0 and AutoTP=8')
Path(sys.argv[2]).write_text(json.dumps(config, indent=2) + '\n')
PY
  # The upstream example caches tokenized data in its working directory and
  # unconditionally saves the final model. Keep all generated files in CI output.
  if [[ ! -s "$CI_OUTPUT_DIR/hf-autotp-work/dataset_dict.pkl" ]]; then
    echo "HF AutoTP shared tokenization cache missing; run ds_hf_autotp setup first" >&2
    exit 1
  fi
  mkdir -p "$CI_OUTPUT_DIR/hf-autotp-work"
  cd "$CI_OUTPUT_DIR/hf-autotp-work"
  echo "running HF Trainer AutoTP=8 on devices $devices with config $config_path"
  ASCEND_RT_VISIBLE_DEVICES="$devices" WANDB_MODE=disabled deepspeed \
    --master_port "$(master_port_for 28)" \
    --num_gpus 8 \
    --no_local_rank \
    "$LAUNCH_PATH" \
    --deepspeed "$config_path" \
    "${EXTRA_ARGS[@]}"
}

overlay_value() {
  local flag="$1" index
  for ((index = 0; index < ${#EXTRA_ARGS[@]}; index++)); do
    if [[ "${EXTRA_ARGS[index]}" == "$flag" ]]; then
      printf '%s\n' "${EXTRA_ARGS[index + 1]:-}"
      return 0
    fi
  done
  return 1
}

run_chat_prompt_eval() {
  require_visible_devices 1 '0'
  require_overlay_path --model_name_or_path_baseline directory
  local devices baseline checkpoint finetune
  devices="$(first_visible_devices 1)"
  baseline="$(overlay_value --model_name_or_path_baseline)"
  checkpoint="$CI_OUTPUT_DIR/prompt-eval-sft"
  finetune="$(overlay_value --model_name_or_path_finetune)"
  if [[ "$finetune" != "$checkpoint" ]]; then
    echo "prompt eval requires the same-job SFT checkpoint: expected=$checkpoint actual=$finetune" >&2
    exit 1
  fi
  if [[ -e "$checkpoint/config.json" ]] || [[ -e "$checkpoint/pytorch_model.bin" ]]; then
    echo "prompt eval checkpoint already exists; refusing to reuse stale training output: $checkpoint" >&2
    exit 1
  fi
  echo "training a genuine SFT checkpoint before baseline/finetuned prompt evaluation"
  ASCEND_RT_VISIBLE_DEVICES="$devices" deepspeed \
    --master_port "$(master_port_for 51)" --num_gpus 1 \
    "$EXAMPLES_ROOT/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/main.py" \
    --model_name_or_path "$baseline" --data_path local/jsonfile \
    --num_train_epochs 1 --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 --max_seq_len 128 \
    --zero_stage 2 --dtype bf16 --print_loss --deepspeed \
    --data_output_path "$CI_OUTPUT_DIR/prompt-eval-data" \
    --output_dir "$checkpoint"
  # save_hf_format in the unchanged SFT entry writes these exact two files.
  if [[ ! -s "$checkpoint/config.json" ]] || [[ ! -s "$checkpoint/pytorch_model.bin" ]]; then
    echo "SFT did not produce a loadable checkpoint for prompt eval: $checkpoint" >&2
    exit 1
  fi
  echo "evaluating baseline against the newly trained SFT checkpoint on NPU $devices"
  ASCEND_RT_VISIBLE_DEVICES="$devices" "$PYTHON" "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
}

run_chat_reward_eval() {
  require_visible_devices 1 '0'
  require_overlay_path --model_name_or_path directory
  local devices
  devices="$(first_visible_devices 1)"
  echo "running upstream reward-head initialization/forward smoke on NPU $devices; not trained-head restoration"
  ASCEND_RT_VISIBLE_DEVICES="$devices" "$PYTHON" "$LAUNCH_PATH" "${EXTRA_ARGS[@]}" \
    | tee "$CI_OUTPUT_DIR/reward-eval.log"
  "$PYTHON" - "$CI_OUTPUT_DIR/reward-eval.log" "$CI_OUTPUT_DIR/reward-eval-scores.json" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

scores = []
for line in Path(sys.argv[1]).read_text().splitlines():
    match = re.fullmatch(r'\s*(good_ans|bad_ans) score:\s*(\S+)\s*', line)
    if match:
        value = float(match.group(2))
        if not math.isfinite(value):
            raise SystemExit('reward evaluation produced a non-finite score')
        scores.append({'kind': match.group(1), 'score': value})
if [score['kind'] for score in scores] != ['good_ans', 'bad_ans'] * 2:
    raise SystemExit('reward evaluation must produce four scores for the two upstream preference pairs')
# An initialized reward head has no quality ordering guarantee; finite forward
# results, not good>bad, are the smoke acceptance criterion.
Path(sys.argv[2]).write_text(json.dumps(scores, indent=2, allow_nan=False) + '\n')
print('validated four finite reward scores (initialized-head forward smoke)')
PY
}

run_hf_bench_length() {
  require_visible_devices 8 '0,1,2,3,4,5,6,7'
  require_overlay_path --model_name_or_path directory
  require_overlay_path --data_path file
  local devices config_path
  devices="$(first_visible_devices 8)"
  config_path="$CI_OUTPUT_DIR/hf-bench-length-config.json"
  if [[ "$(overlay_value --model_max_length)" != 128 ]]; then
    echo "HF fixed-length CI recipe requires --model_max_length 128" >&2
    exit 1
  fi
  if [[ ! -s "$CI_OUTPUT_DIR/hf-bench-length-work/dataset_dict128.pkl" ]]; then
    echo "HF fixed-length shared tokenization cache missing; run ds_hf_bench_length setup first" >&2
    exit 1
  fi
  "$PYTHON" - "$EXAMPLES_ROOT/training/tensor_parallel/hf_integration/configs/ds_config_temp.json" "$config_path" <<'PY'
import json
import sys
from pathlib import Path

template = Path(sys.argv[1]).read_text()
config = json.loads(template.replace('${zero_stage}', '0').replace('${autotp_size}', '8'))
if config['zero_optimization']['stage'] != 0 or config['tensor_parallel']['autotp_size'] != 8:
    raise SystemExit('HF fixed-length CI requires ZeRO=0 and AutoTP=8')
Path(sys.argv[2]).write_text(json.dumps(config, indent=2) + '\n')
PY
  cd "$CI_OUTPUT_DIR/hf-bench-length-work"
  echo "running fixed-length HF Trainer AutoTP=8 on devices $devices"
  ASCEND_RT_VISIBLE_DEVICES="$devices" WANDB_MODE=disabled deepspeed \
    --master_port "$(master_port_for 61)" --num_gpus 8 --no_local_rank \
    "$LAUNCH_PATH" --deepspeed "$config_path" "${EXTRA_ARGS[@]}"
}

run_superoffload() {
  require_visible_devices 2 '0,1'
  require_overlay_path --model_name directory
  require_overlay_path --dataset_name directory
  local devices config_path
  devices="$(first_visible_devices 2)"
  config_path="$CI_OUTPUT_DIR/superoffload-config.json"
  "$PYTHON" - "$config_path" <<'PY'
import json
import sys
from pathlib import Path

# The unchanged entry creates DeepSpeedCPUAdam itself (LR=0.001); do not
# provide a duplicate optimizer or claim that the logging-only --lr sets it.
config = {
    'train_batch_size': 2,
    'train_micro_batch_size_per_gpu': 1,
    'gradient_accumulation_steps': 1,
    'bf16': {'enabled': True},
    'zero_optimization': {
        'stage': 3,
        'overlap_comm': False,
        'contiguous_gradients': True,
        'reduce_bucket_size': 65536,
        'stage3_prefetch_bucket_size': 65536,
        'stage3_param_persistence_threshold': 4096,
        'offload_param': {'device': 'cpu', 'pin_memory': False},
        'offload_optimizer': {'device': 'cpu', 'pin_memory': False},
    },
    'steps_per_print': 1,
    'wall_clock_breakdown': True,
}
Path(sys.argv[1]).write_text(json.dumps(config, indent=2) + '\n')
PY
  mkdir -p "$CI_OUTPUT_DIR/superoffload-work"
  cd "$CI_OUTPUT_DIR/superoffload-work"
  echo "running SuperOffload entry / ZeRO-Offload smoke on devices $devices; native super_offload is not enabled"
  ASCEND_RT_VISIBLE_DEVICES="$devices" WANDB_MODE=disabled deepspeed \
    --master_port "$(master_port_for 71)" --num_gpus 2 \
    "$LAUNCH_PATH" --deepspeed --deepspeed_config "$config_path" "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$CI_OUTPUT_DIR/superoffload.log"
  "$PYTHON" - "$CI_OUTPUT_DIR/superoffload.log" "$CI_OUTPUT_DIR/superoffload-losses.json" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

rank_zero_losses = {}
for line in Path(sys.argv[1]).read_text().splitlines():
    match = re.search(r'Step\s+(\d+)\s*\|\s*Loss:\s*(\S+)\s*\|', line)
    if not match:
        continue
    step, loss = int(match.group(1)), float(match.group(2))
    if not math.isfinite(loss):
        raise SystemExit(f'SuperOffload entry produced a non-finite loss at step {step}')
    # Upstream adds the timestamped handler only on rank 0. Its root logging
    # handler can repeat messages on other ranks, so tolerate duplicates and
    # check every reported loss, but require rank-0 evidence for all 3 steps.
    if ' - finetune_zero3 - INFO - ' in line:
        rank_zero_losses[step] = loss
if sorted(rank_zero_losses) != [1, 2, 3]:
    raise SystemExit('SuperOffload entry must report rank-0 finite losses for steps 1, 2 and 3')
report = [{'step': step, 'loss': rank_zero_losses[step]} for step in sorted(rank_zero_losses)]
Path(sys.argv[2]).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print('validated three finite SuperOffload entry train-step losses (ZeRO-Offload smoke)')
PY
}

run_variable_batch() {
  require_visible_devices 1 '0'
  local devices
  devices="$(first_visible_devices 1)"
  mkdir -p "$CI_OUTPUT_DIR/variable-batch-work"
  cd "$CI_OUTPUT_DIR/variable-batch-work"
  echo "running upstream dynamic sequence packing and LR scaling on device $devices"
  ASCEND_RT_VISIBLE_DEVICES="$devices" deepspeed \
    --master_port "$(master_port_for 31)" \
    --num_gpus 1 \
    --no_local_rank \
    "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
}

run_zenflow() {
  require_visible_devices 2 '0,1'
  local devices
  devices="$(first_visible_devices 2)"
  mkdir -p "$CI_OUTPUT_DIR/zenflow-work"
  cd "$CI_OUTPUT_DIR/zenflow-work"
  echo "running two-rank ZenFlow optimizer-offload smoke on devices $devices"
  ASCEND_RT_VISIBLE_DEVICES="$devices" deepspeed \
    --master_port "$(master_port_for 42)" \
    --num_gpus 2 \
    "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
}

run_rac_prune() {
  require_visible_devices 1 '0'
  require_overlay_path --model directory
  require_overlay_path --dataset file
  local devices
  devices="$(first_visible_devices 1)"
  echo "running upstream calibration-based pruning on NPU $devices"
  # Import the vendor backend normally before executing the unchanged entry.
  # runpy needs the sibling rac package on sys.path, as ordinary python does.
  ASCEND_RT_VISIBLE_DEVICES="$devices" "$PYTHON" -c '
import os
import runpy
import sys
import torch_npu
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(sys.argv[0]))
runpy.run_path(sys.argv[0], run_name="__main__")
' "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
}

run_autotp_equivalence() {
  require_visible_devices 4 '0,1,2,3'
  : "${QWEN3_06B_PATH:?QWEN3_06B_PATH was not exported by setup}"

  local example_dir="$EXAMPLES_ROOT/training/autotp_equivalence"
  local metrics_dir="$CI_OUTPUT_DIR/autotp-equivalence"
  mkdir -p "$metrics_dir"

  local size devices
  for size in 1 3 4; do
    devices="$(first_visible_devices "$size")"
    echo "running AutoTP=$size on ASCEND_RT_VISIBLE_DEVICES=$devices"
    ASCEND_RT_VISIBLE_DEVICES="$devices" deepspeed \
      --master_port "$(master_port_for "$((10 + size))")" \
      --num_gpus "$size" \
      "$LAUNCH_PATH" \
      --deepspeed_config "$example_dir/configs/autotp${size}.json" \
      --metrics_file "$metrics_dir/autotp${size}.jsonl" \
      "${EXTRA_ARGS[@]}"
  done

  "$PYTHON" "$example_dir/compare_loss.py" \
    "$metrics_dir/autotp1.jsonl" "$metrics_dir/autotp3.jsonl" \
    --print-every 1
  "$PYTHON" "$example_dir/compare_loss.py" \
    "$metrics_dir/autotp1.jsonl" "$metrics_dir/autotp4.jsonl" \
    --print-every 1
}

validate_expansion() {
  "$PYTHON" - "$@" <<'PY'
"""Check upstream example outputs; no NPU execution or source interception here."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import sys


def positive(value):
    return isinstance(value, (float, int)) and math.isfinite(value) and value > 0


def validate(kind: str, text: str, output: Path | None = None) -> object:
    if kind in {"model_tensor_offload", "activation_offload"}:
        summaries = [json.loads(line.split("=", 1)[1]) for line in text.splitlines()
                     if line.startswith("DRIVERRESULT=")]
        if len(summaries) != 1:
            raise ValueError("upstream driver must report exactly one two-arm summary")
        report = summaries[0]
        unpinned = "unpinned" if kind == "model_tensor_offload" else "pageable"
        for name in (unpinned, "pinned"):
            row = report[name]
            pin_key = "pin_memory" if kind == "model_tensor_offload" else "use_pin_memory"
            if (row["device"] != "npu" or row["steps"] != 3 or
                    bool(row[pin_key]) != (name == "pinned") or
                    row["experiment"] != kind or
                    not all(positive(row[key]) for key in ("step_avg_s", "step_min_s"))):
                raise ValueError(f"invalid NPU {kind} arm: {row}")
            if kind == "model_tensor_offload" and row["zero_stage"] != 3:
                raise ValueError("model tensor offload must exercise ZeRO-3")
        if not positive(report["speedup"]):
            raise ValueError("speedup must be finite/positive, not necessarily greater than 1")
    elif kind == "h2d_d2h":
        report = [json.loads(line.split("=", 1)[1]) for line in text.splitlines()
                  if line.startswith("RESULT=")]
        expected = {(arm, size) for arm in ("pageable", "torch", "native-unregistered")
                    for size in (1,)}
        if len(report) != len(expected) or {(r["arm"], r["size_mib"]) for r in report} != expected:
            raise ValueError("copy benchmark must report all three 1-MiB buffer modes")
        for row in report:
            if row["experiment"] != kind or not all(positive(row[k]) for k in ("h2d_gbps", "d2h_gbps")):
                raise ValueError(f"invalid copy measurement: {row}")
            if bool(row["accelerator_is_pinned"]) != (row["arm"] != "pageable"):
                raise ValueError(f"buffer pinning did not match the requested arm: {row}")
    elif kind == "zenflow_finetune":
        report = {int(step): float(loss) for step, loss in
                  re.findall(r"Step\s+(\d+), Loss:\s*([^,\s]+)", text)}
        if sorted(report) != list(range(1, 17)) or not all(math.isfinite(v) for v in report.values()):
            raise ValueError("ZenFlow finetune must complete 16 finite-loss optimizer updates")
        if "Training complete!" not in text:
            raise ValueError("ZenFlow did not complete checkpoint/tokenizer saving")
        if output is not None:
            if not (output / "latest").is_file() or not list(output.rglob("*model_states.pt")):
                raise ValueError("missing upstream DeepSpeed checkpoint")
            if not (output / "tokenizer_config.json").is_file():
                raise ValueError("missing saved upstream tokenizer")
    elif kind == "opsd_student":
        matches = re.findall(r"STUDENT_OK loss=(\S+) mem=\S+ offload=0 autotp=2 ws=2", text)
        if len(matches) != 1 or not math.isfinite(float(matches[0])):
            raise ValueError("student must complete finite-loss TP=2/ZeRO-3 forward/backward/step")
        report = {"loss": float(matches[0]), "autotp": 2, "world_size": 2, "offload": False}
    elif kind == "opsd_teacher":
        matches = re.findall(r"TEACHER_OK shape=\(1, 12, (\d+)\) mem=\S+ offload=True autotp=2 ws=2", text)
        if len(matches) != 1 or int(matches[0]) <= 0:
            raise ValueError("teacher must create CPU logit cache from TP=2/ZeRO-3 offload forward")
        report = {"cache_shape": [1, 12, int(matches[0])], "autotp": 2, "world_size": 2, "offload": True}
    elif kind == "opsd":
        report = []
        for step, loss, tokens in re.findall(
                r"\[opsd\]\[step (\d+)\] loss=(\S+).*?resp_tok=(\d+)", text):
            row = {"step": int(step), "loss": float(loss), "response_tokens": int(tokens)}
            if not math.isfinite(row["loss"]) or row["response_tokens"] <= 0:
                raise ValueError(f"invalid OPSD rollout/teacher/student step: {row}")
            report.append(row)
        if [row["step"] for row in report] != [0, 1, 2]:
            raise ValueError("OPSD must complete exactly three rollout/distillation steps")
    else:
        raise ValueError(f"unknown expansion result kind: {kind}")
    return report


if __name__ == "__main__":
    kind, log, report_file, *extra = sys.argv[1:]
    report = validate(kind, Path(log).read_text(), Path(extra[0]) if extra else None)
    Path(report_file).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"validated upstream {kind} results")
PY
}

run_pin_memory_example() {
  require_visible_devices 1 '0'
  local experiment="$1" devices log
  devices="$(first_visible_devices 1)"
  log="$CI_OUTPUT_DIR/$experiment.log"
  # Limits are per-process: setup's soft limit does not carry to this step.
  local memlock_hard
  memlock_hard="$(ulimit -Hl)"
  ulimit -Sl "$memlock_hard"
  # Upstream drivers own the arm subprocesses and allocate fresh rendezvous
  # ports. Do not wrap the two-arm driver in a multi-rank launcher.
  ASCEND_RT_VISIBLE_DEVICES="$devices" "$PYTHON" "$LAUNCH_PATH" "${EXTRA_ARGS[@]}" 2>&1 | tee "$log"
  validate_expansion \
    "$experiment" "$log" "$CI_OUTPUT_DIR/$experiment-results.json"
}

run_zenflow_finetune() {
  require_visible_devices 1 '0'
  require_overlay_path --model_name directory
  : "${ZENFLOW_WORK_DIR:?ZENFLOW_WORK_DIR was not exported by setup}"
  [[ -s "$ZENFLOW_WORK_DIR/tatsu-lab/alpaca/train.parquet" ]] || {
    echo 'ZenFlow local Alpaca parquet fixture is missing' >&2; exit 1;
  }
  local config="$CI_OUTPUT_DIR/zenflow-finetune-config.json" devices
  devices="$(first_visible_devices 1)"
  "$PYTHON" - "$config" <<'PY'
import json
from pathlib import Path
import sys
config = {
    "train_batch_size": 1, "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 1, "gradient_clipping": 1.0,
    "bf16": {"enabled": True}, "zero_allow_untested_optimizer": True,
    "zero_optimization": {"stage": 2,
        "offload_optimizer": {"device": "cpu", "pin_memory": False},
        "zenflow": {"topk_ratio": 0.1, "update_interval": 2,
                    "full_warm_up_rounds": 0, "overlap_step": False}},
    "optimizer": {"type": "AdamW", "params": {"lr": 2e-5,
        "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01}},
}
Path(sys.argv[1]).write_text(json.dumps(config, indent=2) + '\n')
PY
  cd "$ZENFLOW_WORK_DIR"
  # Native datasets local-directory resolution of the original fixed ID.
  # Offline mode makes any accidental fallback to the Hub fail explicitly.
  ASCEND_RT_VISIBLE_DEVICES="$devices" HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    deepspeed --master_port "$(master_port_for 81)" --num_gpus 1 \
    "$LAUNCH_PATH" --deepspeed --deepspeed_config "$config" "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$CI_OUTPUT_DIR/zenflow-finetune.log"
  validate_expansion zenflow_finetune \
    "$CI_OUTPUT_DIR/zenflow-finetune.log" "$CI_OUTPUT_DIR/zenflow-finetune-results.json" \
    "$(overlay_value --output_dir)"
}

run_opsd() {
  require_visible_devices 1 '0'
  : "${QWEN25_STUDENT_PATH:?QWEN25_STUDENT_PATH was not exported by setup}"
  : "${QWEN25_TEACHER_PATH:?QWEN25_TEACHER_PATH was not exported by setup}"
  : "${OPSD_CI_PATH:?OPSD_CI_PATH was not exported by setup}"
  [[ -d "$QWEN25_STUDENT_PATH" && -d "$QWEN25_TEACHER_PATH" && -s "$OPSD_CI_PATH" ]] || {
    echo 'OPSD requires two local ModelScope models and the local prompt fixture' >&2; exit 1;
  }
  local config="$CI_OUTPUT_DIR/opsd-config.json" devices
  devices="$(first_visible_devices 1)"
  "$PYTHON" - "$config" <<'PY'
import json
import os
from pathlib import Path
import sys
output = Path(os.environ['CI_OUTPUT_DIR'])
ds_config = output / 'opsd-student-config.json'
ds_config.write_text(json.dumps({
    'train_micro_batch_size_per_gpu': 1, 'gradient_accumulation_steps': 1,
    'bf16': {'enabled': True}, 'zero_optimization': {'stage': 0},
    'optimizer': {'type': 'AdamW', 'params': {'lr': 1e-6}},
    'gradient_clipping': 1.0,
}, indent=2) + '\n')
cfg = {
    'student': {'model_name_or_path': os.environ['QWEN25_STUDENT_PATH'], 'dtype': 'bfloat16'},
    'teacher': {'model_name_or_path': os.environ['QWEN25_TEACHER_PATH'], 'dtype': 'bfloat16',
                'offload_to_cpu': True, 'autotp_size': 1},
    'rollout': {'engine': 'hybrid_engine', 'max_prompt_length': 512,
                'max_response_length': 8, 'temperature': 0.0, 'n_samples_per_prompt': 1},
    'distillation': {'loss_type': 'reverse_kl', 'temperature': 1.0, 'chunk_size': 8},
    'training': {'micro_batch_size_per_gpu': 1, 'gradient_accumulation_steps': 1,
                 'learning_rate': 1e-6, 'num_train_epochs': 1, 'max_steps': 3,
                 'logging_steps': 1, 'save_steps': 500, 'save_dir': str(output / 'opsd-checkpoints')},
    'data': {'path': os.environ['OPSD_CI_PATH'], 'prompt_field': 'prompt', 'shuffle': False},
    'deepspeed_config': str(ds_config),
}
Path(sys.argv[1]).write_text(json.dumps(cfg, indent=2) + '\n')
PY
  # Default rollout delegates to the original student module.generate(),
  # then the original trainer does teacher-cache + streamed KL backward.
  ASCEND_RT_VISIBLE_DEVICES="$devices" HF_HUB_OFFLINE=1 \
    deepspeed --master_port "$(master_port_for 91)" --num_gpus 1 \
    "$LAUNCH_PATH" --config "$config" "${EXTRA_ARGS[@]}" 2>&1 | tee "$CI_OUTPUT_DIR/opsd.log"
  validate_expansion opsd \
    "$CI_OUTPUT_DIR/opsd.log" "$CI_OUTPUT_DIR/opsd-results.json"
}

run_opsd_isolated() {
  local role="$1" devices log
  require_visible_devices 2 '0,1'
  devices="$(first_visible_devices 2)"
  : "${QWEN25_STUDENT_PATH:?QWEN25_STUDENT_PATH was not exported by setup}"
  : "${QWEN25_TEACHER_PATH:?QWEN25_TEACHER_PATH was not exported by setup}"
  [[ -d "$QWEN25_STUDENT_PATH" && -d "$QWEN25_TEACHER_PATH" ]] || {
    echo 'isolated OPSD requires local ModelScope model paths' >&2; exit 1;
  }
  log="$CI_OUTPUT_DIR/opsd-$role.log"
  # The upstream ENV knobs are the interface; no model/source alias is needed.
  # Student's optional CPU optimizer offload is disabled because its explicitly
  # selected torch AdamW is not the ZeRO CPUAdam recipe; TP=2 + ZeRO-3 remain.
  # Legacy mem= output uses uninitialized CUDA stats (zero), not NPU memory.
  ASCEND_RT_VISIBLE_DEVICES="$devices" AUTOTP=2 \
    OFFLOAD="$([[ "$role" == teacher ]] && printf 1 || printf 0)" \
    STUDENT_MODEL="$QWEN25_STUDENT_PATH" TEACHER_MODEL="$QWEN25_TEACHER_PATH" \
    deepspeed --master_port "$(master_port_for "$([[ "$role" == teacher ]] && printf 93 || printf 92)")" \
    --num_gpus 2 --no_local_rank "$LAUNCH_PATH" "${EXTRA_ARGS[@]}" 2>&1 | tee "$log"
  validate_expansion "opsd_$role" "$log" \
    "$CI_OUTPUT_DIR/opsd-$role-results.json"
}

run_expansion_example() {
  case "$entry_key" in
    benchmarks/pin_memory/model_tensor_offload/bench.py) run_pin_memory_example model_tensor_offload ;;
    benchmarks/pin_memory/activation_offload/bench.py) run_pin_memory_example activation_offload ;;
    benchmarks/pin_memory/h2d_d2h/bench.py) run_pin_memory_example h2d_d2h ;;
    training/DeepSpeed-ZenFlow/finetuning/finetune_llama.py) run_zenflow_finetune ;;
    training/opsd/main.py) run_opsd ;;
    training/opsd/test_student_autotp_zero3.py) run_opsd_isolated student ;;
    training/opsd/test_teacher_autotp_zero3.py) run_opsd_isolated teacher ;;
    *) return 1 ;;
  esac
  # Called outside a conditional so errexit/pipefail remain active.
}
case "$entry_key" in
  benchmarks/pin_memory/model_tensor_offload/bench.py|benchmarks/pin_memory/activation_offload/bench.py|benchmarks/pin_memory/h2d_d2h/bench.py|training/DeepSpeed-ZenFlow/finetuning/finetune_llama.py|training/opsd/main.py|training/opsd/test_student_autotp_zero3.py|training/opsd/test_teacher_autotp_zero3.py)
    run_expansion_example
    exit 0
    ;;
esac

is_new_inference_entry() {
  case "$entry_key" in
    benchmarks/inference/bert-bench.py|benchmarks/inference/gpt-bench.py|\
    inference/huggingface/text-generation/ds-hf-compare.py|\
    inference/huggingface/fill-mask/test-bert.py|\
    inference/huggingface/fill-mask/test-electra.py|\
    inference/huggingface/fill-mask/test-roberta.py|\
    inference/huggingface/translation/test-t5-base.py|\
    inference/huggingface/automatic-speech-recognition/test-wav2vec2.py) return 0 ;;
    *) return 1 ;;
  esac
}

write_inference_bootstrap() {
  local output="$1"
  "$PYTHON" - "$output" <<'PY'
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(r'''import json
import math
import os
from pathlib import Path
import runpy
import sys
import torch
import torch_npu
from deepspeed.accelerator import get_accelerator

kind = os.environ['DS_INFERENCE_KIND']
source = os.environ['DS_INFERENCE_SOURCE']
if get_accelerator().device_name() != 'npu' or not torch.npu.is_available():
    raise SystemExit('inference smoke requires real NPU execution')
get_accelerator().set_device(int(os.environ.get('LOCAL_RANK', '0')))
sys.argv[0] = source
sys.path.insert(0, str(Path(source).parent))
state = runpy.run_path(source, run_name='__main__')

def require(condition, message):
    if not condition:
        raise SystemExit(message)

def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)

if kind == 'asr':
    model = state['model']
    require(next(model.parameters()).device.type == 'npu', 'CTC model weights are not NPU')
    result = state['result']
    require(len(result) == 2 and all(isinstance(x['transcription'], str) for x in result), 'CTC did not process both synthetic waveforms')
    score = state['wer'](result['text'], result['transcription'])
    require(finite(score) and score >= 0, 'CTC fixture WER must be finite, not an accuracy assertion')
    # The original maps argmax predictions; additionally reject NaN/Inf logits
    # on one original waveform without replacing its dataset/model functions.
    speech = state['librispeech_eval'][0]['speech']
    values = state['processor'](speech, sampling_rate=16000, return_tensors='pt', padding='longest').input_values
    with torch.no_grad():
        logits = model(values.to(state['device'])).logits
    require(logits.numel() > 0 and bool(torch.isfinite(logits).all()), 'CTC forward logits must be nonempty and finite')
else:
    pipe = state.get('pipe', state.get('translator'))
    require(pipe is not None and pipe.device.type == 'npu', 'pipeline device is not NPU')
    require(next(pipe.model.parameters()).device.type == 'npu', 'model weights are not on NPU')
    if kind in ('bert-bench', 'gpt-bench'):
        require(len(state['times']) == 4 and all(finite(x) and x > 0 for x in state['times']), 'missing/invalid four NPU timings')
        require(state['mtimes'] and all(finite(x) and x >= 0 for x in state['mtimes']), 'missing/invalid DS model timings')
        require(len(state['responses']) == 4 and all(state['responses']), 'missing benchmark predictions')
    elif kind == 'compare':
        require(state['match_count'] == 2 and state['mismatch_count'] == 0, 'HF/DS outputs must match for both CI prompts')
    elif kind == 'translation':
        require(state['translation'] and state['translation'][0].get('translation_text', '').strip(), 'empty T5 translation')
    elif kind == 'fill-mask':
        predictions = state['output']
        require(predictions and all(finite(x['score']) and x['token_str'].strip() for x in predictions), 'missing/invalid fill-mask predictions')
    else:
        raise SystemExit(f'unknown inference assertion kind: {kind}')
print(f'CI inference assertions passed: {kind}, device=npu')
''', encoding='utf-8')
PY
}

run_new_inference() {
  local kind cards=1 alias='' offset=70
  case "$entry_key" in
    benchmarks/inference/bert-bench.py)
      kind=bert-bench; require_overlay_path --model directory ;;
    benchmarks/inference/gpt-bench.py)
      kind=gpt-bench; offset=71; require_overlay_path --model directory ;;
    inference/huggingface/text-generation/ds-hf-compare.py)
      kind=compare; offset=72; require_overlay_path --model directory ;;
    inference/huggingface/fill-mask/test-bert.py)
      kind=fill-mask; alias=bert-large-cased; offset=73 ;;
    inference/huggingface/fill-mask/test-electra.py)
      kind=fill-mask; alias=google/electra-base-generator; cards=2; offset=74 ;;
    inference/huggingface/fill-mask/test-roberta.py)
      kind=fill-mask; alias=roberta-large; cards=2; offset=75 ;;
    inference/huggingface/translation/test-t5-base.py)
      kind=translation; alias=t5-base; cards=2; offset=76 ;;
    inference/huggingface/automatic-speech-recognition/test-wav2vec2.py)
      kind=asr; alias=facebook/wav2vec2-base-960h; offset=78 ;;
    *) return 1 ;;
  esac
  require_visible_devices "$cards" "$([[ "$cards" == 2 ]] && printf '0,1' || printf '0')"
  local devices bootstrap
  devices="$(first_visible_devices "$cards")"
  if [[ -n "$alias" ]]; then
    if [[ -z "${INFERENCE_CI_WORK:-}" || ! -f "$INFERENCE_CI_WORK/$alias/config.json" ]]; then
      echo "local inference alias missing for $entry_key: $alias" >&2
      exit 1
    fi
    cd "$INFERENCE_CI_WORK"
  else
    mkdir -p "$CI_OUTPUT_DIR/inference-work"
    cd "$CI_OUTPUT_DIR/inference-work"
  fi
  bootstrap="$CI_OUTPUT_DIR/inference-entry.py"
  write_inference_bootstrap "$bootstrap" || exit "$?"
  # no_local_rank works with every native entry, including those with no parser.
  # LOCAL_RANK/WORLD_SIZE are still injected by DeepSpeed's launcher.
  # Explicit || exit is essential: this function is called from an if-condition,
  # which otherwise disables bash errexit throughout the function body.
  DS_INFERENCE_KIND="$kind" DS_INFERENCE_SOURCE="$LAUNCH_PATH" \
  ASCEND_RT_VISIBLE_DEVICES="$devices" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  deepspeed --master_port "$(master_port_for "$offset")" --num_nodes 1 \
    --num_gpus "$cards" --no_local_rank "$bootstrap" "${EXTRA_ARGS[@]}" || exit "$?"
  return 0
}
if is_new_inference_entry; then
  run_new_inference
  exit 0
fi

is_training_expansion_entry() {
  case "$entry_key" in
    training/deepspeed_finetune_demo/finetune_llama.py|training/stable_diffusion/train_sd_distil_lora.py|training/opsd/benchmarks/bench_decode_1p1r.py) return 0 ;;
    *) return 1 ;;
  esac
}

run_training_expansion_executor() {
  case "$entry_key" in
    training/deepspeed_finetune_demo/finetune_llama.py) run_finetune_demo_ci ;;
    training/stable_diffusion/train_sd_distil_lora.py) run_sd_distil_ci ;;
    training/opsd/benchmarks/bench_decode_1p1r.py) run_opsd_decode_ci ;;
    *) echo "unknown training expansion entry: $entry_key" >&2; return 1 ;;
  esac
}

run_finetune_demo_ci() {
  require_visible_devices 2 '0,1'
  require_overlay_path --model_name directory
  require_overlay_path --dataset_name directory
  local devices config log
  devices="$(first_visible_devices 2)"
  config="$CI_OUTPUT_DIR/finetune-demo-config.json"
  log="$CI_OUTPUT_DIR/finetune-demo.log"
  "$PYTHON" - "$config" <<'PY'
import json
from pathlib import Path
import sys

config = {
    'train_batch_size': 2, 'train_micro_batch_size_per_gpu': 1,
    'gradient_accumulation_steps': 1, 'bf16': {'enabled': True},
    'zero_optimization': {'stage': 2},
    'optimizer': {'type': 'AdamW', 'params': {'lr': 2e-5, 'torch_adam': True}},
    'gradient_clipping': 1.0, 'steps_per_print': 1,
}
Path(sys.argv[1]).write_text(json.dumps(config, indent=2) + '\n')
PY
  mkdir -p "$CI_OUTPUT_DIR/finetune-demo-work"
  cd "$CI_OUTPUT_DIR/finetune-demo-work"
  echo "running native SmolLM2/Llama finetune recipe: two-rank DP/ZeRO-2, three updates"
  ASCEND_RT_VISIBLE_DEVICES="$devices" WANDB_MODE=disabled deepspeed \
    --master_port "$(master_port_for 111)" --num_gpus 2 \
    "$LAUNCH_PATH" --deepspeed --deepspeed_config "$config" "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$log"
  "$PYTHON" - "$log" "$CI_OUTPUT_DIR/finetune-demo-losses.json" <<'PY'
import json
import math
from pathlib import Path
import re
import sys

text = Path(sys.argv[1]).read_text()
if not re.search(r'First batch valid shifted labels: [1-9]\d*, finite params: True, finite logits: True,.*finite loss: True', text):
    raise SystemExit('finetune demo did not prove valid labels and finite first-batch tensors')
losses = {}
for match in re.finditer(r'Step\s+(\d+), Loss:\s*([^,\s]+)', text):
    step, value = int(match[1]), float(match[2])
    if not math.isfinite(value):
        raise SystemExit(f'finetune demo produced nonfinite loss at step {step}')
    losses[step] = value
if sorted(losses) != [1, 2, 3] or 'Training complete!' not in text:
    raise SystemExit('finetune demo must finish exactly three finite updates')
Path(sys.argv[2]).write_text(json.dumps(losses, allow_nan=False) + '\n')
print('validated three finite finetune demo updates')
PY
}

run_sd_distil_ci() {
  require_visible_devices 2 '0,1'
  require_overlay_path --pretrained_model_name_or_path directory
  local devices config output log
  devices="$(first_visible_devices 2)"
  config="$CI_OUTPUT_DIR/sd-distil-config.json"
  log="$CI_OUTPUT_DIR/sd-distil.log"
  output="$(overlay_value --output_dir)"
  [[ "$output" == "$CI_OUTPUT_DIR/sd-distil" ]] || {
    echo "SD distillation must save its fresh pipeline under CI output" >&2; return 1;
  }
  [[ ! -e "$output/model_index.json" ]] || {
    echo "SD distillation refuses to reuse a stale output pipeline" >&2; return 1;
  }
  [[ -s "$CI_OUTPUT_DIR/sd-distil-work/poloclub/diffusiondb/train.parquet" ]] || {
    echo "SD local eight-image dataset missing; run ds_sd_distil setup first" >&2; return 1;
  }
  [[ "$(overlay_value --mixed_precision)" == no ]] || {
    echo "SD unwrapped teacher requires the FP32 CI recipe (--mixed_precision no)" >&2; return 1;
  }
  "$PYTHON" - "$config" <<'PY'
import json
from pathlib import Path
import sys

config = {
    'train_batch_size': 2, 'train_micro_batch_size_per_gpu': 1,
    'gradient_accumulation_steps': 1, 'zero_optimization': {'stage': 2},
    'fp16': {'enabled': False}, 'bf16': {'enabled': False},
    'gradient_clipping': 1.0,
}
Path(sys.argv[1]).write_text(json.dumps(config, indent=2) + '\n')
PY
  cd "$CI_OUTPUT_DIR/sd-distil-work"
  echo "running teacher-CFG full-UNet distillation using native local images, FP32 and two-rank DeepSpeed"
  ASCEND_RT_VISIBLE_DEVICES="$devices" WANDB_MODE=disabled TQDM_MININTERVAL=0 \
  accelerate launch --use_deepspeed --num_processes 2 --num_machines 1 \
    --mixed_precision no --deepspeed_config_file "$config" \
    --main_process_port "$(master_port_for 121)" \
    "$LAUNCH_PATH" "${EXTRA_ARGS[@]}" 2>&1 | tee "$log"
  "$PYTHON" - "$log" "$output" "$CI_OUTPUT_DIR/sd-distil-losses.json" <<'PY'
import json
import math
from pathlib import Path
import re
import sys

text = Path(sys.argv[1]).read_text()
losses = {}
# set_postfix(refresh=True) in the original entry prints the current loss
# after each progress update. Parse CR-separated tqdm frames, not wall-clock speed.
for frame in re.split(r'[\r\n]', text):
    match = re.search(r'\b([123])/3\b.*?\bloss=([^,\]\s]+)', frame)
    if match:
        step, value = int(match[1]), float(match[2])
        if not math.isfinite(value):
            raise SystemExit(f'SD distillation nonfinite loss at step {step}')
        losses[step] = value
if sorted(losses) != [1, 2, 3]:
    raise SystemExit('SD distillation must display three finite train-step losses')
output = Path(sys.argv[2])
for name in ('model_index.json', 'unet/config.json', 'unet/diffusion_pytorch_model.safetensors'):
    file = output / name
    if not file.is_file() or not file.stat().st_size:
        raise SystemExit(f'SD distillation failed to save a fresh trained pipeline: {file}')
Path(sys.argv[3]).write_text(json.dumps(losses, allow_nan=False) + '\n')
print('validated three finite SD distillation updates and trained pipeline save')
PY
}

run_opsd_decode_ci() {
  require_visible_devices 1 '0'
  require_overlay_path --model directory
  local devices log
  devices="$(first_visible_devices 1)"
  log="$CI_OUTPUT_DIR/opsd-decode.log"
  mkdir -p "$CI_OUTPUT_DIR/opsd-decode-work"
  cd "$CI_OUTPUT_DIR/opsd-decode-work"
  # Normal backend import and runpy preserve every original API and function.
  ASCEND_RT_VISIBLE_DEVICES="$devices" "$PYTHON" -c '
import os, runpy, sys
import torch, torch_npu
from deepspeed.accelerator import get_accelerator
index = get_accelerator().current_device()
assert torch.empty(1, device="cpu").to(index).device.type == "npu"
assert torch.randint(10, 1000, (1, 2), device=index).device.type == "npu"
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(sys.argv[0]))
runpy.run_path(sys.argv[0], run_name="__main__")
' "$LAUNCH_PATH" "${EXTRA_ARGS[@]}" 2>&1 | tee "$log"
  "$PYTHON" - "$log" "$CI_OUTPUT_DIR/opsd-decode-timings.json" <<'PY'
import json
import math
from pathlib import Path
import re
import sys

text = Path(sys.argv[1]).read_text()
timings = {}
for label in ('Raw decode loop', 'HybridEngine rollout'):
    match = re.search(re.escape(label) + r':\s*([^\s]+)\s+ms', text)
    if not match:
        raise SystemExit(f'OPSD decode missing completed timing: {label}')
    value = float(match[1])
    if not math.isfinite(value) or value <= 0:
        raise SystemExit(f'OPSD decode invalid timing: {label}={value}')
    timings[label] = value
Path(sys.argv[2]).write_text(json.dumps(timings, allow_nan=False) + '\n')
print('validated raw sampling/decode and HybridEngine rollout completion; no speed threshold')
PY
}
if is_training_expansion_entry; then
  run_training_expansion_executor
  exit 0
fi

run_vllm_benchmark() {
  require_visible_devices 1 '0'
  : "${VLLM_CI_MODEL_PATH:?local vLLM model is required}"
  : "${VLLM_CI_BENCH_ROOT:?benchmark tokenizer directory is required}"
  export ASCEND_RT_VISIBLE_DEVICES="$(first_visible_devices 1)"
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export OPENAI_API_KEY=ci-local-not-a-credential
  local port server_pid log ready=0
  port="$(master_port_for 130)"
  log="$CI_OUTPUT_DIR/vllm-server.log"
  vllm serve "$VLLM_CI_MODEL_PATH" --served-model-name ci-model \
    --host 127.0.0.1 --port "$port" --max-model-len 256 \
    --max-num-seqs 4 --gpu-memory-utilization 0.5 --enforce-eager > "$log" 2>&1 &
  server_pid=$!
  trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT
  for ((attempt=0; attempt<180; attempt++)); do
    if curl --noproxy '*' -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then ready=1; break; fi
    if ! kill -0 "$server_pid" 2>/dev/null; then cat "$log"; exit 1; fi
    sleep 2
  done
  if ((ready == 0)); then cat "$log"; echo 'vLLM server did not become ready' >&2; exit 1; fi
  cd "$VLLM_CI_BENCH_ROOT"
  local url="http://127.0.0.1:$port/v1/completions"
  if [[ "$entry_key" == benchmarks/inference/mii/run_benchmark.py ]]; then
    # The client's barrier can keep forked workers alive after a parent error.
    timeout --kill-after=10s 5m "$PYTHON" "$LAUNCH_PATH" "${EXTRA_ARGS[@]}" \
      --openai_api_url "$url" --openai_api_key "$OPENAI_API_KEY"
  else
    PYTHONPATH="$EXAMPLES_ROOT/benchmarks/inference/deepspeedometer/src${PYTHONPATH:+:$PYTHONPATH}" \
      "$PYTHON" -m deepspeedometer.benchmark_runner "${EXTRA_ARGS[@]}" --url "$url"
  fi
  # Clean up while the local PID is in scope; preserve native failures above.
  kill "$server_pid"
  wait "$server_pid" || true
  trap - EXIT
}

case "$entry_key" in
  training/autotuning/hf/*/test_tune.sh)
    require_visible_devices 1 '0'
    : "${AUTOTUNE_WORK:?autotuning dependencies/data were not exported}"
    : "${AUTOTUNE_MODEL_PATH:?local autotuning model is required}"
    cd "$CI_OUTPUT_DIR"
    "$PYTHON" - "$CI_OUTPUT_DIR" <<'PY'
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
config = {'train_micro_batch_size_per_gpu': [1, 2], 'gradient_accumulation_steps': 1,
          'zero_optimization': {'stage': 0},
          'autotuning': {'enabled': True, 'fast': True, 'start_profile_step': 1, 'end_profile_step': 3,
                         'results_dir': str(output / 'tuning-results'), 'exps_dir': str(output / 'tuning-experiments'),
                         'arg_mappings': {'train_micro_batch_size_per_gpu': '--per_device_train_batch_size',
                                          'gradient_accumulation_steps': '--gradient_accumulation_steps'}}}
(output / 'autotuning.json').write_text(json.dumps(config))
(output / 'hostfile').write_text('localhost slots=1\n')
PY
    # evaluate.load('accuracy') finds the unchanged official metric locally.
    cd "$AUTOTUNE_WORK"
    deepspeed --autotuning run --hostfile "$CI_OUTPUT_DIR/hostfile" --num_nodes 1 --num_gpus 1 \
      --master_port "$(master_port_for 132)" "$AUTOTUNE_WORK/train.py" \
      --deepspeed "$CI_OUTPUT_DIR/autotuning.json" --model_name_or_path "$AUTOTUNE_MODEL_PATH" \
      --train_file "$AUTOTUNE_WORK/train.json" --validation_file "$AUTOTUNE_WORK/train.json" \
      --do_train --max_steps 5 \
      --per_device_train_batch_size 1 --save_strategy no --report_to none \
      --output_dir "$CI_OUTPUT_DIR/train" "${EXTRA_ARGS[@]}"
    # The native tuner can log failed experiments and exit zero without a result.
    test -s "$CI_OUTPUT_DIR/tuning-results/ds_config_optimal.json"
    exit 0
    ;;
  compression/reasoning_aware_compression/bash_script/eval_math500.sh)
    require_visible_devices 1 '0'
    : "${VLLM_CI_MODEL_PATH:?local vLLM model is required}"
    : "${MATH500_CI_WORK:?local MATH-500 task/data are required}"
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    # Preserve the shell recipe's model/sampling/metric semantics, using the
    # current native CLI and custom-task hook to select eight local problems.
    lighteval vllm \
      "model_name=$VLLM_CI_MODEL_PATH,max_model_length=256,gpu_memory_utilization=0.8,override_chat_template=true,generation_parameters={max_new_tokens:16,temperature:0.6,top_p:0.95}" \
      'ci_math500|0' --custom-tasks "$MATH500_CI_WORK/task.py" "${EXTRA_ARGS[@]}"
    exit 0
    ;;
  benchmarks/inference/mii/run_benchmark.py|benchmarks/inference/deepspeedometer/src/deepspeedometer/benchmark_runner.py)
    run_vllm_benchmark
    exit 0
    ;;
  deepnvme/ds_io/ds_io_sweep.sh)
    require_visible_devices 1 '0'
    ulimit -Sl "$(ulimit -Hl)"
    mkdir -p "$CI_OUTPUT_DIR/aio"
    # Same ds_io accelerator recipe as the sweep, reduced to one write/read pair.
    ds_io "${EXTRA_ARGS[@]}" --folder_to_device_mapping "$CI_OUTPUT_DIR/aio:0"
    ds_io "${EXTRA_ARGS[@]}" --folder_to_device_mapping "$CI_OUTPUT_DIR/aio:0" --read
    exit 0
    ;;
  training/deepspeed_finetune_demo/evaluate/gsm8k/gen_gsm8k.py|training/deepspeed_finetune_demo/evaluate/mmlu/gen_mmlu.py|compression/reasoning_aware_compression/collect_traces.py)
    require_visible_devices 1 '0'
    require_overlay_path --model directory
    export ASCEND_RT_VISIBLE_DEVICES="$(first_visible_devices 1)"
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    # Fail rather than silently exercising CPU/CUDA instead of Ascend.
    "$PYTHON" -c 'import torch, torch_npu; from vllm.platforms import current_platform; assert current_platform.device_type == "npu" and torch.npu.is_available(), "vLLM must select available NPU"'
    if [[ "$entry_key" == compression/reasoning_aware_compression/collect_traces.py ]]; then
      require_overlay_path --dataset file
    else
      : "${VLLM_CI_DATA_ROOT:?native local evaluation datasets were not exported by setup}"
      [[ -d "$VLLM_CI_DATA_ROOT" ]] || { echo "local evaluation dataset root missing: $VLLM_CI_DATA_ROOT" >&2; exit 1; }
      # load_dataset('openai/gsm8k' or 'cais/mmlu') natively finds the staged
      # local directories from this cwd; no monkeypatch or source edit.
      cd "$VLLM_CI_DATA_ROOT"
    fi
    # Keep the original program's exit status; no answer/accuracy threshold.
    "$PYTHON" "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
    exit 0
    ;;
  training/cifar/run_ds_moe.sh)
    run_cifar_moe
    exit 0
    ;;
  training/cifar/run_ds_prmoe.sh)
    run_cifar_prmoe
    exit 0
    ;;
  training/tensor_parallel/hf_integration/train.py)
    run_hf_autotp
    exit 0
    ;;
  applications/DeepSpeed-Chat/training/step1_supervised_finetuning/prompt_eval.py)
    run_chat_prompt_eval
    exit 0
    ;;
  applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/rw_eval.py)
    run_chat_reward_eval
    exit 0
    ;;
  training/tensor_parallel/hf_integration/train_bench_length.py)
    run_hf_bench_length
    exit 0
    ;;
  training/DeepSpeed-SuperOffload/finetune_zero3.py)
    run_superoffload
    exit 0
    ;;
  training/data_efficiency/variable_batch_size_and_lr/variable_batch_size_and_lr_example.py)
    run_variable_batch
    exit 0
    ;;
  training/DeepSpeed-ZenFlow/benchmark/zf_benchmark.py)
    run_zenflow
    exit 0
    ;;
  compression/reasoning_aware_compression/prune.py)
    run_rac_prune
    exit 0
    ;;
  training/autotp_equivalence/train.py)
    run_autotp_equivalence
    exit 0
    ;;
  *)
    require_visible_devices 1 '0'
    ;;
esac

case "$LAUNCH_PATH" in
  *.sh)
    bash "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
    ;;
  *.py)
    # DeepSpeed-Chat reads torch.distributed.get_rank() even when local_rank
    # keeps its default -1, so it still needs the launcher to initialize a
    # one-rank process group. Its upstream training_scripts wrappers hardcode
    # large models/configs and do not pass arbitrary overlay args through;
    # invoke the same launcher directly with our CI-sized manifest arguments.
    case "${EXEC:-}" in
      applications/DeepSpeed-Chat/training/*/main.py)
        deepspeed --master_port "$(master_port_for 1)" --num_gpus 1 \
          "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
        ;;
      *)
        "$PYTHON" "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
        ;;
    esac
    ;;
  *)
    echo "unsupported example type: $LAUNCH_PATH" >&2
    exit 1
    ;;
esac
