#!/usr/bin/env bash
# Run one supported slime example.
#
# $1 is the manifest entry path. The upstream launchers hardcode paths,
# models and GPU layouts without "$@" passthrough, so the CI train
# recipe lives in the manifest overlay_args (mirroring the fork's NPU
# CI tests) and this script only maps the per-entry engine-call metadata
# that the engine cannot pass through (megatron model type / train
# script), then invokes the fork's own execute_train() helper. Never
# git add/commit/push here.
set -euo pipefail

# The upstream trainer owns pass/fail. Keep debug dumps for diagnosis, but
# do not grade model answers, reasoning delimiters or tool-use coverage here.

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <example-relpath>" >&2
  exit 2
fi

EXAMPLE_REL="$1"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
CI_OUTPUT_DIR="${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"
EXAMPLES_ROOT="${EXAMPLES_ROOT:-$TARGET_ROOT}"
GITHUB_WORKSPACE="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
DEPS_ROOT="$GITHUB_WORKSPACE/deps"
SLIME_PROJECT_ENV="$DEPS_ROOT/slime-example.env"
if [[ -f "$SLIME_PROJECT_ENV" ]]; then
  # GITHUB_ENV normally carries values between Actions steps. Keep a
  # workspace-local copy as a fallback for container runners where that
  # file-command handoff was not reflected in the next step's environment.
  # shellcheck disable=SC1090
  source "$SLIME_PROJECT_ENV"
fi
SLIME_FORK_ROOT="${SLIME_FORK_ROOT:?SLIME_FORK_ROOT was not exported by setup}"
export PYTHONPATH="$DEPS_ROOT/sglang/python:$SLIME_FORK_ROOT:$DEPS_ROOT/Megatron-LM:$DEPS_ROOT/Megatron-Bridge/src:${PYTHONPATH:-}"

EXAMPLE_PATH="$EXAMPLES_ROOT/$EXAMPLE_REL"
if [[ ! -f "$EXAMPLE_PATH" ]]; then
  echo "example not found: $EXAMPLE_PATH" >&2
  exit 1
fi

mkdir -p "$CI_OUTPUT_DIR" "$CI_OUTPUT_DIR/agent-rollouts" "$CI_OUTPUT_DIR/agent-gradients"

# Vendor env scripts assume a login shell and die under `set -u`; relax
# strict mode only while sourcing them (same trap as setup_example.sh).
source_vendor_env() {
  local vendor_file="$1"
  [[ -f "$vendor_file" ]] || return 0
  set +eu
  # shellcheck disable=SC1090
  source "$vendor_file"
  set -eu
}

source_vendor_env /usr/local/Ascend/ascend-toolkit/set_env.sh
source_vendor_env /usr/local/Ascend/nnal/atb/set_env.sh

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  PYTHON=python
fi

# OPD's teacher listens on a separate local port. The manifest expands
# this value into --rm-url before the fork launcher starts Ray.
export SLIME_TEACHER_PORT=$((13141 + ${GITHUB_RUN_ID:-0} % 1000))
export SLIME_TEACHER_URL="http://127.0.0.1:${SLIME_TEACHER_PORT}/generate"

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
    # Parse the recipe first: a local model directory containing spaces
    # remains one CLI token after environment substitution.
    tokens.extend(os.path.expandvars(token) for token in shlex.split(item, posix=True))
print(' '.join(shlex.quote(token) for token in tokens))
PY
}

eval "EXTRA_ARGS=( $(expand_overlay) )"

echo "running $EXAMPLE_REL with ${#EXTRA_ARGS[@]} overlay args"
if ((${#EXTRA_ARGS[@]})); then
  printf 'overlay arg: %q\n' "${EXTRA_ARGS[@]}"
fi

entry_key="$EXAMPLE_REL"

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

# Environment contract copied from the fork's ascend launchers and NPU CI
# tests (scripts/ascend_script/run-qwen3-8B-npu.sh, tests/tests_npu/):
# Ray must not rewrite the visible-device mask, HCCL needs its port range
# and the NPU allocator keeps expandable_segments (no vLLM CaMemAllocator
# in this stack, unlike examples-guards/roll).
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
# Offline wandb: get_default_wandb_args() stays quiet without
# WANDB_API_KEY, and WANDB_MODE=offline guarantees no network call even
# if one is set.
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"

# run-id derived ray dashboard port avoids collisions between parallel
# matrix legs that share a runner.
export RAY_DASHBOARD_PORT=$((8265 + ${GITHUB_RUN_ID:-0} % 100))

# execute_train() takes the full train-arg list as one shell-quoted
# string (fork API contract); rebuild it from the expanded overlay.
case "$entry_key" in
  examples/fully_async/run-qwen2.5-0.5B-fully_async.sh)
    # Recipe source: tests/tests_npu/nightly_CI/
    # test_qwen2.5_0.5B_fully_async_short_npu.py (fork-verified on NPU).
    require_visible_devices 4 '0,1,2,3'
    NUM_GPUS=4
    MODEL_TYPE=qwen2.5-0.5B
    TRAIN_SCRIPT=train_async.py
    ;;
  examples/on_policy_distillation/run-qwen3-8B-opd.sh)
    # Fork NPU ST: 4 actor + 3 rollout devices in Ray, 1 teacher server.
    require_visible_devices 8 '0,1,2,3,4,5,6,7'
    IFS=',' read -r -a opd_devices <<< "$ASCEND_RT_VISIBLE_DEVICES"
    export SLIME_OPD_TRAIN_DEVICES="$(IFS=,; echo "${opd_devices[*]:0:7}")"
    export SLIME_OPD_TEACHER_DEVICE="${opd_devices[7]}"
    NUM_GPUS=7
    MODEL_TYPE=qwen2.5-0.5B
    TRAIN_SCRIPT=train.py
    ;;
  examples/retool/retool_qwen3_4b_rl.sh)
    # The upstream CUDA script colocates train and rollout on four cards.
    # On NPU run #16 the TP2 servers became unhealthy while the train
    # actor was initializing. Reserve four cards for each side instead.
    require_visible_devices 8 '0,1,2,3,4,5,6,7'
    export SLIME_RETOOL_DEVICES="$ASCEND_RT_VISIBLE_DEVICES"
    NUM_GPUS=8
    MODEL_TYPE=qwen3-4B-Instruct-2507
    TRAIN_SCRIPT=train.py
    export PYTHONPATH="$SLIME_FORK_ROOT/examples/retool:$PYTHONPATH"
    ;;
  examples/retool/retool_qwen3_4b_sft.sh)
    require_visible_devices 4 '0,1,2,3'
    IFS=',' read -r -a sft_devices <<< "$ASCEND_RT_VISIBLE_DEVICES"
    export SLIME_RETOOL_DEVICES="$(IFS=,; echo "${sft_devices[*]:0:4}")"
    : "${SLIME_SFT_FIXTURE_JSONL:?SFT fixture was not exported by setup}"
    NUM_GPUS=4
    MODEL_TYPE=qwen3-4B-Instruct-2507
    TRAIN_SCRIPT=train_async.py
    ;;
  examples/on_policy_distillation/run-qwen3-8B-opd-megatron.sh)
    require_visible_devices 8 '0,1,2,3,4,5,6,7'
    export SLIME_RETOOL_DEVICES="$ASCEND_RT_VISIBLE_DEVICES"
    NUM_GPUS=8
    MODEL_TYPE=qwen2.5-0.5B
    TRAIN_SCRIPT=train.py
    ;;
  examples/train_infer_mismatch_helper/run-qwen3-4b-mis.sh)
    require_visible_devices 8 '0,1,2,3,4,5,6,7'
    export SLIME_RETOOL_DEVICES="$ASCEND_RT_VISIBLE_DEVICES"
    NUM_GPUS=8
    MODEL_TYPE=qwen3-4B-Instruct-2507
    TRAIN_SCRIPT=train.py
    ;;
  examples/multi_agent/run-qwen3-30B-A3B-multi-agent.sh)
    require_visible_devices 8 '0,1,2,3,4,5,6,7'
    export SLIME_RETOOL_DEVICES="$ASCEND_RT_VISIBLE_DEVICES"
    NUM_GPUS=8
    MODEL_TYPE=qwen3-4B
    TRAIN_SCRIPT=train.py
    ;;
  examples/eval_multi_task/multi_task.sh)
    require_visible_devices 8 '0,1,2,3,4,5,6,7'
    export SLIME_RETOOL_DEVICES="$ASCEND_RT_VISIBLE_DEVICES"
    : "${SLIME_EVAL_CONFIG:?Multi-task evaluation config was not exported by setup}"
    NUM_GPUS=8
    MODEL_TYPE=qwen2.5-0.5B
    TRAIN_SCRIPT=train.py
    ;;
  examples/strands_sglang/strands_qwen3_8b.sh)
    require_visible_devices 8 '0,1,2,3,4,5,6,7'
    export SLIME_RETOOL_DEVICES="$ASCEND_RT_VISIBLE_DEVICES"
    NUM_GPUS=8
    MODEL_TYPE=qwen3-4B-Instruct-2507
    TRAIN_SCRIPT=train.py
    export PYTHONPATH="$SLIME_FORK_ROOT/examples/strands_sglang:$PYTHONPATH"
    ;;
  examples/search-r1/run_qwen2.5_3B.sh)
    require_visible_devices 8 '0,1,2,3,4,5,6,7'
    export SLIME_RETOOL_DEVICES="$ASCEND_RT_VISIBLE_DEVICES"
    : "${SLIME_SEARCH_INDEX:?Search-R1 BM25 index was not exported by setup}"
    NUM_GPUS=8
    MODEL_TYPE=qwen2.5-0.5B
    TRAIN_SCRIPT=train.py
    export PYTHONPATH="$SLIME_FORK_ROOT/examples/search-r1:$PYTHONPATH"
    ;;
  examples/geo3k_vlm_multi_turn/run_geo3k_vlm_multi_turn.py)
    require_visible_devices 4 '0,1,2,3'
    IFS=',' read -r -a geo_devices <<< "$ASCEND_RT_VISIBLE_DEVICES"
    export SLIME_RETOOL_DEVICES="$(IFS=,; echo "${geo_devices[*]:0:4}")"
    export MODEL_ARGS_ROTARY_BASE=5000000
    NUM_GPUS=4
    MODEL_TYPE=qwen3-1.7B
    TRAIN_SCRIPT=train.py
    ;;
  *)
    echo "no engine-call metadata mapping for $entry_key" >&2
    exit 1
    ;;
esac

: "${SLIME_MODEL_PATH:?SLIME_MODEL_PATH was not exported by setup}"
if [[ "$entry_key" != examples/geo3k_vlm_multi_turn/run_geo3k_vlm_multi_turn.py ]]; then
  : "${SLIME_TORCH_DIST_PATH:?SLIME_TORCH_DIST_PATH was not exported by setup}"
fi
: "${SLIME_FIXTURE_JSONL:?SLIME_FIXTURE_JSONL was not exported by setup}"

cd "$SLIME_FORK_ROOT"

"$PYTHON" - "$SLIME_FORK_ROOT" "$NUM_GPUS" "$MODEL_TYPE" "$TRAIN_SCRIPT" "${EXTRA_ARGS[@]}" <<'PY' 2>&1 | tee "$CI_OUTPUT_DIR/agent-train.log"
import importlib.util
import json
import os
import signal
import shlex
import re
import subprocess
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path

fork_root, num_gpus, model_type, train_script, *train_args = sys.argv[1:]

# The fork execute_train() owns ray start/submit, NPU resource
# injection and the runtime env; load it straight from the fork tree.
if str(fork_root) not in sys.path:
    sys.path.insert(0, str(fork_root))
spec = importlib.util.spec_from_file_location(
    "_fork_command_utils",
    str(Path(fork_root) / "slime" / "utils" / "external_utils" / "command_utils.py"),
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

print(f"execute_train: model_type={model_type} train_script={train_script} num_gpus={num_gpus}")
print("train args:", " ".join(train_args))

teacher_device = os.environ.get("SLIME_OPD_TEACHER_DEVICE")
teacher_process = None
search_process = None
search_log = Path(os.environ["CI_OUTPUT_DIR"]) / "search-retriever.log"
teacher_log = Path(os.environ["CI_OUTPUT_DIR"]) / "opd-teacher.log"


def teacher_log_tail():
    if teacher_log.exists():
        print("teacher log (last 40 lines):", flush=True)
        with teacher_log.open(errors="replace") as output:
            print("".join(deque(output, maxlen=40)), flush=True)


def start_opd_teacher():
    global teacher_process
    teacher_env = os.environ.copy()
    teacher_env.update({
        "CUDA_VISIBLE_DEVICES": teacher_device,
        "ASCEND_RT_VISIBLE_DEVICES": teacher_device,
        "GLOO_SOCKET_IFNAME": "lo",
        "NCCL_SOCKET_IFNAME": "lo",
        "TP_SOCKET_IFNAME": "lo",
        "no_proxy": "127.0.0.1",
    })
    for proxy in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        teacher_env.pop(proxy, None)
    command = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", os.environ["SLIME_MODEL_PATH"],
        "--host", "127.0.0.1",
        "--port", os.environ["SLIME_TEACHER_PORT"],
        "--tp", "1",
        "--mem-fraction-static", "0.6",
    ]
    with teacher_log.open("w") as output:
        teacher_process = subprocess.Popen(
            command, env=teacher_env, stdout=output,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
    print(f"OPD teacher: device={teacher_device} pid={teacher_process.pid} log={teacher_log}", flush=True)
    health_url = os.environ["SLIME_TEACHER_URL"].removesuffix("/generate") + "/health_generate"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if teacher_process.poll() is not None:
            teacher_log_tail()
            raise RuntimeError(f"OPD teacher exited with code {teacher_process.returncode}")
        try:
            with opener.open(health_url, timeout=3) as response:
                if response.status == 200:
                    print("OPD teacher ready", flush=True)
                    return
        except Exception:
            pass
        time.sleep(5)
    teacher_log_tail()
    raise TimeoutError("OPD teacher did not become healthy within 600 seconds")


launch_options = {}
retool_devices = os.environ.get("SLIME_RETOOL_DEVICES")
if teacher_device:
    train_devices = os.environ["SLIME_OPD_TRAIN_DEVICES"]
    # The fork builds Ray's runtime env from its own defaults, so override
    # the hard-coded 0..7 mask to keep the teacher's physical NPU isolated.
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = train_devices
    os.environ["CUDA_VISIBLE_DEVICES"] = train_devices
    launch_options = {
        "before_ray_job_submit": start_opd_teacher,
        "extra_env_vars": {
            "ASCEND_RT_VISIBLE_DEVICES": train_devices,
            "CUDA_VISIBLE_DEVICES": train_devices,
            "ASCEND_TOOLKIT_HOME": "/usr/local/Ascend/ascend-toolkit/latest/",
            "ASCEND_HOME_PATH": "/usr/local/Ascend/ascend-toolkit/latest/",
            "HCCL_IF_IP": "127.0.0.1",
            "TP_SOCKET_IFNAME": "lo",
            "GLOO_SOCKET_IFNAME": "lo",
        },
    }
elif retool_devices:
    # execute_train() otherwise hardcodes 0..7 in its Ray runtime env,
    # even though this colocated job reserves only four physical NPUs.
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = retool_devices
    os.environ["CUDA_VISIBLE_DEVICES"] = retool_devices
    launch_options = {
        "extra_env_vars": {
            "ASCEND_RT_VISIBLE_DEVICES": retool_devices,
            "CUDA_VISIBLE_DEVICES": retool_devices,
        },
    }
    print(f"ReTool Ray-visible NPUs: {retool_devices}", flush=True)


def start_search_server():
    global search_process
    command = [sys.executable, str(Path(fork_root) / "examples/search-r1/local_dense_retriever/retrieval_server.py"),
        "--retriever_name", "bm25", "--index_path", os.environ["SLIME_SEARCH_INDEX"], "--topk", "3"]
    with search_log.open("w") as output:
        search_process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 120
    request = urllib.request.Request("http://127.0.0.1:8000/retrieve",
        data=json.dumps({"queries": ["France capital"], "topk": 3, "return_scores": False}).encode(),
        headers={"Content-Type": "application/json"})
    while time.monotonic() < deadline:
        if search_process.poll() is not None:
            raise RuntimeError(f"BM25 retriever exited; see {search_log}")
        try:
            with opener.open(request, timeout=3) as response:
                result = json.load(response)
            if "Paris" not in json.dumps(result):
                raise RuntimeError("Native retriever returned no matching fixture fact")
            print("Search-R1 native BM25 /retrieve ready", flush=True)
            return
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    raise TimeoutError(f"Native BM25 retriever did not become ready; see {search_log}")


if os.environ.get("SLIME_SEARCH_INDEX"):
    launch_options["before_ray_job_submit"] = start_search_server


def print_retool_worker_errors():
    # Ray's submitted-job traceback may omit the actor's real assertion
    # site. Keep a bounded excerpt from its local worker logs on failure.
    logs = Path("/tmp/ray/session_latest/logs")
    if not logs.exists():
        return
    try:
        candidates = list(logs.glob("worker-*.err"))
        candidates += list(logs.glob("python-core-worker-*.log"))
        for path in sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)[:4]:
            with path.open(errors="replace") as output:
                tail = "".join(deque(output, maxlen=35))
            if "Traceback" in tail or "AssertionError" in tail or "ERROR" in tail:
                print(f"Ray worker log tail ({path.name}):\n{tail}", flush=True)
    except OSError as log_error:
        print(f"could not read Ray worker diagnostics: {log_error}", flush=True)


try:
    module.execute_train(
        train_args=shlex.join(train_args),
        num_gpus_per_node=int(num_gpus),
        megatron_model_type=model_type,
        train_script=train_script,
        **launch_options,
    )
    if search_process:
        if search_process.poll() is not None:
            raise RuntimeError(f"BM25 retriever exited during training; see {search_log}")
except Exception:
    if retool_devices:
        print_retool_worker_errors()
    raise
finally:
    if search_process and search_process.poll() is None:
        os.killpg(search_process.pid, signal.SIGTERM)
        try:
            search_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(search_process.pid, signal.SIGKILL)
            search_process.wait()
    if teacher_process and teacher_process.poll() is None:
        os.killpg(teacher_process.pid, signal.SIGTERM)
        try:
            teacher_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(teacher_process.pid, signal.SIGKILL)
            teacher_process.wait()
PY
