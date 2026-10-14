#!/usr/bin/env bash
# Run one specforge example from a CI working copy of the target tree.
# $1 is the manifest entry path. EXEC, when set, names the launchable
# file relative to the target root; otherwise path itself must be a
# launchable file. Overlay CLI args come from OVERLAY_ARGS (JSON array,
# possibly []). Shell and python examples that do not pass "$@" get it
# attached in this working copy only (last command line). YAML recipes
# are dispatched to `specforge train -c <path>` so the manifest entry
# can be a typed run config without a wrapper script.
# Never git add/commit/push.
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
if [[ ! -f "$LAUNCH_PATH" && ! -L "$LAUNCH_PATH" ]]; then
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
  # The workflow serializes manifest.overlay_args as JSON. Each array item is
  # exactly ONE CLI argument (a JSON array of strings is argv-shaped), passed
  # through verbatim after ${CI_OUTPUT_DIR} / ${SPECFORGE_MODEL_PATH}-style
  # env expansion. Do NOT shlex.split the items: posix shlex strips embedded
  # double quotes, which corrupts structured specforge overrides —
  # trainer_cuda_visible_devices=["1"] would arrive as [1] (int, rejected by
  # the List[str] schema) and
  # capture_servers=[{"port":30000,...}] as YAML flow {port:30000,...} whose
  # colon-without-space keys all parse wrong (run 35811075823, both
  # managed-local legs). Quoted values in the manifest are YAML/JSON syntax,
  # not shell quoting, and must survive intact.
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
    tokens.append(os.path.expandvars(item))
print(' '.join(shlex.quote(token) for token in tokens))
PY
}

# expand_overlay exits nonzero on malformed OVERLAY_ARGS (bad JSON / non-array
# / non-string items). Command substitution inside eval would swallow that
# status and silently continue with zero overlays — the run would then execute
# the recipe defaults (full training scale, missing fixture/model paths), so
# make the failure fatal here.
if ! OVERLAY_TOKENS="$(expand_overlay)"; then
  echo "FAILED - OVERLAY_ARGS expansion error (see above)" >&2
  exit 1
fi
# NOTE: the eval argument is deliberately NOT wrapped in double quotes.
# OVERLAY_TOKENS is a space-joined list of shlex.quote'd tokens; eval must
# re-parse them with that quoting intact. A `eval "EXTRA_ARGS=( ... )"` wrapper
# would treat every embedded double quote as a string delimiter and silently
# strip it — exactly what corrupted the structured managed-local overlays in
# run 35811075823 (["1"] -> [1], {"port":30000,...} -> {port:30000,...}).
eval EXTRA_ARGS=\($OVERLAY_TOKENS\)

echo "running $LAUNCH_PATH with ${#EXTRA_ARGS[@]} overlay args"
if ((${#EXTRA_ARGS[@]})); then
  printf 'overlay arg: %q\n' "${EXTRA_ARGS[@]}"
fi

ensure_passthrough() {
  # Shell examples that already forward "$@" need no patch; otherwise
  # attach it in this CI working copy only, on the last non-comment
  # line (the tail of the example's main command). YAML recipes are
  # dispatched below and never get "$@" attached — specforge reads
  # its overrides from the CLI argument list, which is what we already
  # pass through EXTRA_ARGS.
  local script="$1"
  case "$script" in
    *.yaml|*.yml) return 0 ;;
  esac
  [[ "$script" == *.sh ]] || return 0
  if grep -qE '"\$@"' "$script"; then
    echo "example already has \"\$@\"; skipping patch"
    return
  fi
  "$PYTHON" - "$script" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
lines = path.read_text(encoding='utf-8').splitlines(keepends=True)
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
        lines[i] = raw.rstrip() + ' "$@"' + newline
        path.write_text(''.join(lines), encoding='utf-8')
        print(f'patched {path} to pass "$@" on last command line')
        raise SystemExit(0)
raise SystemExit(f'{path}: cannot find a command line to attach "$@"')
PY
}

ensure_passthrough "$LAUNCH_PATH"

# The `external/` disaggregated recipes expect an already-running Mooncake
# master + SGLang capture server (their endpoints are hard-coded in the YAML
# as 127.0.0.1:35551 / 35880 / 30000). specforge only plays producer+consumer,
# it never spawns those (only `managed_local` does). The examples engine has a
# single "Run example" step, so we bring the two prerequisites up here first —
# mirroring Quick-start-Ascend.md's smoke-start-mooncake / smoke-start-sglang —
# then run `specforge train` against them and tear them down afterwards.
start_mooncake() {
  pkill -9 -f '^mooncake_master' 2>/dev/null || true
  MOONCAKE_RPC_PORT="${SPECFORGE_MOONCAKE_RPC_PORT:-35551}"
  MOONCAKE_HTTP_PORT="${SPECFORGE_MOONCAKE_HTTP_PORT:-35880}"
  nohup mooncake_master \
    --enable_http_metadata_server=true \
    --rpc_port="$MOONCAKE_RPC_PORT" \
    --http_metadata_server_port="$MOONCAKE_HTTP_PORT" \
    --metrics_port=35903 \
    --enable_metric_reporting=false \
    >/tmp/examples-mooncake.log 2>&1 &
  MOONCAKE_PID=$!
  for _ in $(seq 1 30); do
    if "$PYTHON" -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(0.5)
try:
    s.connect(('127.0.0.1', $MOONCAKE_RPC_PORT))
except Exception:
    sys.exit(1)
finally:
    s.close()
" 2>/dev/null; then
      echo "mooncake master ready (rpc $MOONCAKE_RPC_PORT, pid=$MOONCAKE_PID)"
      return 0
    fi
    sleep 1
  done
  echo "FAILED - mooncake_master did not bind $MOONCAKE_RPC_PORT in 30s" >&2
  tail -50 /tmp/examples-mooncake.log >&2
  return 1
}

# Resolve the capture contract (method + aux layer ids + run_id) for a typed
# YAML recipe. The spec-capture flags differ per algorithm (dflash/dflash2/domino
# -> method dflash; dspark -> dspark; eagle3 -> eagle3) and per draft config
# (dflash_config.target_layer_ids), so they cannot be hard-coded. We derive them
# from the same typed composition root specforge uses for managed-local launch:
# load_config -> resolve_run -> resolve_server_capture_contract. Only the model
# path override is needed (target config must be resolvable locally).
resolve_capture() {
  "$PYTHON" - "$LAUNCH_PATH" "$SPECFORGE_MODEL_PATH" <<'PY'
import json, sys
from specforge.config import load_config
from specforge.application import resolve_run
from specforge.training.capture_contract import resolve_server_capture_contract

recipe, model = sys.argv[1], sys.argv[2]
cfg = load_config(recipe, [f"model.target_model_path={model}"])
resolved = resolve_run(cfg)
contract = resolve_server_capture_contract(cfg, algorithm=resolved.algorithm)
print(f"CAP_METHOD={contract.method!r}")
print(f"CAP_AUX={' '.join(str(i) for i in contract.aux_layer_ids)!r}")
print(f"CAP_RUN_ID={cfg.run_id!r}")
PY
}

# Offline colocated recipes consume pre-captured hidden states from disk, so the
# engine must first run the producer: scripts/prepare_hidden_states.py, which
# drives a local in-proc SGLang capture through the model's native capture
# hooks (no server, no mooncake, no spec-capture patch). Its arguments come
# from the same typed config the trainer uses — load the recipe WITH the
# overlay args applied so max_length / train_data_path / hidden_states_path
# match exactly what `specforge train` will see.
resolve_offline() {
  "$PYTHON" - "$LAUNCH_PATH" "${EXTRA_ARGS[@]}" <<'PY'
import sys

from specforge.config import load_config

recipe = sys.argv[1]
cfg = load_config(recipe, list(sys.argv[2:]))
print(f"PREP_STRATEGY={cfg.training.strategy!r}")
print(f"PREP_DRAFT_CFG={cfg.model.draft_model_config!r}")
print(f"PREP_CHAT_TEMPLATE={cfg.data.chat_template!r}")
print(f"PREP_MAX_LENGTH={cfg.data.max_length}")
print(f"PREP_HIDDEN_STATES={cfg.data.hidden_states_path!r}")
print(f"PREP_TRUST={int(bool(cfg.model.trust_remote_code))}")
print(f"CAP_RUN_ID={cfg.run_id!r}")
PY
}

# One-shot in-proc capture for an offline colocated recipe. Runs on the capture
# card and exits before the trainer starts, so the card is free again. The atb
# lib path matters: the qwen3-family in-proc capture lazy-loads libatb.so (same
# reason start_sglang_capture exports it below).
#
# Input data: offline recipes only carry data.hidden_states_path (the config
# validator forbids combining it with data.train_data_path), so the raw
# conversations fed to the capture come from the engine's fixture, not from a
# train overlay.
prepare_hidden_states() {
  local prep_data="${SPECFORGE_PREPARE_DATA:-${FIXTURE_DIR:?FIXTURE_DIR is required}/sharegpt_train.jsonl}"
  local trust_args=()
  if [[ "$PREP_TRUST" == "1" ]]; then
    trust_args=(--trust-remote-code)
  fi
  local draft_cfg="$PREP_DRAFT_CFG"
  if [[ "$draft_cfg" != /* ]]; then
    draft_cfg="$TARGET_ROOT/$draft_cfg"
  fi
  local atb_lib=/usr/local/Ascend/nnal/atb/9.0.0/atb/cxx_abi_1/lib
  local ld_path="${LD_LIBRARY_PATH:-}"
  if [[ -d "$atb_lib" ]]; then
    ld_path="$atb_lib:$ld_path"
  fi
  # 8B BF16 weights need mem-fraction > 0.537 on a 32G card; 0.65 covers both
  # 4B and 8B (verified on 910B4).
  ASCEND_RT_VISIBLE_DEVICES="${SPECFORGE_CAPTURE_DEVICE:-0}" \
  LD_LIBRARY_PATH="$ld_path" \
  PYTHONUNBUFFERED=1 \
    torchrun --nproc_per_node=1 "$TARGET_ROOT/scripts/prepare_hidden_states.py" \
      --target-model-path "${SPECFORGE_MODEL_PATH:?SPECFORGE_MODEL_PATH is required}" \
      --data-path "$prep_data" \
      --output-path "$PREP_HIDDEN_STATES" \
      --chat-template "$PREP_CHAT_TEMPLATE" \
      --max-length "$PREP_MAX_LENGTH" \
      --tp-size 1 --batch-size 1 --num-samples 1 \
      --strategy "$PREP_STRATEGY" \
      --draft-model-config "$draft_cfg" \
      --sglang-attention-backend ascend \
      --sglang-mem-fraction-static "${SPECFORGE_PREPARE_MEM_FRACTION:-0.65}" \
      ${trust_args[@]+"${trust_args[@]}"}
}

start_sglang_capture() {
  local method="$1" aux_layer_ids="$2"
  pkill -9 -f '^python -m sglang\.launch_server' 2>/dev/null || true
  CAPTURE_DEVICE="${SPECFORGE_CAPTURE_DEVICE:-0}"
  SGLANG_PORT="${SPECFORGE_SGLANG_PORT:-30000}"
  SGLANG_HEALTH_TIMEOUT="${SPECFORGE_SGLANG_HEALTH_TIMEOUT:-600}"
  MOONCAKE_RPC_PORT="${SPECFORGE_MOONCAKE_RPC_PORT:-35551}"
  MOONCAKE_HTTP_PORT="${SPECFORGE_MOONCAKE_HTTP_PORT:-35880}"
  export ASCEND_RT_VISIBLE_DEVICES=$CAPTURE_DEVICE
  export MOONCAKE_LOCAL_HOSTNAME=127.0.0.1
  export MOONCAKE_METADATA_SERVER=http://127.0.0.1:$MOONCAKE_HTTP_PORT/metadata
  export MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:$MOONCAKE_RPC_PORT
  export MOONCAKE_PROTOCOL=tcp
  export MOONCAKE_GLOBAL_SEGMENT_SIZE=$((32<<30))
  ATB_LIB=/usr/local/Ascend/nnal/atb/9.0.0/atb/cxx_abi_1/lib
  if [[ -d "$ATB_LIB" ]]; then
    export LD_LIBRARY_PATH="$ATB_LIB:${LD_LIBRARY_PATH:-}"
  fi
  : "${SPECFORGE_MODEL_PATH:?SPECFORGE_MODEL_PATH is required}"
  nohup python -m sglang.launch_server \
    --model-path "$SPECFORGE_MODEL_PATH" \
    --trust-remote-code \
    --skip-tokenizer-init \
    --tp-size 1 \
    --mem-fraction-static 0.65 \
    --context-length 1024 \
    --chunked-prefill-size -1 \
    --attention-backend ascend \
    --enable-spec-capture --spec-capture-method "$method" \
    --spec-capture-aux-layer-ids $aux_layer_ids \
    --host 127.0.0.1 --port "$SGLANG_PORT" \
    >/tmp/examples-sglang.log 2>&1 &
  SGLANG_PID=$!
  HEALTH_DEADLINE=$((SGLANG_HEALTH_TIMEOUT / 5))
  for _ in $(seq 1 "$HEALTH_DEADLINE"); do
    if curl -fsS "http://127.0.0.1:$SGLANG_PORT/health" >/dev/null 2>&1; then
      echo "sglang capture server ready (pid=$SGLANG_PID)"
      return 0
    fi
    sleep 5
  done
  echo "FAILED - SGLang capture server not healthy after ${SGLANG_HEALTH_TIMEOUT}s" >&2
  tail -50 /tmp/examples-sglang.log >&2
  return 1
}

cleanup_services() {
  pkill -9 -f '^python -m sglang\.launch_server' 2>/dev/null || true
  pkill -9 -f '^mooncake_master' 2>/dev/null || true
}

# Dispatch on file extension:
#   *.yaml / *.yml: specforge typed run configs — invoke `specforge train -c`
#     and forward EXTRA_ARGS as dotted section.field=value overrides.
#     - offline/colocated recipes: pre-generate the hidden states with the
#       in-proc capture first (prepare_hidden_states), then train.
#     - managed-local recipes: one `specforge train` owns the whole stack.
#     - external recipes: bring up Mooncake + SGLang capture server first.
#   *.sh: shell example, cd into its directory and bash it.
#   * (default): python entry point, run from target root.
case "$LAUNCH_PATH" in
  *.yaml|*.yml)
    cd "$TARGET_ROOT"
    if [[ "$LAUNCH_PATH" == *"/offline/"* ]]; then
      # Offline colocated recipes: the trainer reads hidden states from disk;
      # the engine runs the producer (in-proc capture) first, then the
      # pure-torch trainer. No external services, no mooncake, no spec-capture
      # server patch.
      if ! OFFLINE_VARS="$(resolve_offline)"; then
        echo "FAILED - resolve_offline could not load the offline recipe config" >&2
        exit 1
      fi
      eval "$OFFLINE_VARS"
      rm -rf "outputs/${CAP_RUN_ID}"
      if [[ -n "${PREP_HIDDEN_STATES:-}" ]]; then
        rm -rf "${PREP_HIDDEN_STATES%/}"
      fi
      prepare_hidden_states
      ATB_LIB=/usr/local/Ascend/nnal/atb/9.0.0/atb/cxx_abi_1/lib
      TRAIN_LD_PATH="${LD_LIBRARY_PATH:-}"
      if [[ -d "$ATB_LIB" ]]; then
        TRAIN_LD_PATH="$ATB_LIB:$TRAIN_LD_PATH"
      fi
      ASCEND_RT_VISIBLE_DEVICES="${SPECFORGE_TRAINER_DEVICE:-1}" \
      HCCL_CONNECT_TIMEOUT=7200 HCCL_EXEC_TIMEOUT=7200 \
      PYTHONUNBUFFERED=1 \
      PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
      LD_LIBRARY_PATH="$TRAIN_LD_PATH" \
        specforge train -c "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
    elif [[ "$LAUNCH_PATH" == *"/managed-local/"* ]]; then
      # managed-local recipes: one `specforge train` owns the whole single-node
      # stack (Mooncake master + capture server + producer + consumer via the
      # launch_plan managed supervisor). No external services to start, and the
      # supervisor assigns device ordinals from the recipe's managed_local
      # block, so no ASCEND_RT_VISIBLE_DEVICES pin here either.
      eval "$(resolve_capture)"
      rm -rf "outputs/${CAP_RUN_ID}"
      # The supervisor spawns the capture server as a child of this process, so
      # it inherits our env — export the atb lib here for the same reason
      # start_sglang_capture does (Qwen3.5-family models lazy-load libatb.so
      # through torch_npu op_plugin; without it the server dies at first
      # _npu_reshape_and_cache with OSError from torch.ops.load_library).
      ATB_LIB=/usr/local/Ascend/nnal/atb/9.0.0/atb/cxx_abi_1/lib
      if [[ -d "$ATB_LIB" ]]; then
        export LD_LIBRARY_PATH="$ATB_LIB:${LD_LIBRARY_PATH:-}"
      fi
      PYTHONUNBUFFERED=1 \
      PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
        specforge train -c "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
    else
      # external recipes: bring the Mooncake master + SGLang capture server up
      # first, then run `specforge train` against them.
      eval "$(resolve_capture)"
      rm -rf "outputs/${CAP_RUN_ID}"
      start_mooncake
      start_sglang_capture "$CAP_METHOD" "$CAP_AUX"
      trap cleanup_services EXIT
      ASCEND_RT_VISIBLE_DEVICES="${SPECFORGE_TRAINER_DEVICE:-1}" \
      HCCL_CONNECT_TIMEOUT=7200 HCCL_EXEC_TIMEOUT=7200 \
      PYTHONUNBUFFERED=1 \
      PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
        specforge train -c "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
    fi
    ;;
  *.sh)
    cd "$(dirname "$LAUNCH_PATH")"
    bash "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
    ;;
  *)
    cd "$TARGET_ROOT"
    "$PYTHON" "$LAUNCH_PATH" "${EXTRA_ARGS[@]}"
    ;;
esac
