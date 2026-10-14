#!/usr/bin/env bash
# Run one example from a CI working copy of the target tree.
# Overlay CLI args come from OVERLAY_ARGS (JSON array). Never
# git add/commit/push.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <example-relpath>" >&2
  exit 2
fi

EXAMPLE_REL="$1"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
CI_OUTPUT_DIR="${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"
EXEC_REL="${EXEC:?EXEC is required}"
PROFILE="${PROFILE:-}"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

EXAMPLE_PATH="$TARGET_ROOT/$EXAMPLE_REL"
if [[ ! -e "$EXAMPLE_PATH" ]]; then
  echo "example not found: $EXAMPLE_PATH" >&2
  exit 1
fi

EXEC_PATH="$TARGET_ROOT/$EXEC_REL"
if [[ ! -f "$EXEC_PATH" ]]; then
  echo "exec not found: $EXEC_PATH" >&2
  exit 1
fi

if [[ "$PROFILE" != "ascend-direct" ]]; then
  echo "unknown profile for run: ${PROFILE:-empty} (supported: ascend-direct)" >&2
  exit 2
fi

mkdir -p "$CI_OUTPUT_DIR"

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  PYTHON=python
fi

source_cann() {
  export PATH="/usr/local/sbin:/usr/local/bin:$PATH"
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
}

stop_pid() {
  local p="$1"
  kill -TERM "$p" 2>/dev/null || true
  local _
  for _ in $(seq 1 30); do
    if ! kill -0 "$p" 2>/dev/null; then
      wait "$p" 2>/dev/null || true
      return 0
    fi
    sleep 1
  done
  kill -KILL "$p" 2>/dev/null || true
  wait "$p" 2>/dev/null || true
}

read_overlay_args() {
  local overlay_file token
  overlay_file=$(mktemp)
  "$PYTHON" "$SCRIPT_DIR/expand_overlay_args.py" >"$overlay_file"
  EXTRA_ARGS=()
  while IFS= read -r token || [[ -n "$token" ]]; do
    [[ -n "$token" ]] || continue
    EXTRA_ARGS+=("$token")
  done <"$overlay_file"
  rm -f "$overlay_file"
}

read_overlay_args
echo "running $EXEC_PATH for $EXAMPLE_REL with ${#EXTRA_ARGS[@]} overlay args"
if ((${#EXTRA_ARGS[@]})); then
  printf 'overlay arg: %q\n' "${EXTRA_ARGS[@]}"
fi

source_cann
export GLOG_logtostderr=1
cd "$TARGET_ROOT"

TARGET_LOG="$CI_OUTPUT_DIR/target.log"
INIT_LOG="$CI_OUTPUT_DIR/initiator.log"
COMBINED_LOG="$CI_OUTPUT_DIR/transfer_engine_ascend_direct_perf.log"

launch=(stdbuf -oL -eL "$EXEC_PATH")

"${launch[@]}" \
  --mode=target \
  --device_logicid=0 \
  --local_server_name=127.0.0.1:12345 \
  --metadata_server=P2PHANDSHAKE \
  "${EXTRA_ARGS[@]}" >"$TARGET_LOG" 2>&1 &
target_pid=$!
# shellcheck disable=SC2064
trap "stop_pid $target_pid" EXIT

endpoint=""
for _ in $(seq 1 60); do
  if ! kill -0 "$target_pid" 2>/dev/null; then
    echo "target exited before listen" >&2
    cat "$TARGET_LOG" >&2
    exit 1
  fi
  if endpoint=$("$PYTHON" "$SCRIPT_DIR/parse_listen_endpoint.py" "$TARGET_LOG"); then
    break
  fi
  sleep 1
done
if [[ -z "$endpoint" ]]; then
  echo "target did not print a listening endpoint" >&2
  cat "$TARGET_LOG" >&2
  exit 1
fi
echo "target listening on $endpoint"

set +e
"${launch[@]}" \
  --mode=initiator \
  --device_logicid=1 \
  --local_server_name=127.0.0.1:12346 \
  --metadata_server=P2PHANDSHAKE \
  --segment_id="$endpoint" \
  --operation=write \
  "${EXTRA_ARGS[@]}" >"$INIT_LOG" 2>&1
init_ec=$?
set -e
cat "$INIT_LOG"

{
  echo '===== target ====='
  cat "$TARGET_LOG"
  echo '===== initiator ====='
  cat "$INIT_LOG"
} >"$COMBINED_LOG"

if [[ "$init_ec" != 0 ]]; then
  echo "initiator exited $init_ec" >&2
  exit "$init_ec"
fi
if grep -q 'Failed to install Ascend transport' "$COMBINED_LOG"; then
  echo "Ascend Direct transport failed to install" >&2
  exit 1
fi
if grep -qE 'getTransferStatus FAILED|Sync data transfer timeout' "$INIT_LOG"; then
  echo "initiator reported a failed or timed-out transfer" >&2
  exit 1
fi
if ! grep -q 'Success to initialize adxl engine' "$COMBINED_LOG"; then
  echo "ADXL engine did not initialize; check /etc/hccn.conf and card mounts" >&2
  exit 1
fi
if ! grep -q 'Test completed:' "$INIT_LOG"; then
  echo "initiator log missing Test completed" >&2
  exit 1
fi
# registerLocalMemory location is "npu:<logicid>". Keep mem type:device
# as the library line that is printed only for a device buffer.
if ! grep -Eq 'npu:[0-9]+|mem type:device' "$COMBINED_LOG"; then
  echo "run used no NPU device buffer; check card mounts and USE_ASCEND_DIRECT" >&2
  exit 1
fi
