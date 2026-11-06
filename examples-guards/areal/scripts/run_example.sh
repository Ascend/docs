#!/usr/bin/env bash
# Run one AReaL example from a CI working copy of the target tree.
# $1 is the example path relative to TARGET_ROOT. Overlay CLI args come from
# OVERLAY_ARGS (JSON array, serialized by the workflow from
# manifest.overlay_args); ${...} items are expanded from the job environment.
#
# AReaL examples are plain `python examples/.../foo.py <overrides>` entrypoints
# (Hydra-style key=value overrides), so no launcher is involved.
set -euo pipefail

EXAMPLE_REL="${1:?example path is required}"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
LAUNCH_PATH="$TARGET_ROOT/$EXAMPLE_REL"
[[ -f "$LAUNCH_PATH" ]] || { echo "no such example: $LAUNCH_PATH" >&2; exit 2; }

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
items = json.loads(raw)
if items in (None, ''):
    raise SystemExit(0)
tokens = []
for item in items:
    tokens.extend(shlex.split(os.path.expandvars(item), posix=True))
print(' '.join(shlex.quote(token) for token in tokens))
PY
}

eval "EXTRA_ARGS=( $(expand_overlay) )"
echo "running $LAUNCH_PATH with ${#EXTRA_ARGS[@]} overlay args"
if ((${#EXTRA_ARGS[@]})); then
  printf 'overlay arg: %q\n' "${EXTRA_ARGS[@]}"
fi

cd "$TARGET_ROOT"
# Repo root first so dotted module paths used by workflow kwargs resolve
# (e.g. boba_grpo.py's reward_fn "examples.math.boba_grpo.boba_reward_fn");
# the example dir covers sibling modules imported by the script itself.
export PYTHONPATH="$TARGET_ROOT:$(dirname "$LAUNCH_PATH")${PYTHONPATH:+:$PYTHONPATH}"
# Forked runtime logs (data-worker / data-router / data-gateway under
# <fileroot>/logs/) are NOT part of this job's stdout, so a crash there is
# otherwise invisible; dump them when the example fails.
dump_forked_logs() {
  local root="${AREAL_LOG_ROOT:-/tmp/areal/experiments/logs}"
  [[ -d "$root" ]] || return 0
  local f
  while IFS= read -r f; do
    echo "===== $f (tail -n 200) ====="
    tail -n 200 "$f" || true
   done < <(find "$root" -type f -name 'data-*.log' 2>/dev/null)
 }

# ---------------------------------------------------------------------------
# Shared helpers for the orchestration branches below: the online-RL entries
# all drive gateways over HTTP and wait on background processes.
# ---------------------------------------------------------------------------

# POST <base>/<path> with a Bearer key and a JSON body; $5 = client timeout
# (default 120s). Prints the response body on success; on failure prints the
# status + body head to stderr and prints nothing (rc stays 0 - callers test
# for empty output).
http_post() {
  "$PYTHON" - "$1" "$2" "$3" "$4" "${5:-120}" <<'PY'
import sys
import urllib.error
import urllib.request

base, path, key, body, timeout = sys.argv[1:6]
req = urllib.request.Request(
    f"{base}/{path}",
    data=body.encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
        print(resp.read().decode())
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}: {e.read().decode()[:300]}", file=sys.stderr)
except OSError as e:
    print(f"{type(e).__name__}: {e}", file=sys.stderr)
PY
}

# Extract a JSON field from stdin: json_pick '["api_key"]' (v1 flat session
# contract) or json_pick '["sessions"][0]["session_api_key"]' (v2 nested).
# Prints nothing on a parse failure.
json_pick() {
  "$PYTHON" -c "import json,sys; print(json.load(sys.stdin)$1)" 2>/dev/null || true
}

# True when the URL answers within 5s.
http_healthy() {
  "$PYTHON" -c "import urllib.request; urllib.request.urlopen('$1', timeout=5)" >/dev/null 2>&1
}

# Wait for "Proxy gateway available at <url>" in a background process's log.
# Echoes the gateway URL to STDOUT (callers capture it via $(...)); the
# heartbeat and error tails go to STDERR so they reach the job log without
# polluting the captured value. The URL charset is strict: areal log
# lines can carry ANSI colour bytes right after the URL, and a looser
# pattern would swallow the ESC byte into the address.
wait_gateway() {
  local log=$1 pid=$2 label=${3:-proc} gw i
  for i in $(seq 1 120); do
    gw=$(grep 'Proxy gateway available at' "$log" 2>/dev/null | tail -1 \
      | grep -oE 'https?://[A-Za-z0-9._:-]+' | tail -1 || true)
    [[ -n "$gw" ]] && { echo "$gw"; return 0; }
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$label exited before the gateway came up; tail of $log:" >&2
      tail -n 120 "$log" >&2
      return 1
    fi
    echo "[wait $((i * 10))s] $label: $(tail -n 1 "$log" 2>/dev/null | cut -c1-200)" >&2
    sleep 10
  done
  echo "$label gateway did not come up in 20min; tail of $log:" >&2
  tail -n 120 "$log" >&2
  return 1
}

# Launch the tau2 user-simulator vLLM on physical device 1 (the child env
# override replaces the job-wide ASCEND_RT_VISIBLE_DEVICES=0 - the rollout
# owns card 0) and wait for its /health. Sets USER_SIM_PID/USER_SIM_LOG;
# the caller's EXIT trap kills it. The multiple --served-model-name aliases
# absorb whatever model string the tau2 client sends (econfig.user_llm uses
# litellm "openai/..." naming) - vLLM matches the model field strictly.
start_user_sim() {
  local log=$1 i
  USER_SIM_PORT=18888
  USER_SIM_LOG="$log"
  ASCEND_RT_VISIBLE_DEVICES=1 "$PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$AREAL_MODEL_PATH" \
    --served-model-name default openai/self-hosted-Qwen2.5-72B self-hosted-Qwen2.5-72B \
    --port "$USER_SIM_PORT" \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    >"$USER_SIM_LOG" 2>&1 &
  USER_SIM_PID=$!
  echo "user-sim vLLM started (pid=$USER_SIM_PID, port=$USER_SIM_PORT, log=$USER_SIM_LOG)"
  for i in $(seq 1 120); do
    http_healthy "http://127.0.0.1:${USER_SIM_PORT}/health" && { echo "user-sim vLLM healthy"; return 0; }
    if ! kill -0 "$USER_SIM_PID" 2>/dev/null; then
      echo "user-sim vLLM exited early; tail of $USER_SIM_LOG:" >&2
      tail -n 120 "$USER_SIM_LOG" >&2
      return 1
    fi
    echo "[user-sim wait $((i * 5))s] $(tail -n 1 "$USER_SIM_LOG" 2>/dev/null | cut -c1-200)"
    sleep 5
  done
  echo "user-sim vLLM not healthy after 10min; tail of $USER_SIM_LOG:" >&2
  tail -n 120 "$USER_SIM_LOG" >&2
  return 1
}

# Bounded wait for a background process: return once the pid dies, once
# <done_pattern> appears in <logfile> (if given), or after <timeout_s>
# (TERM then KILL; sets WAIT_TIMED_OUT=1 so the caller can fail the job).
wait_bounded() {
  local pid=$1 timeout=$2 logfile=$3 pattern=${4:-}
  local waited=0
  WAIT_TIMED_OUT=0
  while kill -0 "$pid" 2>/dev/null; do
    if [[ -n "$pattern" ]] && grep -q "$pattern" "$logfile" 2>/dev/null; then
      sleep 3
      return 0
    fi
    if (( waited >= timeout )); then
      WAIT_TIMED_OUT=1
      kill "$pid" 2>/dev/null || true
      sleep 5
      kill -9 "$pid" 2>/dev/null || true
      return 0
    fi
    sleep 15
    waited=$((waited + 15))
  done
  return 0
}

run_rc=0

# Warm the guard import chain before the entry forks RPC guards: the data
# controller waits a hardcoded 30s for worker readiness (data_service/
# controller/controller.py:73 - no config/env outlet), while a cold
# `python -m areal.infra.rpc.guard` boot (areal.infra pulls torch/torch_npu)
# takes ~30s on these runners. gsm8k_sft already lost that race by under a
# second (guard ready at +30.8s). One throwaway import in this step warms
# the page cache for every guard the entry forks afterwards.
if [[ "$LAUNCH_PATH" == *.py ]]; then
  echo "warming guard import chain (torch/torch_npu page cache) ..."
  "$PYTHON" -c "import areal.infra.rpc.guard" >/dev/null 2>&1 || true
fi

case "$EXAMPLE_REL" in
  examples/experimental/inference_service/online_rollout.py)
    # Rollout-only online example: online_rollout.py starts the gateway then
    # BLOCKS in rollout_batch(data=None) waiting for EXTERNAL traffic. We
    # supply that traffic ourselves over HTTP (start_session ->
    # chat/completions -> set_reward -> refresh) - pure client calls, no
    # source patch - until it has collected batch_size trajectories and
    # prints "Rollout complete". (batch_size is pinned to 1 via the overlay.)
    ROLLOUT_LOG="$PWD/online_rollout.log"
    OR_ADMIN_KEY="sk-test123456"

    online_rollout_cleanup() {
      if [[ -n "${ROLLOUT_PID:-}" ]] && kill -0 "$ROLLOUT_PID" 2>/dev/null; then
        kill "$ROLLOUT_PID" 2>/dev/null || true
      fi
    }
    trap online_rollout_cleanup EXIT

    "$PYTHON" "$LAUNCH_PATH" "${EXTRA_ARGS[@]}" >"$ROLLOUT_LOG" 2>&1 &
    ROLLOUT_PID=$!

    GW=$(wait_gateway "$ROLLOUT_LOG" "$ROLLOUT_PID" rollout) || exit 1
    echo "inference gateway: $GW"

    # v2 session shape: request {"task_id","group_size"}; response
    # {"group_id","sessions":[{"session_id","session_api_key"}]}.
    SESS_KEY=""
    for attempt in $(seq 1 24); do
      BODY=$(http_post "$GW" rl/start_session "$OR_ADMIN_KEY" '{"task_id":"ci-online-rollout","group_size":1}' 60)
      SESS_KEY=$(printf '%s' "$BODY" | json_pick '["sessions"][0]["session_api_key"]')
      [[ -n "$SESS_KEY" ]] && break
      echo "[session attempt $attempt] no session yet"
      sleep 5
    done
    [[ -n "$SESS_KEY" ]] || { echo "start_session never returned a key"; tail -n 40 "$ROLLOUT_LOG"; exit 1; }
    echo "session: $SESS_KEY"

    CHAT_BODY='{"model":"default","messages":[{"role":"user","content":"What is 12 * 15 + 3?"}],"max_tokens":256}'
    CHAT=$(http_post "$GW" chat/completions "$SESS_KEY" "$CHAT_BODY" 600)
    [[ -n "$CHAT" ]] || { echo "chat/completions failed"; tail -n 40 "$ROLLOUT_LOG"; exit 1; }
    echo "chat completion: $(printf '%s' "$CHAT" | cut -c1-160)"

    # Marking the reward makes the data proxy report the trajectory ready; the
    # online workflow then picks it up and exports it (no v1-style refresh).
    REWARD=$(http_post "$GW" rl/set_reward "$SESS_KEY" '{"interaction_id":null,"reward":1.0}' 60)
    [[ -n "$REWARD" ]] || { echo "set_reward failed"; tail -n 40 "$ROLLOUT_LOG"; exit 1; }
    echo "reward applied: $REWARD"

    # online_rollout.py should now print "Rollout complete" and exit 0.
    wait_bounded "$ROLLOUT_PID" 600 "$ROLLOUT_LOG" 'Rollout complete'
    if grep -q 'Rollout complete' "$ROLLOUT_LOG" 2>/dev/null; then
      echo "online_rollout completed"
      grep 'Rollout complete' "$ROLLOUT_LOG" | tail -1
    else
      run_rc=1
      echo "no 'Rollout complete' seen; tail of $ROLLOUT_LOG:"
      tail -n 120 "$ROLLOUT_LOG"
    fi
    ;;
  examples/experimental/inference_service/tau2_rollout.py)
    # Rollout-only tau2: the SAME real agent + user simulator as the
    # training entry (Tau2AgentWorkflow, airline domain), but driven through
    # RolloutControllerV2 directly - no PPO stack - and self-terminating:
    # it iterates its dataloader, prints per-batch reward stats, then
    # "Rollout complete (N batches)" and exits 0. Same card layout as the
    # training entry: rollout vLLM pinned to card 0 via npu_devices, the
    # user-simulator vLLM on card 1 (child env override).
    TR_LOG="$PWD/tau2_rollout.log"

    tau2_rollout_cleanup() {
      if [[ -n "${USER_SIM_PID:-}" ]] && kill -0 "$USER_SIM_PID" 2>/dev/null; then
        kill "$USER_SIM_PID" 2>/dev/null || true
      fi
    }
    trap tau2_rollout_cleanup EXIT

    # 1) User-simulator vLLM on physical device 1 (identical launch to the
    #    tau2 training entry; start_user_sim sets USER_SIM_PID/USER_SIM_LOG).
    start_user_sim "$PWD/tau2_rollout_usersim.log" || exit 1

    # 2) tau2_rollout in the background with a heartbeat: it self-drives
    #    the batch loop (agent sessions through the gateway it hosts, user
    #    turns against the sidecar above) and exits when the dataloader is
    #    exhausted. Its whole output goes to $TR_LOG - without a heartbeat
    #    the job log stays silent for the entire run (engine init alone is
    #    ~3min; the dataloader iterates the FULL tau2 split, and
    #    get_tau2_dataset pads anything under 128 tasks up to 128).
    "$PYTHON" "$LAUNCH_PATH" "${EXTRA_ARGS[@]}" >"$TR_LOG" 2>&1 &
    TR_PID=$!
    while kill -0 "$TR_PID" 2>/dev/null; do
      echo "[tau2_rollout] $(tail -n 1 "$TR_LOG" 2>/dev/null | cut -c1-200)"
      sleep 30
    done
    tr_rc=0
    wait "$TR_PID" || tr_rc=$?
    if (( tr_rc == 0 )) && grep -q 'Rollout complete' "$TR_LOG"; then
      echo "tau2_rollout completed"
      grep 'Rollout complete' "$TR_LOG" | tail -1
      grep 'avg_reward' "$TR_LOG" | tail -3 || true
    else
      run_rc=${tr_rc:-1}
      echo "tau2_rollout failed (rc=$tr_rc, or no 'Rollout complete'); tail of $TR_LOG:"
      tail -n 120 "$TR_LOG"
    fi
    ;;
  *)
    "$PYTHON" "$LAUNCH_PATH" "${EXTRA_ARGS[@]}" || run_rc=$?
    ;;
esac

if ((run_rc != 0)); then
  echo "example exited rc=$run_rc; dumping forked runtime logs"
  dump_forked_logs
fi
exit "$run_rc"
