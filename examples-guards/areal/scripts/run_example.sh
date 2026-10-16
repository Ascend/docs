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
# examples/scaffolding/*.py use package-relative imports (from ._compat
# import ...), so they must run as modules, not as plain scripts.
#
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
# Echoes the gateway URL; returns 1 (with a log tail on stderr) when the
# process dies first or 20min elapse. The URL charset is strict: areal log
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
    echo "[wait $((i * 10))s] $label: $(tail -n 1 "$log" 2>/dev/null | cut -c1-200)"
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
  examples/scaffolding/*.py)
    MODULE="${EXAMPLE_REL%.py}"
    MODULE="${MODULE//\//.}"
    "$PYTHON" -m "$MODULE" "${EXTRA_ARGS[@]}" || run_rc=$?
    ;;
  examples/hermes/train.py)
    # Online RL: train.py blocks waiting for an externally-driven session
    # lifecycle, so a plain foreground run would hang until timeout.
    # Orchestrate one full loop instead: background trainer (starts the
    # vLLM rollout engine + proxy gateway) -> Agent Service (HermesAgent)
    # -> start_session over HTTP (mint session key; see step 3 for why the
    # CLI is bypassed) -> one piped conversation via hermes_loop (EOF exits
    # cleanly) -> set_reward -> start_session refresh (ends the session,
    # exports the trajectory) -> the trainer's _OnlineAgent unblocks, trains
    # its single step and exits.
    HERMES_ADMIN_KEY="areal-agent-ci"
    TRAIN_LOG="$PWD/hermes_train.log"
    AGENT_LOG="$PWD/hermes_agent.log"

    hermes_cleanup() {
      areal agent stop >/dev/null 2>&1 || true
      if [[ -n "${TRAINER_PID:-}" ]] && kill -0 "$TRAINER_PID" 2>/dev/null; then
        kill "$TRAINER_PID" 2>/dev/null || true
      fi
    }
    trap hermes_cleanup EXIT

    # 1) Trainer in the background; its log is the source of truth for
    #    the proxy/inference gateway address (rl_trainer logs
    #    "Proxy gateway available at http://host:port" once online mode
    #    is up - engine load takes minutes).
    "$PYTHON" "$LAUNCH_PATH" "${EXTRA_ARGS[@]}" >"$TRAIN_LOG" 2>&1 &
    TRAINER_PID=$!

    GW=$(wait_gateway "$TRAIN_LOG" "$TRAINER_PID" trainer) || exit 1
    echo "inference gateway: $GW"

    # 2) Agent Service with the Hermes agent (run blocks until the stack
    #    is healthy, then exits leaving the service up).
    #    hermes-agent enables 30+ toolsets by default; their definitions ride
    #    EVERY /v1/responses request and overflow vllm's context, making the
    #    agent death-loop its own compressor. HermesAgent reads
    #    HERMES_DISABLED_TOOLSETS in __init__ (hermes.py:118) - i.e. in the
    #    Agent Service worker process - so it must be EXPORTED here, BEFORE
    #    `areal agent run`, for the workers to inherit it. Keep only "clarify".
    export HERMES_DISABLED_TOOLSETS="web,search,x_search,vision,video,image_gen,video_gen,computer_use,terminal,skills,browser,cronjob,file,tts,todo,memory,context_engine,session_search,project,code_execution,delegation,homeassistant,kanban,discord,discord_admin,yuanbao,feishu_doc,feishu_drive,spotify,debugging,safe,coding"
    if ! areal agent run --agent examples.hermes.hermes.HermesAgent \
        --admin-api-key "$HERMES_ADMIN_KEY" --force >"$AGENT_LOG" 2>&1; then
      echo "areal agent run failed; tail of $AGENT_LOG:"
      tail -n 120 "$AGENT_LOG"
      exit 1
    fi
    AGENT_GW=$(grep -oE 'gateway=[A-Za-z0-9._:/-]+' "$AGENT_LOG" | tail -1 | cut -d= -f2 || true)
    if [[ -z "$AGENT_GW" ]]; then
      AGENT_GW=$(areal agent status 2>/dev/null | grep -oE 'https?://[A-Za-z0-9._:-]+' | tail -1 || true)
    fi
    if [[ -z "$AGENT_GW" ]]; then
      echo "agent gateway not found; tail of $AGENT_LOG:"
      tail -n 120 "$AGENT_LOG"
      exit 1
    fi
    echo "agent gateway: $AGENT_GW"

    # 3) Mint the session key on the inference gateway. Talk HTTP directly
    #    instead of examples/hermes/start_session.py: that CLI hardcodes the
    #    V2 gateway contract (HTTP 201 + {"sessions":[{"session_api_key":..}]})
    #    while the v1 proxy gateway returns 200 + flat {"api_key","session_id"}
    #    (proxy_gateway.py NEW path returns the worker response as-is) - the
    #    CLI dies on a perfectly successful session, the retry loop then
    #    spawns zombie sessions until all 3 workers' capacity is exhausted
    #    and everything 429s. 429 ("no capacity") is retried: workers only
    #    register in the readiness queue after the trainer is fully up.
    SESS_KEY=""
    for attempt in $(seq 1 24); do
      BODY=$(http_post "$GW" rl/start_session sk-areal-ci '{"task_id":"ci-hermes-smoke"}' 180)
      SESS_KEY=$(printf '%s' "$BODY" | json_pick '["api_key"]')
      [[ -n "$SESS_KEY" ]] && break
      echo "[session attempt $attempt] no session yet (capacity/registering)"
      sleep 5
    done
    if [[ -z "$SESS_KEY" ]]; then
      echo "start_session never returned a session key"
      tail -n 40 "$TRAIN_LOG"
      exit 1
    fi
    echo "session: $SESS_KEY"

    # 4) Drive ONE agent turn directly over HTTP, bypassing hermes_loop.py:
    #    that CLI hardcodes its httpx client timeout to 120s (:47), which the
    #    ~11.5K-token Hermes round-trip exceeds on this NPU - it dies with
    #    "Read timed out" while the server is still working. Sending the same
    #    /v1/responses POST ourselves with a 600s client timeout avoids the
    #    hardcoded limit (no source patch). inf-* route the agent's LLM calls
    #    through the inference gateway under the session key (self-evolution
    #    capture); model id "default" is InferenceDataProxy's served name.
    agent_turn() {
      "$PYTHON" - "$AGENT_GW" "$HERMES_ADMIN_KEY" "$1" "$GW" "$SESS_KEY" <<'PY'
import json
import sys
import urllib.error
import urllib.request

base, key, content, inf_base, sess_key = sys.argv[1:6]
payload = {
    "input": [{"type": "message", "content": content}],
    "model": "hermes-agent",
    "user": "hermes-ci",
    "inf_base_url": inf_base,
    "inf_model": "default",
    "session_api_key": sess_key,
}
req = urllib.request.Request(
    f"{base.rstrip('/')}/v1/responses",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode())
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}: {e.read().decode()[:300]}", file=sys.stderr)
    sys.exit(1)
except OSError as e:
    print(f"{type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)
texts = []
for item in data.get("output", []):
    if item.get("type") == "message":
        for block in item.get("content", []):
            if block.get("type") == "output_text":
                texts.append(block["text"])
print("".join(texts) if texts else json.dumps(data)[:300])
PY
    }

    if ! AGENT_REPLY=$(agent_turn "Say hello and introduce yourself in one sentence."); then
      echo "agent turn failed; tails:"
      tail -n 60 "$AGENT_LOG"
      tail -n 40 "$TRAIN_LOG"
      exit 1
    fi
    echo "agent reply: $AGENT_REPLY"

    # 5) Reward the session, then refresh it: the refresh call (start_session
    #    with the existing api_key) ends the old session, waits for the
    #    training pipeline to cycle and starts a new one with the same key -
    #    the trajectory export unblocks the trainer's _OnlineAgent. Both go
    #    through http_post (v1 contract: 200 on success). Refresh can 429 while
    #    the pipeline cycles (refresh_timeout on the gateway side), so retry.
    REWARD_RESP=$(http_post "$GW" rl/set_reward "$SESS_KEY" '{"reward":1.0}' 180)
    if [[ -z "$REWARD_RESP" ]]; then
      echo "set_reward failed; tail of $TRAIN_LOG:"
      tail -n 40 "$TRAIN_LOG"
      exit 1
    fi
    echo "reward applied: $REWARD_RESP"

    REFRESHED=""
    for attempt in $(seq 1 12); do
      BODY=$(http_post "$GW" rl/start_session sk-areal-ci \
        "{\"task_id\":\"ci-hermes-smoke\",\"api_key\":\"$SESS_KEY\"}" 180)
      NEW_KEY=$(printf '%s' "$BODY" | json_pick '["api_key"]')
      if [[ -n "$NEW_KEY" ]]; then
        REFRESHED=1
        echo "session refreshed (key: $NEW_KEY)"
        break
      fi
      echo "[refresh attempt $attempt] pipeline still cycling"
      sleep 10
    done
    if [[ -z "$REFRESHED" ]]; then
      echo "session refresh never succeeded; the trainer may be stuck waiting"
      tail -n 40 "$TRAIN_LOG"
      exit 1
    fi

    # 6) Trainer trains its single step and exits. Bounded wait: a rejected
    #    trajectory (empty/error-only session) leaves the trainer waiting
    #    for the NEXT session forever - instead of holding the runner until
    #    the job timeout, fail after 15 min with the trainer log dumped.
    wait_bounded "$TRAINER_PID" 900 "$TRAIN_LOG"
    if (( WAIT_TIMED_OUT )); then
      run_rc=1
      echo "trainer still running after 900s - likely waiting for another session (rejected trajectory?); tail of $TRAIN_LOG:"
      tail -n 120 "$TRAIN_LOG"
    elif wait "$TRAINER_PID"; then
      echo "hermes online-rl loop completed"
    else
      run_rc=$?
      echo "trainer exited rc=$run_rc; tail of $TRAIN_LOG:"
      tail -n 120 "$TRAIN_LOG"
    fi
    ;;
  examples/openclaw/train.py)
    # Online RL via the proxy gateway (same architecture as hermes): the agent
    # runtime is any OpenAI-compatible client. We drive one episode DIRECTLY
    # over HTTP instead of the upstream examples/openclaw/demo_lifecycle.py,
    # whose /chat/completions client timeout is hardcoded to 30s (:264) - the
    # NPU generation exceeds that, so it dies with "Read timed out" before
    # set_reward. Doing the same calls ourselves with a long client timeout
    # avoids the hardcoded limit (no source patch).
    OC_ADMIN_KEY="sk-test123456"
    TRAIN_LOG="$PWD/openclaw_train.log"

    openclaw_cleanup() {
      if [[ -n "${TRAINER_PID:-}" ]] && kill -0 "$TRAINER_PID" 2>/dev/null; then
        kill "$TRAINER_PID" 2>/dev/null || true
      fi
    }
    trap openclaw_cleanup EXIT

    # 1) Trainer in the background; its log carries the gateway address.
    "$PYTHON" "$LAUNCH_PATH" "${EXTRA_ARGS[@]}" >"$TRAIN_LOG" 2>&1 &
    TRAINER_PID=$!

    GW=$(wait_gateway "$TRAIN_LOG" "$TRAINER_PID" trainer) || exit 1
    echo "inference gateway: $GW"

    # 2) start_session (admin) -> per-session key. 429 ("no capacity") is
    #    retried: workers only register once the trainer is fully up.
    SESS_KEY=""
    for attempt in $(seq 1 24); do
      BODY=$(http_post "$GW" rl/start_session "$OC_ADMIN_KEY" '{"task_id":"ci-openclaw-smoke"}' 60)
      SESS_KEY=$(printf '%s' "$BODY" | json_pick '["api_key"]')
      [[ -n "$SESS_KEY" ]] && break
      echo "[session attempt $attempt] no session yet"
      sleep 5
    done
    [[ -n "$SESS_KEY" ]] || { echo "start_session never returned a key"; tail -n 40 "$TRAIN_LOG"; exit 1; }
    echo "session: $SESS_KEY"

    # 3) chat/completions under the session key - REAL generation (tokens +
    #    logprobs recorded for RL). Long client timeout: the NPU generates
    #    slowly (~1.5 tok/s), far past demo_lifecycle's hardcoded 30s.
    CHAT_BODY='{"model":"default","messages":[{"role":"system","content":"You are a helpful math tutor. Show your work step by step."},{"role":"user","content":"Solve step by step: What is 12 * 15 + 3?"}],"temperature":0.7,"max_tokens":256}'
    CHAT=$(http_post "$GW" chat/completions "$SESS_KEY" "$CHAT_BODY" 600)
    [[ -n "$CHAT" ]] || { echo "chat/completions failed"; tail -n 40 "$TRAIN_LOG"; exit 1; }
    echo "chat completion: $(printf '%s' "$CHAT" | cut -c1-200)"

    # 4) set_reward (session key), then refresh: start_session with the key
    #    ends the session and exports the trajectory -> the trainer's
    #    _OnlineAgent unblocks and trains its single step.
    REWARD=$(http_post "$GW" rl/set_reward "$SESS_KEY" '{"reward":1.0}' 60)
    [[ -n "$REWARD" ]] || { echo "set_reward failed"; tail -n 40 "$TRAIN_LOG"; exit 1; }
    echo "reward applied: $REWARD"

    REFRESHED=""
    for attempt in $(seq 1 12); do
      BODY=$(http_post "$GW" rl/start_session "$OC_ADMIN_KEY" \
        "{\"task_id\":\"ci-openclaw-smoke\",\"api_key\":\"$SESS_KEY\"}" 130)
      NEW_KEY=$(printf '%s' "$BODY" | json_pick '["api_key"]')
      [[ -n "$NEW_KEY" ]] && { REFRESHED=1; echo "session refreshed (key: $NEW_KEY)"; break; }
      echo "[refresh attempt $attempt] pipeline still cycling"
      sleep 10
    done
    [[ -n "$REFRESHED" ]] || { echo "refresh never succeeded; trainer may be stuck"; tail -n 40 "$TRAIN_LOG"; exit 1; }

    # 5) Trainer trains its single step and exits (bounded).
    wait_bounded "$TRAINER_PID" 1200 "$TRAIN_LOG"
    if (( WAIT_TIMED_OUT )); then
      run_rc=1
      echo "trainer wait timed out; tail of $TRAIN_LOG:"
      tail -n 120 "$TRAIN_LOG"
    elif wait "$TRAINER_PID"; then
      echo "openclaw online-rl loop completed"
    else
      run_rc=$?
      echo "trainer exited rc=$run_rc; tail of $TRAIN_LOG:"
      tail -n 120 "$TRAIN_LOG"
    fi
    ;;
  examples/camel/train.py)
    # train.py defines its config dataclass (AgentRLConfig) at module scope.
    # Whether run as a plain script or via `python -m`, the class's __module__
    # is "__main__" (verified), so the engine worker cannot import it and
    # deserialize_value degrades the config to a plain dict -> config.seed
    # AttributeError. Importing the file as a module instead gives the class
    # the path "examples.camel.train.AgentRLConfig" (importable in the worker,
    # repo root is on PYTHONPATH) and we call main() explicitly.
    "$PYTHON" -c "import sys; from examples.camel.train import main; main(sys.argv[1:])" "${EXTRA_ARGS[@]}" || run_rc=$?
    ;;
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
  examples/experimental/inference_service/human_in_the_loop_demo.py)
    # The demo brings its own driver (launches online_rollout.py itself,
    # temporarily points ~/.zeroclaw/config.toml at the local gateway, runs
    # 4 HITL rounds via the zeroclaw CLI, then checks the rollout log). Its
    # exit code is LOOSE though: zeroclaw failures are caught (empty answer,
    # reward 0) and a missing "Rollout complete" only prints a warning
    # before "Demo finished" + exit 0. Make the pass criterion strict:
    # exit 0 AND "Rollout complete" in the demo output (on success it echoes
    # the rollout log tail incl. "Databatch detected").
    HITL_LOG="$PWD/hitl_demo.log"
    # Diagnose the zeroclaw build BEFORE the demo swallows its stderr
    # (check_output discards the captured output on failure): run the demo's
    # exact invocation shape against the placeholder config
    # (custom:http://localhost:1). rc=2 + usage text = the installed CLI
    # lacks the demo's interface - zeroclaw.com's installer ships the
    # OFFICIAL build, while the demo targets the dhh1995 fork. A connection
    # error instead means the syntax is fine.
    # Dump the official CLI surface first: version (for later pinning) and
    # whether `agent -m` / --session-state-file exist at all.
    { zeroclaw --version; echo "--------"; zeroclaw --help; echo "--------"; zeroclaw agent --help; } \
      >"$PWD/zc_help.log" 2>&1 || true
    echo "--- zeroclaw version / --help / agent --help (head -n 60) ---"
    head -n 60 "$PWD/zc_help.log" 2>/dev/null || true
    timeout 30 zeroclaw agent -m "probe" \
      --session-state-file "$PWD/zc_probe.json" >"$PWD/zc_probe.log" 2>&1 \
      || echo "zeroclaw probe rc=$? (usage error = wrong build; connection error = syntax OK)"
    cat "$PWD/zc_probe.log" 2>/dev/null || true
    demo_rc=0
    "$PYTHON" "$LAUNCH_PATH" "${EXTRA_ARGS[@]}" >"$HITL_LOG" 2>&1 || demo_rc=$?
    if (( demo_rc == 0 )) && grep -q 'Rollout complete' "$HITL_LOG"; then
      echo "hitl demo completed"
      grep 'Rollout complete' "$HITL_LOG" | tail -1
    else
      run_rc=1
      echo "hitl demo failed (rc=$demo_rc, or no 'Rollout complete' - hollow run?); tail of $HITL_LOG:"
      tail -n 80 "$HITL_LOG"
    fi
    ;;
  examples/tau2/train.py)
    # Tau2 keeps its REAL user simulator (solo_mode=false): host the user
    # LLM locally on the runner's 2nd card with the SAME weights (zero extra
    # download) and point econfig.user_llm_base_url at it (the overlay sets
    # the URL). The trainer is pinned to card 0 via npu_devices, so the two
    # never collide. Multiple --served-model-name aliases absorb whatever
    # model string the tau2 client sends (econfig.user_llm uses litellm
    # "openai/..." naming) - vLLM matches the model field strictly.
    tau2_cleanup() {
      if [[ -n "${USER_SIM_PID:-}" ]] && kill -0 "$USER_SIM_PID" 2>/dev/null; then
        kill "$USER_SIM_PID" 2>/dev/null || true
      fi
    }
    trap tau2_cleanup EXIT

    # 1) User-simulator vLLM on physical device 1 via the shared helper
    #    (sets USER_SIM_PID/USER_SIM_LOG; the EXIT trap above kills it).
    start_user_sim "$PWD/tau2_user_sim.log" || exit 1

    # 2) Trainer in the foreground; its exit code is authoritative (the
    #    user-sim server is torn down by the EXIT trap).
    tau2_rc=0
    "$PYTHON" "$LAUNCH_PATH" "${EXTRA_ARGS[@]}" || tau2_rc=$?
    if (( tau2_rc == 0 )); then
      echo "tau2 training completed"
    else
      run_rc=$tau2_rc
      echo "tau2 trainer exited rc=$tau2_rc; user-sim tail:"
      tail -n 40 "$USER_SIM_LOG"
    fi
    ;;
  examples/agent_service/run_agent_service.py)
    # The Agent Service hosts NO model itself: ClaudeAgent shells out via
    # claude-agent-sdk, whose ClaudeSDKClient spawns the Claude Code CLI -
    # the API traffic goes wherever ANTHROPIC_BASE_URL points. Point it at a
    # SIDECAR online_rollout stack: its v2 gateway serves the Anthropic
    # /v1/messages route and the data proxy translates Anthropic->OpenAI onto
    # the local vLLM, so no real Anthropic key is needed. The service's own
    # stdin loop is interactive: hold it open with a FIFO writer, drive ONE
    # /v1/responses request (the shape documented in the example README),
    # then close the FIFO -> EOF -> clean exit 0 = pass. The sidecar's
    # rollout_batch never completes (no RL sessions are driven); it is just
    # the LLM backend and is killed at the end.
    AS_ADMIN_KEY="areal-agent-admin"
    AS_SIDECAR_KEY="sk-test123456"
    AS_SVC_LOG="$PWD/agent_service.log"
    AS_ROLLOUT_LOG="$PWD/agent_service_rollout.log"
    AS_FIFO="$PWD/agent_service_stdin.fifo"

    agent_service_cleanup() {
      exec 3>&- 2>/dev/null || true
      if [[ -n "${AS_SVC_PID:-}" ]] && kill -0 "$AS_SVC_PID" 2>/dev/null; then
        kill "$AS_SVC_PID" 2>/dev/null || true
      fi
      if [[ -n "${AS_ROLLOUT_PID:-}" ]] && kill -0 "$AS_ROLLOUT_PID" 2>/dev/null; then
        kill "$AS_ROLLOUT_PID" 2>/dev/null || true
      fi
    }
    trap agent_service_cleanup EXIT

    # 1) Sidecar: online_rollout.py (vLLM + gateway serving /v1/messages,
    #    registered model name "default").
    "$PYTHON" examples/experimental/inference_service/online_rollout.py \
      --config examples/experimental/inference_service/online_rollout.yaml \
      rollout.backend=vllm:d1p1t1 \
      actor.backend=fsdp:d1p1t1 \
      actor.path="$AREAL_MODEL_PATH" \
      train_dataset.batch_size=1 \
      vllm.gpu_memory_utilization=0.25 \
      vllm.max_model_len=16384 \
      gconfig.max_new_tokens=256 \
      rollout.setup_timeout=1800 \
      >"$AS_ROLLOUT_LOG" 2>&1 &
    AS_ROLLOUT_PID=$!

    AS_INF_GW=$(wait_gateway "$AS_ROLLOUT_LOG" "$AS_ROLLOUT_PID" sidecar) || exit 1
    echo "sidecar inference gateway: $AS_INF_GW"

    # 2) The Agent Service, stdin held open by a FIFO writer on fd 3 (a plain
    #    background run would hit EOF immediately and exit). Claude Code CLI
    #    env: traffic -> the sidecar gateway; nonessential phones-home
    #    disabled (those hosts are unreachable from the CN runners and would
    #    stall the CLI); generous API timeout for the slow NPU generation.
    mkfifo "$AS_FIFO"
    ANTHROPIC_BASE_URL="$AS_INF_GW" \
    ANTHROPIC_API_KEY="$AS_SIDECAR_KEY" \
    CLAUDE_MODEL=default \
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
    DISABLE_TELEMETRY=1 \
    API_TIMEOUT_MS=600000 \
      "$PYTHON" "$LAUNCH_PATH" "${EXTRA_ARGS[@]}" <"$AS_FIFO" >"$AS_SVC_LOG" 2>&1 &
    AS_SVC_PID=$!
    exec 3>"$AS_FIFO"

    # 3) Wait for the service gateway ("Gateway: http://..."), then /health.
    AS_AGENT_GW=""
    for i in $(seq 1 60); do
      AS_AGENT_GW=$(grep 'Gateway:' "$AS_SVC_LOG" 2>/dev/null | tail -1 \
        | grep -oE 'https?://[A-Za-z0-9._:-]+' | tail -1 || true)
      [[ -n "$AS_AGENT_GW" ]] && break
      kill -0 "$AS_SVC_PID" 2>/dev/null || break
      sleep 5
    done
    if [[ -n "$AS_AGENT_GW" ]]; then
      for i in $(seq 1 24); do
        http_healthy "${AS_AGENT_GW}/health" && break
        kill -0 "$AS_SVC_PID" 2>/dev/null || break
        sleep 5
      done
    fi
    if [[ -z "$AS_AGENT_GW" ]] || ! http_healthy "${AS_AGENT_GW}/health"; then
      echo "agent service gateway not healthy; tail of $AS_SVC_LOG:"
      tail -n 120 "$AS_SVC_LOG"
      exit 1
    fi
    echo "agent service gateway: $AS_AGENT_GW"

    # 4) ONE conversation through the service (the README's documented
    #    request shape). The worker spawns the Claude Code CLI, whose API
    #    calls land on the sidecar's /v1/messages -> local vLLM.
    AS_RESP=$(http_post "$AS_AGENT_GW" v1/responses "$AS_ADMIN_KEY" \
      '{"input":[{"type":"message","content":"Say hello in one sentence."}],"model":"claude-agent","user":"ci-agent-service"}' 600)
    if [[ -z "$AS_RESP" ]] || ! printf '%s' "$AS_RESP" | grep -q 'completed'; then
      run_rc=1
      echo "agent turn failed; response head + service tail:"
      printf '%s\n' "$AS_RESP" | cut -c1-400
      tail -n 80 "$AS_SVC_LOG"
    else
      echo "agent reply: $(printf '%s' "$AS_RESP" | grep -oE '"text": *"[^"]{0,120}' | head -1)"
    fi

    # 5) Close the FIFO writer -> the interactive loop sees EOF -> cleanup +
    #    exit 0. The service's exit code is the pass criterion.
    exec 3>&-
    wait_bounded "$AS_SVC_PID" 120 "$AS_SVC_LOG"
    if (( WAIT_TIMED_OUT == 0 )); then
      as_rc=0
      wait "$AS_SVC_PID" || as_rc=$?
      if (( as_rc == 0 )) && (( run_rc == 0 )); then
        echo "agent_service completed"
      else
        run_rc=${as_rc:-1}
        echo "agent service exited rc=$as_rc; tail of $AS_SVC_LOG:"
        tail -n 120 "$AS_SVC_LOG"
      fi
    else
      run_rc=1
      echo "agent service did not exit after EOF; tail of $AS_SVC_LOG:"
      tail -n 120 "$AS_SVC_LOG"
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
  *.sh)
    # Shell examples dispatch with bash (generic guard contract: .sh -> bash,
    # .py -> python).
    bash "$LAUNCH_PATH" "${EXTRA_ARGS[@]}" || run_rc=$?
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
