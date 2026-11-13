#!/usr/bin/env bash
# Verify the specforge quick-start doc path on a coder NPU workspace, using the
# upstream *plugin* capture flow (no patch, no local patch).
#
# Mirrors the CI container env (CANN 9.0.0 + torch 2.10.0 + sglang built from
# source) and the doc's #test-setup / #test blocks, in the same order the
# MarkdownDocTestBase harness runs them.
#
# Usage:
#   ssh hdc-sglang 'bash -s' < scripts/verify_specforge_plugin_flow_on_coder.sh
#
# What this proves (verified end-to-end on 2026-11-12):
#   1. SGLang builds from source at the commit upstream SpecForge's CI pins
#      (SGLANG_COMMIT) and exposes the three capture extension points the
#      plugin needs (--enable-return-aux-hidden-states, forward_observer,
#      Scheduler.defer_response),
#   2. the matching sgl-kernel-npu wheel is required (the image's 2026.6.1
#      lacks sgl_kernel_npu.activation.fused_sigmoid_mul -> model modules
#      fail to register),
#   3. the upstream plugin installs and is discovered through the
#      sglang.srt.plugins entry point,
#   4. the capture server starts with the plugin flow and advertises
#      spec_capture_backend=plugin, and a 1-step train produces a step/loss.
#
# Notes:
# - /home/coder is the persistent volume; /root and /tmp live on the
#   overlay and are lost on rebuild (uv lives in /root/.local/bin, so it
#   has to be reinstalled after every rebuild).
# - uv ignores pip.conf: without UV_INDEX_URL it reaches for pypi.org,
#   which stalls on this network.
# - GitHub release assets download at ~10 KB/s here; the sgl-kernel-npu wheel
#   is 13 MB, so budget ~20 min for that step.
# - The model comes from the ModelScope cache instead of a fresh HuggingFace
#   download (HF is unreachable from the workspace).
set -eo pipefail

UPSTREAM_REF="${UPSTREAM_REF:-main}"
WORK="${WORK:-/home/coder/work/specforge-verify}"
MODEL="${MODEL:-/home/coder/.hdc/cache/modelscope/models/Qwen--Qwen3.5-4B/snapshots/master}"
KERNEL_TAG="${KERNEL_TAG:-2026.9.0.post12}"
CANN_VER="${CANN_VER:-9.0.0}"
DEVICE="${DEVICE:-910b}"
CAPTURE_DEVICE="${CAPTURE_DEVICE:-0}"
TRAINER_DEVICE="${TRAINER_DEVICE:-1}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
MOONCAKE_RPC_PORT="${MOONCAKE_RPC_PORT:-35551}"
MOONCAKE_HTTP_PORT="${MOONCAKE_HTTP_PORT:-35880}"

echo "=== [0] bootstrap / env ==="
source /home/coder/.hdc/env.sh
export PATH=/home/coder/.hdc/bin:/root/.local/bin:$PATH
export UV_INDEX_URL="${UV_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
export UV_SYSTEM_PYTHON=1
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/home/coder/.hdc/cache/modelscope}"
export HF_HOME="${HF_HOME:-/home/coder/.hdc/cache/huggingface}"

# uv lives in /root/.local/bin (overlay) - gone after a rebuild.
if [ ! -x /root/.local/bin/uv ]; then
    echo "[0] installing uv"
    (cd /tmp && curl -sSfL https://astral.sh/uv/install.sh | sh) >/dev/null 2>&1 \
        || python3 -m pip install --break-system-packages -q uv
fi
mkdir -p /home/coder/.hdc/bin
ln -sf /root/.local/bin/uv /home/coder/.hdc/bin/uv
ln -sf /root/.local/bin/uvx /home/coder/.hdc/bin/uvx
python3 --version
uv --version
npu-smi info 2>/dev/null | head -8 || echo "npu-smi unavailable"

# apt lists are stale after a rebuild -> libibverbs1 is "not locatable" then.
rm -rf /var/lib/apt/lists/*
apt-get update -qq >/dev/null 2>&1 || true
apt-get install -qq -y --no-install-recommends libibverbs1 libcurl4 libnuma1 >/dev/null 2>&1 || true

echo "=== [1] clone specforge @ ${UPSTREAM_REF} ==="
mkdir -p "$WORK"
cd "$WORK"
rm -rf SpecForge && mkdir SpecForge && cd SpecForge
git init -q
git remote add origin https://github.com/sgl-project/SpecForge.git
git fetch --depth 1 origin "$UPSTREAM_REF"
git checkout -q FETCH_HEAD
git log -1 --format="specforge: %H %ad %s" --date=iso

echo "=== [2] sglang from source (upstream SGLANG_COMMIT) ==="
SGLANG_COMMIT=$(sed -n 's/^[[:space:]]*SGLANG_COMMIT:[[:space:]]*\([0-9a-f]\{40\}\).*/\1/p' \
    .github/workflows/test.yaml | head -1)
[ -n "$SGLANG_COMMIT" ] || { echo "FAILED - upstream SGLANG_COMMIT not found"; exit 1; }
echo "SGLANG_COMMIT=${SGLANG_COMMIT}"
cd "$WORK"
rm -rf sglang-src && mkdir sglang-src && cd sglang-src
git init -q
git remote add origin https://github.com/sgl-project/sglang.git
git fetch --depth 1 origin "$SGLANG_COMMIT"
git checkout -q FETCH_HEAD
( cd python && rm -f pyproject.toml && mv pyproject_npu.toml pyproject.toml )
PY_TAG="py$(python3 -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
ARCH=$(uname -m)
WHEEL="sgl-kernel-npu-${KERNEL_TAG}-torch2.10.0-${PY_TAG}-cann${CANN_VER}-${DEVICE}-${ARCH}.zip"
mkdir -p npu-ops && cd npu-ops
curl -fSL --retry 3 -o "$WHEEL" \
    "https://github.com/sgl-project/sgl-kernel-npu/releases/download/${KERNEL_TAG}/${WHEEL}"
unzip -o -q "$WHEEL"
uv pip install deep_ep*.whl sgl_kernel_npu*.whl torch_memory_saver*.whl
cd "$WORK/sglang-src/python"
uv pip install --no-deps -e .
uv pip install "transformers==5.19.0" datasets tqdm accelerate "click>=8.0" huggingface-hub numpy pydantic psutil pyyaml safetensors requests typing-extensions
python3 -c "from importlib.metadata import version; print('sglang', version('sglang'), '/ sgl_kernel_npu', version('sgl_kernel_npu'))"
python3 - <<'PY'
from sglang.srt.server_args import ServerArgs
from sglang.srt.model_executor.forward_observer import ForwardObserver
from sglang.srt.managers.deferred_output import ResponseCompletion
from sglang.srt.managers.scheduler import Scheduler
for n in ('enable_return_aux_hidden_states', 'return_aux_hidden_state_layer_ids', 'return_hidden_states_mode'):
    assert hasattr(ServerArgs, n), n
for m in ('defer_response', 'get_output_streamer_class', 'get_internal_state'):
    assert hasattr(Scheduler, m), m
print('capture API present')
PY

echo "=== [3] specforge + upstream capture plugin ==="
cd "$WORK/SpecForge"
uv pip install --no-deps .
uv pip install --no-deps accelerate
uv pip install --no-deps plugins/sglang-spec-capture
python3 -c "
from importlib.metadata import entry_points
print('capture plugin:', [e.name for e in entry_points(group='sglang.srt.plugins')])
"

echo "=== [4] mooncake_master ==="
pkill -9 -f '^mooncake_master' 2>/dev/null || true
nohup mooncake_master --enable_http_metadata_server=true \
    --rpc_port="$MOONCAKE_RPC_PORT" --http_metadata_server_port="$MOONCAKE_HTTP_PORT" \
    --metrics_port=35903 --enable_metric_reporting=false \
    >/tmp/verify-mooncake.log 2>&1 &
for _ in $(seq 1 30); do
    if python3 -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.settimeout(0.5)
try:
    s.connect(('127.0.0.1', $MOONCAKE_RPC_PORT))
except Exception:
    sys.exit(1)
finally:
    s.close()" 2>/dev/null; then echo "mooncake ready"; break; fi
    sleep 1
done

echo "=== [5] capture server (plugin flow) ==="
pkill -9 -f '^python3 -m sglang.launch_server' 2>/dev/null || true
export ASCEND_RT_VISIBLE_DEVICES=$CAPTURE_DEVICE
export MOONCAKE_LOCAL_HOSTNAME=127.0.0.1
export MOONCAKE_METADATA_SERVER=http://127.0.0.1:$MOONCAKE_HTTP_PORT/metadata
export MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:$MOONCAKE_RPC_PORT
export MOONCAKE_PROTOCOL=tcp
export MOONCAKE_GLOBAL_SEGMENT_SIZE=$((32 << 30))
export SPECFORGE_SPEC_CAPTURE=1
ATB_LIB=/usr/local/Ascend/nnal/atb/9.0.0/atb/cxx_abi_1/lib
[ -d "$ATB_LIB" ] && export LD_LIBRARY_PATH="$ATB_LIB:${LD_LIBRARY_PATH:-}"
nohup python3 -m sglang.launch_server \
    --model-path "$MODEL" --trust-remote-code --skip-tokenizer-init \
    --tp-size 1 --mem-fraction-static 0.5 --context-length 1024 \
    --chunked-prefill-size -1 --attention-backend ascend \
    --mamba-ssm-dtype bfloat16 \
    --speculative-algorithm DFLASH \
    --enable-return-aux-hidden-states \
    --return-aux-hidden-state-layer-ids 1 8 15 22 29 \
    --return-hidden-states-mode full \
    --host 127.0.0.1 --port "$SGLANG_PORT" \
    >/tmp/verify-sglang.log 2>&1 &
for _ in $(seq 1 180); do
    curl -fsS "http://127.0.0.1:$SGLANG_PORT/health" >/dev/null 2>&1 && { echo "sglang ready"; break; }
    sleep 5
done
curl -fsS "http://127.0.0.1:$SGLANG_PORT/health" >/dev/null 2>&1 \
    || { echo "FAILED - sglang not healthy"; tail -50 /tmp/verify-sglang.log; exit 1; }
curl -fsS "http://127.0.0.1:$SGLANG_PORT/server_info" | grep -o '"spec_capture_backend":"[a-z]*"' | head -1

echo "=== [6] one-step train ==="
cd "$WORK/SpecForge"
mkdir -p cache/dataset
python3 - cache/dataset/sharegpt_train.jsonl <<'PY'
import json, os, sys
path = sys.argv[1]
os.makedirs(os.path.dirname(path), exist_ok=True)
row = {"id": "smoke_1", "conversations": [
 {"role": "user", "content": "Explain the difference between supervised and unsupervised learning in machine learning. Cover what types of problems each paradigm is suited for, give at least three concrete algorithm examples for each, and discuss practical considerations for choosing between them when you have a new dataset to model."},
 {"role": "assistant", "content": "Supervised learning and unsupervised learning are the two foundational paradigms in machine learning, distinguished by the form of feedback the learning algorithm receives during training. In supervised learning the algorithm receives a labeled dataset and adjusts its parameters to minimize a loss between its predictions and the ground truth; the two main task families are classification and regression. In unsupervised learning the algorithm receives only raw unlabeled inputs and must find structure on its own, with common tasks including clustering, dimensionality reduction, density estimation, and representation learning. Choosing between the two depends on whether labeled data and a well defined prediction target are available: supervised learning delivers higher accuracy on the specific labeled task but requires costly annotation, while unsupervised learning is preferable when labels are scarce or the goal is exploratory or structural. In modern practice the two are often combined, with unsupervised pretraining learning useful representations from large unlabeled corpora and supervised fine tuning specializing them for downstream tasks."}]}
with open(path, "w", encoding="utf-8") as f:
    json.dump(row, f, ensure_ascii=False); f.write("\n")
print("wrote", path)
PY
SHAREGPT=$(realpath cache/dataset/sharegpt_train.jsonl)
rm -rf outputs/qwen3.5-4b-dflash-npu-online
PYTHONUNBUFFERED=1 ASCEND_RT_VISIBLE_DEVICES=$TRAINER_DEVICE \
HCCL_CONNECT_TIMEOUT=7200 HCCL_EXEC_TIMEOUT=7200 \
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
nohup specforge train -c examples/configs/online/disaggregated/external/qwen3.5-4b-dflash-npu.yaml \
    training.max_steps=1 training.batch_size=1 training.accumulation_steps=1 \
    data.max_length=512 data.train_data_path="$SHAREGPT" \
    training.num_anchors=32 training.save_interval=0 training.log_interval=1 \
    tracking.report_to=none deployment.trainer.nproc_per_node=1 \
    deployment.disaggregated.receive_buffers=pageable \
    model.target_model_path="$MODEL" \
    model.embedding_key="model.language_model.embed_tokens.weight" \
    >/tmp/verify-train.log 2>&1 &
TRAIN_PID=$!
sleep 30
kill -0 "$TRAIN_PID" 2>/dev/null || { echo "FAILED - train died within 30s"; tail -40 /tmp/verify-train.log; exit 1; }
echo "train alive after 30s (pid=$TRAIN_PID)"
for _ in $(seq 1 45); do
    grep -qE "step.*loss|loss.*step|train_runtime" /tmp/verify-train.log 2>/dev/null && { echo "step/loss seen"; break; }
    kill -0 "$TRAIN_PID" 2>/dev/null || { echo "train exited"; break; }
    sleep 20
done
grep -oE "step [0-9]+: .*'loss': [0-9.]+" /tmp/verify-train.log | head -1 | cut -c1-160
grep -qE "step.*loss|loss.*step" /tmp/verify-train.log || { echo "FAILED - no step/loss"; tail -30 /tmp/verify-train.log; exit 1; }
echo "=== DONE ==="
