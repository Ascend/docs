#!/usr/bin/env bash
# Prepare the CI environment for one supported AReaL example.
# $1 is the manifest profile. Unknown profiles fail before any install.
#
# CI base image is ghcr.io/hwvanici/areal_npu, which already has the full stack:
# CANN, torch, vLLM-Ascend, Megatron, MindSpeed, huggingface_hub etc.
# We should NOT reinstall these; just install the target AReaL source and assets.
# Exception (section 1b): the vllm/vllm-ascend pair is swapped to the 0.22.1
# line to match the guarded v2.x code's vllm layout.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"

SUPPORTED_PROFILES="areal-math-sft areal-swe-sft areal-online-rollout areal-tau2-grpo"
TAU2_PREP=0
case "$PROFILE" in
  areal-math-sft) MODEL_ID="Qwen/Qwen3-1.7B" ;;
  # swe/train_sft.py: dataset path must contain an "swe" token to route
  # into the SWE pipeline (fixture swe_sft_pairs.jsonl via FIXTURE_DIR).
  areal-swe-sft) MODEL_ID="Qwen/Qwen3-1.7B" ;;
  # online_rollout.py: rollout-only online demo; no extra runtime to install
  # (drive is run_example.sh's online_rollout branch).
  areal-online-rollout) MODEL_ID="Qwen/Qwen3-1.7B" ;;
  # tau2_rollout.py: rollout-only tau2 (agent + user simulator); section 1f
  # installs the tau2-bench fork + data, run_example.sh's tau2_rollout branch
  # hosts the user-sim vLLM on card 1.
  areal-tau2-grpo) MODEL_ID="Qwen/Qwen3-1.7B"; TAU2_PREP=1 ;;
  *)
    echo "unknown profile: ${PROFILE} (supported: ${SUPPORTED_PROFILES})" >&2
    exit 1
    ;;
esac

if [[ -z "${TARGET_ROOT:-}" || -z "${GITHUB_WORKSPACE:-}" || -z "${GITHUB_ENV:-}" ]]; then
  echo "TARGET_ROOT / GITHUB_WORKSPACE / GITHUB_ENV must be set" >&2
  exit 2
fi

# -------------------------------------------------------
# Shared helpers: mirror pip installs and shallow clones - one place
# for the mirror URLs.
# -------------------------------------------------------
pip_tuna() {
  uv pip install --system --index-url https://pypi.tuna.tsinghua.edu.cn/simple "$@"
}

pip_aliyun() {
  uv pip install --system --index-url https://mirrors.aliyun.com/pypi/simple "$@"
}

shallow_clone() {
  local url=$1 branch=$2 dest=$3 attempt
  # CI 出口经 gh-proxy/squid，长传输偶发 curl 18/early EOF 截断；
  # 失败时目录可能残留，重试前清理，最多 3 次。
  for attempt in 1 2 3; do
    rm -rf "$dest"
    if git clone --depth 1 --branch "$branch" "$url" "$dest"; then
      return 0
    fi
    echo "shallow_clone failed (attempt ${attempt}/3): ${url}@${branch}" >&2
    sleep 15
  done
  return 1
}

# -------------------------------------------------------
# 1. Install the AReaL source under test.
# The official image ships its own copy of AReaL, but we need to use
# the checked-out TARGET_ROOT version because it may contain fixes.
# -------------------------------------------------------
command -v uv >/dev/null 2>&1 || pip install -q uv
uv pip install --no-deps -e "$TARGET_ROOT" --system

# -------------------------------------------------------
# 1b. vLLM stack: swap the image's 0.23 line for 0.22.1.
# The image ships the ascend-v1.0.5 branch stack (vllm/vllm-ascend
# 0.23.0). The guarded v2.x code imports the pre-0.23 vllm layout
# (vllm.entrypoints.openai.utils / vllm.entrypoints.utils exist
# through 0.22.1, removed in 0.23) - without this swap every
# rollout-backed entry dies on
#   ModuleNotFoundError: No module named 'vllm.entrypoints.openai.utils'
#
# Recipe per vllm-ascend's own Dockerfile at v0.22.1rc1:
#   - vllm-ascend 0.22.1rc1 PyPI wheel with --no-deps: its pins would
#     re-resolve the frozen image stack (torch_npu 2.10.0.post2 etc);
#   - vllm 0.22.1 from SOURCE with VLLM_TARGET_DEVICE=empty (pure
#     Python, no compiled kernels - the ascend plugin supplies the
#     device layer). NEVER install vllm from PyPI here: the wheel is
#     built against torch==2.11.0, incompatible with the image's
#     torch 2.10.0 and areal's torch<2.11 pin. --no-deps keeps the
#     image stack untouched, --no-build-isolation lets setup.py see
#     the system torch;
#   - triton-ascend stays as the image's 3.2.2 (same 3.2.x generation
#     as this line's 3.2.1);
#   - the source build additionally needs setuptools-rust/setuptools-scm
#     in the system env: --no-build-isolation skips the declared
#     build-system requires, but setup.py imports both unconditionally.
#     The optional rust extension is skipped when cargo is absent (same
#     as vllm-ascend's own image, which ships no rust toolchain).
# Applies to ALL profiles: fsdp-only entries (gsm8k_sft) double as
# the control that the swap does not disturb the vllm-free path.
# TODO(vllm-0.23): drop this block once upstream ships a v2.x NPU
# image or vllm-ascend catches up with mainline's target layout.
# -------------------------------------------------------
VLLM_SRC="$GITHUB_WORKSPACE/vllm-0.22.1-src"
shallow_clone https://github.com/vllm-project/vllm.git v0.22.1 "$VLLM_SRC"
pip_aliyun --no-deps vllm-ascend==0.22.1rc1
pip_aliyun "setuptools-rust>=1.9.0" "setuptools-scm>=8.0"
VLLM_TARGET_DEVICE=empty uv pip install --system --no-deps --no-build-isolation \
  -e "$VLLM_SRC"
python -c "import torch, torch_npu, vllm, vllm_ascend; from vllm.entrypoints.openai.utils import validate_json_request; print(f'vllm {vllm.__version__} + vllm-ascend 0.22.1rc1 layout OK; torch {torch.__version__}, torch_npu {torch_npu.__version__}')"

# -------------------------------------------------------
# 1f. tau2 profile: the tau2-bench fork (async orchestrator
# + user simulator) and its task data. TAU2_DATA_DIR must reach the trainer
# AND the rollout workers (the inline workflow loads tasks from the
# registry), so it is exported through GITHUB_ENV - the local scheduler's
# workers inherit the environment.
# -------------------------------------------------------
if [[ "$TAU2_PREP" == 1 ]]; then
  TAU2_DIR="$GITHUB_WORKSPACE/areal_tau2/tau2-bench"
  shallow_clone https://github.com/dhh1995/tau2-bench.git dhh/async-and-custom-completion "$TAU2_DIR"
  [[ -d "$TAU2_DIR/data" ]] || { echo "tau2-bench data/ not found in fork" >&2; exit 2; }
  pip_tuna "$TAU2_DIR"
  echo "TAU2_DATA_DIR=$TAU2_DIR/data" >> "$GITHUB_ENV"
  echo "tau2-bench fork installed; TAU2_DATA_DIR=$TAU2_DIR/data"
fi

# -------------------------------------------------------
# 2. Runtime Environment setup.
# -------------------------------------------------------
echo "PYTHONPATH=/areal-workspace/MindSpeed:/areal-workspace/Megatron-Bridge/src:${PYTHONPATH:-}" >> "$GITHUB_ENV"
echo "HCCL_IF_BASE_PORT=63000" >> "$GITHUB_ENV"
echo "HCCL_NPU_SOCKET_PORT_RANGE=62100-62350" >> "$GITHUB_ENV"
# Rank-0 weight broadcast (memory_efficient_load) makes rank 1 wait inside
# the HCCL collective while rank 0 slowly CPU-loads the full weights (~4min
# for 7B on CI disk); the default connect/exec timeouts (~2min) kill the
# link first (EI0006 "Getting socket times out"). Same values as the proven
# gsm8k_grpo_npu.yaml scheduling_spec env suite.
echo "HCCL_CONNECT_TIMEOUT=7200" >> "$GITHUB_ENV"
echo "HCCL_EXEC_TIMEOUT=14400" >> "$GITHUB_ENV"
echo "ACL_DEVICE_SYNC_TIMEOUT=14400" >> "$GITHUB_ENV"
echo "TASK_QUEUE_ENABLE=1" >> "$GITHUB_ENV"
echo "OMP_NUM_THREADS=1" >> "$GITHUB_ENV"
echo "WANDB_MODE=disabled" >> "$GITHUB_ENV"
# swanlab resolves the run mode from SWANLAB_MODE only: SwanLabRun.__init__
# does `self.__mode = get_mode()` (swanlab/env.py), which reads that env and
# defaults to "cloud"; the mode= kwarg AReaL passes to swanlab.init() is not
# threaded into the run. AReaL calls swanlab.init() unconditionally
# (areal/utils/stats_logger.py:91) with the config default mode="disabled", so
# without this env the run resolves to cloud -> FileUploadManager(mode="cloud")
# -> get_client() -> ValueError "client object is not initialized". Every
# trainer builds StatsLogger, so this applies to all profiles.
echo "SWANLAB_MODE=disabled" >> "$GITHUB_ENV"
echo "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True" >> "$GITHUB_ENV"
echo "USE_OPTIMIZED_MODEL=0" >> "$GITHUB_ENV"
echo "AREAL_ALLOW_DEFAULT_ADMIN_KEY=1" >> "$GITHUB_ENV"

# -------------------------------------------------------
# 3. Pre-download Model & Dataset (using image's native tools)
# -------------------------------------------------------
# No local fixtures and no shared-cache dependency (cache-seed is NOT used):
# the dataset is fetched online at train time (`load_dataset`, inside the data
# service), and the model is downloaded fresh on every run - HF first,
# ModelScope (魔搭) fallback - into the per-run workspace. HF_ENDPOINT points
# at huggingface.co (this repo's runner pool reaches it directly; hf-mirror's
# 308 redirect on namespace-less ids such as t5-base is not followed by
# huggingface_hub), exported through GITHUB_ENV so the run-example step (and
# any later load_dataset) inherits it.
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
echo "HF_ENDPOINT=${HF_ENDPOINT}" >> "$GITHUB_ENV"
export AREAL_MODEL_ID="$MODEL_ID"

# The resolved path is captured into the AREAL_MODEL_PATH shell variable (tee
# keeps the diagnostics streaming into the job log) and written to GITHUB_ENV
# for the run step (overlay args reference AREAL_MODEL_PATH), so it must be
# the real on-disk download dir.
AREAL_MODEL_PATH=$(python3 <<'PY' | tee /dev/stderr | sed -n 's/^AREAL_MODEL_PATH=//p' | tail -n 1
import os
import subprocess
import sys

model_id = os.environ["AREAL_MODEL_ID"]
workspace_dir = os.path.join(
    os.environ["GITHUB_WORKSPACE"], "areal_models", model_id.split("/")[-1]
)
os.makedirs(workspace_dir, exist_ok=True)


def from_hf():
    from huggingface_hub import snapshot_download

    return snapshot_download(model_id, local_dir=workspace_dir)


def from_modelscope():
    try:
        from modelscope import snapshot_download as ms_snapshot
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "modelscope==1.37.0"],
            check=True,
        )
        from modelscope import snapshot_download as ms_snapshot
    ms_snapshot(model_id, local_dir=workspace_dir)
    return workspace_dir


try:
    snapshot = from_hf()
    print(f"{model_id}: downloaded from HF -> {snapshot}", flush=True)
except Exception as e:
    print(
        f"HF download failed ({type(e).__name__}: {e}); trying ModelScope",
        flush=True,
    )
    snapshot = from_modelscope()
    print(f"{model_id}: downloaded from ModelScope -> {snapshot}", flush=True)

with open(os.environ["GITHUB_ENV"], "a") as fh:
    fh.write(f"AREAL_MODEL_PATH={snapshot}\n")
print(f"AREAL_MODEL_PATH={snapshot}", flush=True)
PY
)
[[ -n "${AREAL_MODEL_PATH}" ]] || {
  echo "model resolution printed no AREAL_MODEL_PATH line" >&2
  exit 2
}
