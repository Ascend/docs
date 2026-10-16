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

SUPPORTED_PROFILES="areal-vlm-grpo areal-vlm-mt-grpo areal-vlm-sft areal-clevr-sft areal-tir-grpo areal-scaffold-grpo areal-agents-grpo areal-math-grpo areal-math-sft areal-math-aime areal-math-boba areal-countdown-grpo areal-hermes-grpo areal-openclaw-grpo areal-camel-grpo areal-online-rollout areal-hitl-demo areal-tau2-grpo areal-agent-service areal-align"
AIME_PREP=0
BOBA_PREP=0
COUNTDOWN_PREP=0
HHRLHF_PREP=0
TORL_PREP=0
SCAFFOLD_PREP=0
CLEVR_SUBSET_PREP=0
HERMES_PREP=0
OPENCLAW_PREP=0
CAMEL_PREP=0
HITL_PREP=0
TAU2_PREP=0
AGENT_SERVICE_PREP=0
case "$PROFILE" in
  areal-vlm-grpo) MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct" ;;
  areal-vlm-mt-grpo) MODEL_ID="Qwen/Qwen3-VL-2B-Instruct" ;;
  areal-vlm-sft) MODEL_ID="Qwen/Qwen3-VL-2B-Instruct" ;;
  # clevr_count_70k_sft.py: see section 10 (local subset; the full 70k train
  # split blows the data worker when the SFT loader materializes pixel_values).
  areal-clevr-sft) MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct"; CLEVR_SUBSET_PREP=1 ;;
  # tir/train_tir.py: see section 8 (fixture staging; the loader's own
  # GitHub download is unreachable from the CI job containers).
  areal-tir-grpo) MODEL_ID="Qwen/Qwen2.5-Math-1.5B"; TORL_PREP=1 ;;
  # gsm8k_rlvr_scaffolding.py: 1.5B model + section 9 (model alias symlink).
  areal-scaffold-grpo) MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"; SCAFFOLD_PREP=1 ;;
  areal-agents-grpo) MODEL_ID="Qwen/Qwen2-1.5B-Instruct" ;;
  # gsm8k_rl.py and gsm8k_eval.py share the same model.
  areal-math-grpo) MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct" ;;
  areal-math-sft) MODEL_ID="Qwen/Qwen3-1.7B" ;;
  areal-math-aime) MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"; AIME_PREP=1 ;;
  areal-math-boba) MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"; BOBA_PREP=1 ;;
  areal-countdown-grpo) MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"; COUNTDOWN_PREP=1 ;;
  # hermes/train.py: online RL; agent runtime package installed in
  # section 1c, orchestration in run_example.sh's hermes branch.
  areal-hermes-grpo) MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"; HERMES_PREP=1 ;;
  # openclaw/train.py: online RL through the OpenAI-compatible proxy gateway;
  # no agent runtime package to install (demo_lifecycle.py is pure HTTP).
  # No source patch (no-patch policy) - demo_lifecycle.py runs as shipped.
  # Reuses hermes' model.
  areal-openclaw-grpo) MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"; OPENCLAW_PREP=1 ;;
  # camel/train.py: camel-ai ChatAgent GRPO. Needs the undeclared camel-ai
  # package (section 1d) and a `python -m` launch (run_example.sh camel branch).
  areal-camel-grpo) MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"; CAMEL_PREP=1 ;;
  # online_rollout.py: rollout-only online demo; no extra runtime to install
  # (drive is run_example.sh's online_rollout branch).
  areal-online-rollout) MODEL_ID="Qwen/Qwen3-1.7B" ;;
  # human_in_the_loop_demo.py: zeroclaw-driven HITL demo; section 1e installs
  # the CLI and plants its home config skeleton.
  areal-hitl-demo) MODEL_ID="Qwen/Qwen3-0.6B"; HITL_PREP=1 ;;
  # tau2/train.py: multi-turn customer-service RL (real user simulator; the
  # user LLM is hosted locally by run_example.sh's tau2 branch). Section 1f
  # installs the tau2-bench fork + data.
  areal-tau2-grpo) MODEL_ID="Qwen/Qwen3-1.7B"; TAU2_PREP=1 ;;
  # agent_service/run_agent_service.py: Claude Agent SDK micro-service; the
  # Claude Code CLI it spawns is installed in section 1g; run_example.sh's
  # agent-service branch points its API traffic at a local sidecar gateway.
  areal-agent-service) MODEL_ID="Qwen/Qwen3-0.6B"; AGENT_SERVICE_PREP=1 ;;
  # hhrlhf_dpo.py: DPO pipeline is model-agnostic; upstream yaml's 7B swapped
  # for 1.5B (7B fp32 init OOMs the 32GB cards in the pool, 15GB per run).
  areal-align) MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"; HHRLHF_PREP=1 ;;
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
# Shared helpers: mirror pip installs, shallow clones, tarball
# fetches - one place for the mirror URLs and CI timeout budget.
# -------------------------------------------------------
pip_tuna() {
  uv pip install --system --index-url https://pypi.tuna.tsinghua.edu.cn/simple "$@"
}

pip_aliyun() {
  uv pip install --system --index-url https://mirrors.aliyun.com/pypi/simple "$@"
}

shallow_clone() {
  git clone --depth 1 --branch "$2" "$1" "$3"
}

fetch_tarball() {
  curl -fsSL --connect-timeout 30 --max-time 600 "$1" -o "$2" \
    || { echo "failed to download $1" >&2; exit 2; }
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
# 1c. hermes profile: the agent runtime package. The model is the
# standard Qwen2.5-1.5B-Instruct (downloaded by section 3); hermes-agent
# itself is model-agnostic - it drives whatever OpenAI-compatible endpoint
# the rollout engine serves. The Agent Service imports
# examples.hermes.hermes.HermesAgent from the target checkout.
# Index: tsinghua (not aliyun) - aliyun's PEP 658 .metadata sidecar for
# hermes-agent's python-dotenv dep 404s intermittently; tuna mirrors the
# metadata reliably (same mirror as transformers-quick-start.yml).
# -------------------------------------------------------
if [[ "$HERMES_PREP" == 1 ]]; then
  pip_tuna hermes-agent==0.19.0
fi

# -------------------------------------------------------
# 1d. camel profile: the camel-ai agent runtime.
# examples/camel/train.py imports camel.agents.ChatAgent, and
# areal.experimental.camel.openai_model imports camel.logger / camel.messages /
# camel.models.base_model / camel.types / camel.utils; camel-ai is NOT an
# AReaL-declared dependency, so it is installed here. camel-ai goes in with
# --no-deps so the job's NPU torch/vllm stack is never touched (camel is
# API-based); its lighter runtime deps come in separately (none pull torch).
# -------------------------------------------------------
if [[ "$CAMEL_PREP" == 1 ]]; then
  pip_tuna --no-deps camel-ai
  pip_tuna tiktoken docstring-parser colorama python-dotenv jsonschema
fi

# -------------------------------------------------------
# 1e. hitl-demo profile: the zeroclaw chat client + its home
# config skeleton. human_in_the_loop_demo.py drives its own traffic through
# the zeroclaw CLI (`zeroclaw agent -m ... --session-state-file ...`), so the
# binary must be on PATH. The demo's FIRST action on the config is
# shutil.copy2(~/.zeroclaw/config.toml, .demo_bak) -> FileNotFoundError if the
# file is missing, and its regex patch silently skips a missing
# default_provider line - so plant a minimal skeleton in the runner home dir
# (the demo backs it up and restores it on exit; AReaL source is untouched).
# If zeroclaw.com turns out unreachable from the CN runners, switch to the
# fork's GitHub Releases aarch64-linux tarball.
# -------------------------------------------------------
if [[ "$HITL_PREP" == 1 ]]; then
  # PINNED to v0.6.0: the last release line whose CLI matches the demo's
  # hardcoded invocation `zeroclaw agent -m ... --session-state-file ...`.
  # v0.6.0's agent subcommand has NO --agent flag and its Providers list
  # documents `custom:<URL>` - the inline OpenAI-compatible provider the
  # demo's config patcher writes. v0.7+ introduced the agent-alias system
  # and made --agent REQUIRED ("Required - there is no default agent"),
  # so the current official build (zeroclaw.com / releases/latest, v0.8.x)
  # rejects the demo's invocation with a usage error (observed rc=2).
  # zeroclaw.com's installer has no version flag, so fetch the pinned
  # release tarball directly - github release downloads are reachable from
  # the CN runners (zeroclaw.com's own installer fetches from there).
  # /usr/local/bin is on the default PATH, so no PATH plumbing is needed.
  ZC_VERSION=v0.6.0
  ZC_ASSET="zeroclaw-aarch64-unknown-linux-gnu.tar.gz"
  ZC_URL="https://github.com/zeroclaw-labs/zeroclaw/releases/download/${ZC_VERSION}/${ZC_ASSET}"
  ZC_TMP="$(mktemp -d)"
  # a 404 here usually means the aarch64 asset is missing in $ZC_VERSION
  fetch_tarball "$ZC_URL" "$ZC_TMP/$ZC_ASSET"
  tar -xzf "$ZC_TMP/$ZC_ASSET" -C "$ZC_TMP"
  [[ -x "$ZC_TMP/zeroclaw" ]] || { echo "$ZC_ASSET did not contain a zeroclaw binary" >&2; exit 2; }
  install -m 0755 "$ZC_TMP/zeroclaw" /usr/local/bin/zeroclaw
  rm -rf "$ZC_TMP"
  zeroclaw --version || { echo "zeroclaw failed to run" >&2; exit 2; }
  # Same flat keys the demo's patcher regexes target (plus workspace_dir,
  # mirroring what v0.6.0's own installer writes); the demo rewrites
  # default_provider/api_key to the live gateway on start.
  mkdir -p "$HOME/.zeroclaw"
  cat > "$HOME/.zeroclaw/config.toml" <<'TOML'
default_provider = "custom:http://localhost:1"
default_model = "default"
api_key = "placeholder"
workspace_dir = "/tmp/zeroclaw-workspace"
TOML
  echo "planted minimal ~/.zeroclaw/config.toml skeleton"
fi

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
# 1g. agent-service profile: the Claude Agent SDK stack.
# examples/agent_service's ClaudeAgent shells out through claude-agent-sdk,
# whose ClaudeSDKClient spawns the Claude Code CLI (an npm package) - so
# besides the pip package we need Node >= 18 and the CLI itself. The NPU
# image ships no node, so install a pinned Node from the npmmirror CDN
# (CN-reachable, unlike nodejs.org) and the CLI from the npmmirror
# registry. The API traffic is pointed at the LOCAL gateway by
# run_example.sh's agent-service branch (ANTHROPIC_BASE_URL) - no real
# Anthropic key is ever used.
# -------------------------------------------------------
if [[ "$AGENT_SERVICE_PREP" == 1 ]]; then
  pip_tuna claude-agent-sdk
  AS_NEED_NODE=1
  if command -v node >/dev/null 2>&1; then
    AS_NODE_MAJOR="$(node --version | sed 's/^v\([0-9]*\).*/\1/')"
    if (( AS_NODE_MAJOR >= 18 )); then AS_NEED_NODE=0; fi
  fi
  if [[ "$AS_NEED_NODE" == 1 ]]; then
    AS_NODE_VERSION=v22.14.0
    AS_NODE_DIR="/usr/local/node-${AS_NODE_VERSION}"
    fetch_tarball \
      "https://registry.npmmirror.com/-/binary/node/${AS_NODE_VERSION}/node-${AS_NODE_VERSION}-linux-arm64.tar.gz" \
      /tmp/as-node.tar.gz
    mkdir -p "$AS_NODE_DIR"
    tar -xzf /tmp/as-node.tar.gz -C "$AS_NODE_DIR" --strip-components=1
    export PATH="$AS_NODE_DIR/bin:$PATH"
    echo "PATH=$PATH" >> "$GITHUB_ENV"
  fi
  node --version || { echo "node unavailable for the claude CLI" >&2; exit 2; }
  npm install -g @anthropic-ai/claude-code \
    --registry=https://registry.npmmirror.com \
    || { echo "claude code CLI install failed" >&2; exit 2; }
  claude --version || { echo "claude CLI failed to run" >&2; exit 2; }
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
# service), and the model is downloaded fresh on every run - HF mirror first,
# ModelScope (魔搭) fallback - into the per-run workspace. HF_ENDPOINT is
# exported through GITHUB_ENV so the run-example step (and any later
# load_dataset) inherits it.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
echo "HF_ENDPOINT=${HF_ENDPOINT}" >> "$GITHUB_ENV"
export AREAL_MODEL_ID="$MODEL_ID"

# The resolved path is captured into the AREAL_MODEL_PATH shell variable (tee
# keeps the diagnostics streaming into the job log) and written to GITHUB_ENV
# for the run step. Later sections of this script - countdown's tokenizer
# (section 6) and the scaffolding model alias (section 9) - consume it, so it
# must be the real on-disk download dir.
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

# -------------------------------------------------------
# 4. AIME dataset + derived config (areal-math-aime).
# -------------------------------------------------------
# areal/dataset/aime.py loads <path>/aime_train.parquet and
# <path>/aime_test.parquet from a LOCAL directory (upstream ships no aime
# config; the loader takes no HF repo id). Convert the public MathArena
# parquets (same flow as examples/distillation/mopd/README.md), then derive
# the GRPO config from gsm8k_grpo_npu.yaml: point the datasets at the local
# dir and set rollout.agent to null (aime_rl.py runs the standard
# RLVRWorkflow, not an agent workflow). The generated config is exposed to
# overlay args via AREAL_AIME_CONFIG.
if [[ "$AIME_PREP" == 1 ]]; then
python3 <<'PY'
import os

import pandas as pd
import yaml
from huggingface_hub import hf_hub_download

workspace = os.environ["GITHUB_WORKSPACE"]
target_root = os.environ["TARGET_ROOT"]
aime_dir = os.path.join(workspace, "areal_data", "aime")
os.makedirs(aime_dir, exist_ok=True)

# hf_hub_download honors HF_ENDPOINT (mirror) set in section 3.
for repo, dst in [
    ("MathArena/aime_2025", "aime_train.parquet"),
    ("MathArena/aime_2026", "aime_test.parquet"),
]:
    src = hf_hub_download(
        repo_id=repo, repo_type="dataset",
        filename="data/train-00000-of-00001.parquet",
    )
    df = pd.read_parquet(src).rename(columns={"problem": "question"})
    df = df[["question", "answer"]]
    df.to_parquet(os.path.join(aime_dir, dst))
    print(f"prepared {dst}: {len(df)} rows", flush=True)

base = os.path.join(target_root, "examples", "math", "gsm8k_grpo_npu.yaml")
with open(base, encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh)
cfg["experiment_name"] = "aime-grpo"
cfg["train_dataset"]["path"] = aime_dir
cfg["valid_dataset"]["path"] = aime_dir
config_path = os.path.join(aime_dir, "aime_grpo_npu.yaml")
with open(config_path, "w", encoding="utf-8") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)

with open(os.environ["GITHUB_ENV"], "a") as fh:
    fh.write(f"AREAL_AIME_CONFIG={config_path}\n")
print(f"generated {config_path}", flush=True)
PY
fi

# -------------------------------------------------------
# 5. BoBa dataset (areal-math-boba).
# -------------------------------------------------------
# boba_grpo.py loads train_dataset.path as a LOCAL jsonl file
# (load_dataset("json", data_files=path)); the upstream yaml ships the HF
# repo id, which would fail with FileNotFoundError. Pre-download the public
# AReaL-boba-106k.jsonl and expose its local path via AREAL_BOBA_DATA.
if [[ "$BOBA_PREP" == 1 ]]; then
python3 <<'PY'
import os
import shutil

from huggingface_hub import hf_hub_download

src = hf_hub_download(
    repo_id="inclusionAI/AReaL-boba-Data", repo_type="dataset",
    filename="AReaL-boba-106k.jsonl",
)
dst = os.path.join(
    os.environ["GITHUB_WORKSPACE"], "areal_data", "boba", "AReaL-boba-106k.jsonl"
)
os.makedirs(os.path.dirname(dst), exist_ok=True)
shutil.copyfile(src, dst)
with open(os.environ["GITHUB_ENV"], "a") as fh:
    fh.write(f"AREAL_BOBA_DATA={dst}\n")
print(f"downloaded boba dataset to {dst}", flush=True)
PY
fi

# -------------------------------------------------------
# 6. Countdown dataset (areal-countdown-grpo).
# -------------------------------------------------------
# countdown.py writes ./data/countdown/qwen/*.jsonl relative to CWD; run it
# from TARGET_ROOT so the generated files land where train_config.yaml's
# default train_dataset.path=data/countdown/qwen/train_e.jsonl expects them
# (run_example.sh also cds to TARGET_ROOT). The default generation size is
# 500k samples; 32/8 is plenty for the 1-step smoke. The tokenizer comes
# from the model predownloaded in section 3, so generation is offline.
if [[ "$COUNTDOWN_PREP" == 1 ]]; then
  MODEL_DIR="$AREAL_MODEL_PATH"
  mkdir -p "$TARGET_ROOT/data/countdown/qwen"
  (cd "$TARGET_ROOT" && python3 examples/countdown/countdown.py \
    --num_samples 32 --eval_size 8 --tokenizer_path "$MODEL_DIR")
fi

# -------------------------------------------------------
# 7. HH-RLHF dataset (areal-align: hhrlhf_dpo.py / hhrlhf_rw.py).
# -------------------------------------------------------
# Anthropic/hh-rlhf is public (NOT gated), but its jsonl.gz files live in
# per-subset subdirectories (harmless-base/, helpful-base/, ...), so the
# upstream loader's load_dataset(path, split=...) without a config name
# fails with a multi-config error. Work around it boba-style: download the
# harmless-base subset online (via HF_ENDPOINT mirror) into a flat local
# dir; load_dataset(<dir>, split="train"/"test") then infers splits from
# the file names. Exposed to overlay args via AREAL_HHRLHF_DATA.
# NOTE: the dir is deliberately named "hh-rlhf" 鈥?areal's dataset dispatch
# matches the literal substring "hh-rlhf" in the path; a dir named
# "hhrlhf" falls through to the load_from_disk fallback and errors out.
if [[ "$HHRLHF_PREP" == 1 ]]; then
python3 <<'PY'
import os
import shutil

from huggingface_hub import hf_hub_download

dst_dir = os.path.join(os.environ["GITHUB_WORKSPACE"], "areal_data", "hh-rlhf")
os.makedirs(dst_dir, exist_ok=True)
for filename in ("train.jsonl.gz", "test.jsonl.gz"):
    src = hf_hub_download(
        repo_id="Anthropic/hh-rlhf", repo_type="dataset",
        filename=f"harmless-base/{filename}",
    )
    shutil.copyfile(src, os.path.join(dst_dir, filename))
with open(os.environ["GITHUB_ENV"], "a") as fh:
    fh.write(f"AREAL_HHRLHF_DATA={dst_dir}\n")
print(f"prepared hh-rlhf harmless-base at {dst_dir}", flush=True)
PY
fi

# -------------------------------------------------------
# 8. ToRL dataset (areal-tir-grpo): fixture staging.
# -------------------------------------------------------
# The torl_data loader downloads its parquets from github.com at load time
# (areal/dataset/torl_data.py), but the CI job containers cannot reach
# github.com (mainland network, no proxy; the runner agent pulls code on the
# host side). No HF/ModelScope mirror of ToRL exists (ModelScope 404;
# upstream ships GitHub only), so the two tiny parquets are vendored as
# fixtures - the documented last resort. prepare_torl_data() skips its
# download once /tmp/areal/torl_data/_SUCCESS exists, so staging the files
# plus the flag here makes the loader fully offline. The data service worker
# runs in the same container, so /tmp is shared with the run step.
# Fixture pin (GAIR-NLP/ToRL @ main): train.parquet blob 57bd7d5d (6035244
# bytes), test.parquet blob e8ca20fd (51798 bytes).
if [[ "$TORL_PREP" == 1 ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  TORL_FIXTURE="$SCRIPT_DIR/../fixtures"
  for f in train.parquet test.parquet; do
    [[ -s "$TORL_FIXTURE/$f" ]] || {
      echo "missing fixture: $TORL_FIXTURE/$f (see section 8 comment for source)" >&2
      exit 2
    }
  done
  mkdir -p /tmp/areal/torl_data
  cp "$TORL_FIXTURE/train.parquet" "$TORL_FIXTURE/test.parquet" /tmp/areal/torl_data/
  : > /tmp/areal/torl_data/_SUCCESS
  echo "staged torl fixtures at /tmp/areal/torl_data (loader download skipped)"
fi

# -------------------------------------------------------
# 9. Scaffolding model alias (areal-scaffold-grpo).
# -------------------------------------------------------
# examples/scaffolding hardcodes model="default" in its OpenAI-compatible
# worker (workflow.py:102, a TRT-LLM convention), while AReaL's vLLM server
# serves under the model path - requests 404 and every rollout comes back
# empty, which then trips the "total loss weight must be positive" assert.
# vLLMConfig has no served_model_name passthrough (strict dataclass, unknown
# keys are rejected at the structured merge), so serve the model under the
# literal name "default" instead: vLLM uses the raw --model string as the
# served model name, so a relative "default" path resolving to the model dir
# makes the name match. The server subprocess inherits the run CWD
# (TARGET_ROOT), hence the symlink location.
if [[ "$SCAFFOLD_PREP" == 1 ]]; then
  MODEL_DIR="$AREAL_MODEL_PATH"
  [[ -d "$MODEL_DIR" ]] || { echo "model dir missing: $MODEL_DIR" >&2; exit 2; }
  ln -sfn "$MODEL_DIR" "$TARGET_ROOT/default"
  echo "linked $TARGET_ROOT/default -> $MODEL_DIR (scaffolding model alias)"
fi

# -------------------------------------------------------
# 10. CLEVR subset (areal-clevr-sft).
# -------------------------------------------------------
# The clevr SFT loader materializes per-sample pixel_values across the whole
# split (clevr_count_70k.py:104-127), unlike the RL loader which only keeps
# JPEG bytes, so the 70k train split blows the data worker (host OOM / tens
# of GB of arrow cache). The example hardcodes split="train" and
# get_custom_dataset ignores train_dataset.split when the script passes a
# non-None split (dataset/__init__.py:322,328), and the loader takes no extra
# kwargs - so subset via path instead: train_dataset.path points at a dir
# whose name contains "clevr_count_70k" (dispatch substring) holding
# train.parquet (load_dataset infers the split from the file name). The
# absolute slice train[:200] downloads only the first shard (~0.5GB of the
# 10.4GB), and the parquet keeps the HF feature metadata so images stay PIL.
if [[ "$CLEVR_SUBSET_PREP" == 1 ]]; then
python3 <<'PY'
import os

from datasets import load_dataset

dst = os.path.join(os.environ["GITHUB_WORKSPACE"], "areal_data", "clevr_count_70k")
os.makedirs(dst, exist_ok=True)
ds = load_dataset("BUAADreamer/clevr_count_70k", split="train[:200]")
ds.to_parquet(os.path.join(dst, "train.parquet"))
with open(os.environ["GITHUB_ENV"], "a") as fh:
    fh.write(f"AREAL_CLEVR_SUBSET_DIR={dst}\n")
print(f"prepared clevr subset ({len(ds)} rows) at {dst}", flush=True)
PY
fi
