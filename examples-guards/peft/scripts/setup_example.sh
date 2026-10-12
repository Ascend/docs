#!/usr/bin/env bash
# Prepare the CI environment for one supported peft example.
# $1 is the manifest profile. Unknown profiles fail before any install.
# peft itself is installed from TARGET_ROOT (the release checkout under
# test), so the guarded tag is exactly the code that runs.
#
# The upstream examples/sft/requirements.txt is deliberately NOT
# installed: it pins everything to git main (transformers/peft/trl@main
# + flash-attn + unsloth + bitsandbytes), which conflicts with testing
# a release tag and contains CUDA-only packages. We install the
# checkout plus the minimal SFTTrainer stack instead.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"

CLUSTER_PIP_HOST=cache-service.nginx-pypi-cache.svc.cluster.local
export CLUSTER_PIP_INDEX="http://${CLUSTER_PIP_HOST}/pypi/simple"
ASCEND_PIP_INDEX=https://repo.huaweicloud.com/ascend/repos/pypi
ALIYUN_PIP_INDEX=https://mirrors.aliyun.com/pypi/simple/

pip_ascend() {
  python -m pip install --extra-index-url "$ASCEND_PIP_INDEX" "$@"
}

select_pip_index() {
  # Runners live in mainland China: prefer the cluster pip cache, fall
  # back to the Aliyun mirror. The ascend index stays available via
  # PIP_EXTRA_INDEX_URL (set by the engine) for torch_npu wheels.
  if python -c "
import os
import urllib.error
import urllib.request
try:
    urllib.request.urlopen(os.environ['CLUSTER_PIP_INDEX'], timeout=3)
except urllib.error.HTTPError:
    pass
" 2>/dev/null; then
    export PIP_INDEX_URL="$CLUSTER_PIP_INDEX"
    export PIP_TRUSTED_HOST="$CLUSTER_PIP_HOST"
  else
    export PIP_INDEX_URL="$ALIYUN_PIP_INDEX"
    unset PIP_TRUSTED_HOST
  fi
  echo "pip index: $PIP_INDEX_URL"
}

torch_stack_for_profile() {
  # Two coexisting torch stacks (2026-09-21), selected per profile:
  # - default (peft / peft_dreambooth): torch 2.12.0 + torch_npu 2.12.0,
  #   the post-2026-09-20 verified line for single-card entries. The
  #   2.12 upgrade was originally made for the sparse-COO tuner shira
  #   (torch_npu 2.9 crashes at torch.sparse_coo_tensor construction
  #   plus dense+=sparse); shira was pulled back to unsupported on
  #   2026-09-21 (torch_npu 2.12's sparse backward still loses gradient
  #   values under bf16, run 35575049729 — see examples_manifest.yaml),
  #   but the 2.12 line stays as default for the remaining entries.
  # - peft_29 / peft_ds (the 4 multi-card sft entries): torch 2.9.0 +
  #   torch_npu 2.9.0.post2, the pre-2026-09-20 verified stack. torch
  #   2.12's c10d broadcast() computes sm90_or_more via
  #   torch.cuda.get_device_capability(tensor.device)[0], and torch_npu
  #   patches Tensor.is_cuda = Tensor.is_npu while its
  #   get_device_capability shim returns None -> None[0] TypeError.
  #   That kills FSDP checkpoint save (dist_cp gather_object ->
  #   broadcast_object_list) and DeepSpeed ZeRO zero-init param
  #   broadcast (run 35502546409). Upstream torch_npu bug; pin these
  #   profiles back to 2.9 until fixed.
  case "$1" in
    peft_29|peft_ds) printf '2.9.0 2.9.0.post2' ;;
    *)               printf '2.12.0 2.12.0' ;;
  esac
}

ensure_torch_stack() {
  local torch_ver npu_ver
  read -r torch_ver npu_ver <<< "$(torch_stack_for_profile "$PROFILE")"
  if python -c "
import torch, torch_npu
raise SystemExit(
    0 if torch.__version__.startswith('$torch_ver')
    and torch_npu.__version__.startswith('$npu_ver') else 1)
"; then
    echo "reusing image torch stack ($(python -c 'import torch; print(torch.__version__)'))"
    return
  fi
  if [[ "$torch_ver" == "2.9.0" ]]; then
    echo "installing torch==$torch_ver torch_npu==$npu_ver (pre-2.12 verified stack)"
    pip_ascend "torch==${torch_ver}" "torch_npu==${npu_ver}"
    return
  fi
  # torch 2.12.0 + torch_npu 2.12.0 + CANN 9.1.0 (upgraded 2026-09-20
  # from 2.9.0 to unblock the sparse-COO tuners shira, see above).
  #
  # torch==2.12.0 is NOT installed via `pip_ascend`: aliyun (and the
  # cluster pip cache, both PyPI mirrors) only host the CUDA torch
  # wheel, whose METADATA declares `Requires-Dist: cuda-toolkit`, which
  # conflicts with constraints-npu.txt `cuda-toolkit<0`. The CPU
  # variant is `torch==2.12.0+cpu` (PEP 440 local label) published ONLY
  # at https://download.pytorch.org/whl/cpu/ - so fetch it by direct
  # URL. We compute the cp tag at runtime because the manifest mixes
  # py3.10 (30 entries) and py3.12 (15 entries) images.
  echo "installing torch==2.12.0+cpu (direct URL) + torch_npu==2.12.0"
  CP_ABI=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
  python -m pip install --no-deps \
    "https://download.pytorch.org/whl/cpu/torch-2.12.0%2Bcpu-${CP_ABI}-${CP_ABI}-manylinux_2_28_aarch64.whl"
  # torch's pure-Python deps (filelock / typing-extensions / sympy /
  # networkx / jinja2 / fsspec) from aliyun so `import torch` succeeds
  # (the +cpu wheel does not pull them; none declare cuda-toolkit).
  python -m pip install -i "$ALIYUN_PIP_INDEX" \
    'filelock' 'typing-extensions>=4.10.0' 'setuptools<82' \
    'sympy>=1.13.3' 'networkx>=2.5.1' 'jinja2' 'fsspec>=0.8.5'
  pip_ascend torch_npu==2.12.0
}

install_cpu_torchvision() {
  # torchvision matching torch 2.12 is 0.27.0; like torch it ships as a
  # +cpu direct-download wheel only (PyPI linux wheels link libcudart.so
  # and are cuda-toolkit-gated). Only the adamss image example needs it.
  # pillow is torchvision's image-codec dep (numpy already comes from
  # scikit-learn/evaluate); install it separately so its deps resolve.
  local cp_abi
  cp_abi=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
  python -m pip install --no-deps \
    "https://download.pytorch.org/whl/cpu/torchvision-0.27.0%2Bcpu-${cp_abi}-${cp_abi}-manylinux_2_28_aarch64.whl"
  python -m pip install pillow
}

resolve_hub_paths() {
  # $@ = env var names from scripts/hub_cache.py ASSETS. Each resolves to
  # a snapshot in the shared HF hub cache, downloaded on a miss, and is
  # appended to GITHUB_ENV for manifest overlay_args.
  python "$HERE/hub_cache.py" "$@"
}

# Copy CI fixture data into the target root so that example scripts can
# load them via a local path under $TARGET_ROOT/fixtures/ (same
# decoupling from the workflows-checkout subtree as trl).
#
# Both flavors are supported, all top-level entries in $FIXTURE_DIR:
#   *.jsonl    → single-file fixtures (e.g. ci_sft_8.jsonl)
#   */         → directory fixtures (e.g. ci_alpaca_10/ with train.jsonl,
#                ci_corda_8/ with train.jsonl + test.jsonl)
# Examples that need a directory data_path (corda, glora, hira, olora,
# waveft) fail with FileNotFoundError if directory fixtures are not
# copied here — root cause of CI failures in run 35222723526.
prepare_fixtures() {
  local src="${FIXTURE_DIR:?FIXTURE_DIR is required}"
  local dst="$TARGET_ROOT/fixtures"
  echo "preparing fixtures from $src to $dst"
  if [[ ! -d "$src" ]]; then
    echo "FATAL: fixture source dir not found: $src" >&2
    exit 1
  fi
  shopt -s nullglob
  # Top-level *.jsonl files
  local n_files=0
  for f in "$src"/*.jsonl; do
    mkdir -p "$dst"
    cp "$f" "$dst/"
    n_files=$((n_files + 1))
  done
  # Top-level subdirectories (e.g. ci_alpaca_10/, ci_corda_8/)
  local n_dirs=0
  for d in "$src"/*/; do
    [[ -d "$d" ]] || continue
    mkdir -p "$dst"
    cp -r "$d" "$dst/"
    n_dirs=$((n_dirs + 1))
  done
  shopt -u nullglob
  if (( n_files == 0 && n_dirs == 0 )); then
    echo "FATAL: no fixture files (*.jsonl) or subdirs found in $src" >&2
    exit 1
  fi
  echo "copied $n_files fixture file(s) and $n_dirs fixture dir(s) to $dst"
}

setup_peft() {
  # peft from the guarded release checkout, plus the verified dependency
  # line (2026-09-15, coder npu-1 端到端验证结论):
  #   transformers 4.57.1 + datasets>=4.7.0,<6 + hub<1.0 + trl 1.12.0
  # - transformers 4.57.1: 5.x 移除 send_example_telemetry 等旧 API
  # - datasets>=4.7.0,<6: trl 1.12+ 在 wheel metadata 声明 datasets>=4.7.0
  #   (pyproject.toml 自 v1.0.0 起 commit ac5421b4 引入，datasets<4 会
  #   ResolutionImpossible)，<6 留出口避开未来 6.x breaking
  # - hub<1.0: hub 1.x 拒绝 imdb 等无命名空间数据集
  # - trl 1.12.0: trl ≥ 1.12 默认 chunked_nll 走 _patch_chunked_ce_lm_head
  #   (sft_trainer.py:233)。该 patch 第 383 行 (1.13 在 386) 走
  #   inspect.signature(original_forward.__func__)，假设 forward 是普通
  #   method；但 miss/mica 用 device_map="auto" 加载，accelerate
  #   .add_hook_to_module 把 forward 包成 functools.partial，没 __func__
  #   → AttributeError（纯 PyTorch + accelerate，硬件无关，CUDA 同问题）。
  #   overlay 在 examples_manifest.yaml 的 miss/mica 例里显式 --loss_type nll
  #   跳过 chunked patch 走标准 cross-entropy。
  #   注：sft_trainer.py:1331 isinstance(BaseTunerLayer) guard 在这两例不
  #   会触发（target_modules 不含 lm_head），guard 放过后 patch 内部才崩。
  # - scikit-learn: adamss 的 ASA 回调（peft.tuners.adamss）硬性 import
  #   sklearn；evaluate.load("glue") 的 metric 模块同样要 sklearn.metrics。
  #   coder 验证机里碰巧预装，CANN 裸镜像没有（run 35045940066 实测缺失）。
  # PIP_CONSTRAINT keeps CUDA metapackages out.
  echo "installing peft from $TARGET_ROOT"
  python -m pip install -e "$TARGET_ROOT"
  # modelscope: hub_cache.py's fallback when HuggingFace fails; 1.40.1
  # fails to import next to hub<1.0.
  python -m pip install "transformers==4.57.1" "datasets>=4.7.0,<6" \
    "huggingface_hub<1.0" "trl==1.12.0" evaluate scikit-learn \
    "modelscope==1.37.0"
  # torchvision only pairs with the 2.12 stack (adamss image example
  # imports it); the 2.9 profiles are sft-only and don't import
  # torchvision (its 0.27 wheel links the torch 2.12 ABI).
  local _torch_ver _npu_ver
  read -r _torch_ver _npu_ver <<< "$(torch_stack_for_profile "$PROFILE")"
  if [[ "$_torch_ver" == "2.12.0" ]]; then
    install_cpu_torchvision
  fi
  # t5-base seq2seq FSDP 例（peft_lora_seq2seq_accelerate_fsdp.py）零 CLI，
  # 数据路径硬编码 cwd 相对 temp/data/FinancialPhraseBank-v1.0/ 下两个
  # jsonl——从仓内 fixture 放置（run 阶段 cwd=$TARGET_ROOT 直接命中）。
  mkdir -p "$TARGET_ROOT/temp/data/FinancialPhraseBank-v1.0"
  cp "$TARGET_ROOT/fixtures/financial_phrasebank/financial_phrase_bank_train.jsonl" \
     "$TARGET_ROOT/temp/data/FinancialPhraseBank-v1.0/"
  cp "$TARGET_ROOT/fixtures/financial_phrasebank/financial_phrase_bank_val.jsonl" \
     "$TARGET_ROOT/temp/data/FinancialPhraseBank-v1.0/"
  python -c "import peft, trl, transformers, datasets, accelerate; print('peft', peft.__version__, '/ trl', trl.__version__, '/ transformers', transformers.__version__)"

  # Local snapshot paths for overlay_args: cache hit as is, a miss is
  # downloaded into the shared cache once. Hub ids the examples hardcode
  # (mt0-small, dinov2-base, glue, t5-base, opt-350m, imdb ...) are
  # fetched by from_pretrained / load_dataset into the same cache.
  resolve_hub_paths SFT_MODEL_PATH ROBERTA_BASE_PATH BERT_BASE_UNCASED_PATH
}

setup_peft_dreambooth() {
  # SD dreambooth 例（lora/oft/deft/hra/stable_diffusion ×5）：SFT 栈之外
  # 还要 diffusers + tensorboard。run 35303810367 实测缺 diffusers 直接
  # import 崩（train_dreambooth.py:14）→ 初版修复"diffusers tensorboard"
  # 不 pin，以为 pip 不会动已装版本——run 35316518546（2026-09-18）5 条
  # dreambooth 全挂打脸：最新 diffusers 0.40 要求 huggingface-hub>=1.23，
  # pip 把 hub 从 0.36.2 升到 1.31+（对已装的 transformers 只给 warning
  # 不回退），transformers 4.57.1 的 dependency_versions_check 硬校验
  # hub<1.0 → `import transformers` 直接 ImportError。
  # 修复（coder npu-5 2026-09-18 实测 5 条 exit 0）：hub<1.0 +
  # diffusers==0.39.0 双 pin（0.39 与 hub<1.0 兼容，0.40 起不兼容）。
  # hra 例外：v0.21.0 脚本 :319 在 cwd/data/dreambooth 不存在时无条件
  # git clone github.com/google/dreambooth（runner 网络不通 + 纯死代码，
  # 该路径只用于 clone 自身）；预先 mkdir 空目录即可绕过，无需 patch。
  setup_peft
  echo "installing dreambooth stack (diffusers==0.39.0 + tensorboard + wandb, hub<1.0)"
  # wandb：boft_dreambooth 走 --report_to wandb（脚本 `import wandb` 硬性，
  # wandb_init 只在 wandb 分支定义）；sitecustomize 的 wandb neutralizer
  # 把 wandb.init 强制 disabled，无需 API key。
  python -m pip install "huggingface_hub<1.0" "diffusers==0.39.0" tensorboard wandb
  python -c "import diffusers, tensorboard, wandb; print('diffusers', diffusers.__version__)"
  mkdir -p "$TARGET_ROOT/data/dreambooth"

  # SD v1.5 只有本 profile 需要，只取配置与 safetensors；其它例不解析它。
  resolve_hub_paths SD_MODEL_PATH
}

setup_peft_29() {
  # Multi-card fsdp sft entries (run_peft_fsdp.sh /
  # run_peft_qlora_fsdp.sh): identical deps to setup_peft, only the
  # torch stack differs (2.9.0 pair, see torch_stack_for_profile).
  setup_peft
}

setup_peft_fp4() {
  # fp4_finetuning/finetune_fp4_opt_bnb_peft.py：唯一走 bnb 的条目。
  # bnb 0.50.2 走默认 CPU 后端在 NPU 上跑 4-bit NF4（无需 NPU 专用
  # kernel）；run_example.sh 对该目录跳过 transfer_to_npu（见
  # SKIP_TRANSFER_TO_NPU）。bnb **不能**进全局 setup_peft：diffusers
  # 0.39 的 quantizers/auto.py 硬 import bnb，而 dreambooth 条目挂
  # transfer_to_npu 时 bnb 的 cuda backend 崩 torch._C.
  # _cuda_getCurrentRawStream——装了 bnb 会把 6 条 dreambooth 全带崩。
  setup_peft
  python -m pip install --index-url "$ALIYUN_PIP_INDEX" bitsandbytes
}

setup_peft_ds() {
  # deepspeed 多卡 sft 例（run_peft_deepspeed.sh /
  # run_peft_qlora_deepspeed_stage3.sh）：setup_peft 之上装 deepspeed。
  # torch 栈固定 2.9.0 对（torch_stack_for_profile）：torch 2.12 的
  # c10d broadcast sm90 检查 + torch_npu get_device_capability 返回
  # None 会在 ZeRO zero-init 的 dist.broadcast(param.data) 直接崩
  # （run 35502546409）。
  # 0.19.7 的 npu accelerator 自动识别 Ascend + HCCL，coder npu-5
  # 2026-09-18 实跑 ZeRO-3 2 卡 exit 0（loss 4.64）。DS_BUILD_OPS=0
  # 跳过 CPU op 内核编译：ZeRO-3 bf16 训练路径不需要编译 op，且
  # 编译耗时 + 裸镜像缺编译链，CI 不划算。
  setup_peft
  echo "installing deepspeed for ZeRO-3 sft entries"
  DS_BUILD_OPS=0 python -m pip install deepspeed==0.19.7
  python -c "import deepspeed; from deepspeed.accelerator import get_accelerator; print('deepspeed', deepspeed.__version__, 'accelerator', get_accelerator()._name)"
}

supported_profiles() {
  declare -F | awk '/^declare -f setup_/ { sub(/^declare -f setup_/, ""); print }' | paste -sd' ' -
}

if ! declare -F "setup_${PROFILE}" >/dev/null 2>&1; then
  echo "unknown profile: ${PROFILE} (supported: $(supported_profiles))" >&2
  exit 1
fi

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
GITHUB_WORKSPACE="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
GITHUB_ENV="${GITHUB_ENV:?GITHUB_ENV is required}"

HERE=$(cd "$(dirname "$0")" && pwd)
export PIP_CONSTRAINT="$(cd "$HERE/.." && pwd)/constraints-npu.txt"
# This runner pool reaches huggingface.co directly. The engine's
# hf-mirror default 308-redirects namespace-less ids such as roberta-base
# to huggingface.co, a redirect huggingface_hub does not follow.
export HF_ENDPOINT=https://huggingface.co

source /usr/local/Ascend/ascend-toolkit/set_env.sh

select_pip_index
python -m pip install -U pip setuptools wheel
ensure_torch_stack
prepare_fixtures

"setup_${PROFILE}"
