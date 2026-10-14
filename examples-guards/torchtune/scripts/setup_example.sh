#!/usr/bin/env bash
# Prepare the CI environment for one supported torchtune example.
# $1 is the manifest profile. Unknown profiles fail before any install.
# torchtune itself is installed from TARGET_ROOT (the release checkout
# under test), so the guarded tag is exactly the code that runs.
#
# Zero-source-patch policy: supported examples run WITHOUT modifying
# upstream source — anything that needs a source patch belongs in
# unsupported. The one distributed recipe in supported
# (full_finetune_distributed) needs no patch: backend resolves via
# get_distributed_backend("npu") → hccl, and the manifest's seed=42
# overlay skips _broadcast_tensor's CPU broadcast (training/seed.py
# only broadcasts when seed is None).
#
# Asset sourcing: Qwen/Qwen2.5-{0.5,1.5}B-Instruct are resolved through
# scripts/hub_cache.py — the shared HF hub cache is hit as-is, or the
# asset is downloaded (HuggingFace first, ModelScope as fallback) and
# published for the next run; the snapshot paths are injected as
# TT_MODEL_PATH / TT_TEACHER_PATH for the overlay args.
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

# Per-profile torch stack install (torch/torchvision versions are
# uniform across profiles; torch_npu version varies).
#
# $1 = torch_npu version (mandatory; caller picks the version that
#      matches the recipes it serves).
#
# Why torch + torchvision stay the same across profiles:
#   - torch 2.11.0+cpu: floor for `from torch.nn.functional import
#     ScalingType` that torchao 0.18 needs (main HEAD path). The +cpu
#     wheel comes from aliyun's mirror of pytorch.org/whl/cpu; the
#     cluster pip cache ships only CUDA torch wheels (Requires-Dist:
#     cuda-toolkit) which collide with constraints-npu.txt's
#     cuda-toolkit<0 — see torchtitan-cuda-torch-wheel-trap memory.
#     `--find-links` to the +cpu-only directory sidesteps it.
#   - torchvision 0.26.0+cpu: main HEAD unconditionally imports
#     torchvision at torchtune/data/_utils.py:12, triggered by
#     `from torchtune import datasets` (datasets/__init__.py:7 →
#     multimodal → _llava → data._messages → data → data._utils →
#     import torchvision). Per upstream's published compat table,
#     torch 2.11 ↔ torchvision 0.26; pinning ≤0.28 also dodges the
#     v0.29.0 stable ABI requirement (calls stable::permute, torch
#     2.14 only; torch_npu has no 2.14 release) — see
#     torchvision-v29-stable-abi memory. The +cpu wheel is the only
#     one in the aliyun pytorch-wheels/cpu find-links repo.
ensure_torch_stack() {
  local torch_npu_version="${1:?ensure_torch_stack requires torch_npu version as \$1}"
  echo "installing torch==2.11.0+cpu (aliyun pytorch-wheels/cpu find-links, deps from PIP_INDEX_URL)"
  # The 148MB torch+cpu aarch64 wheel is the long pole of setup — CI run
  # 35483424375 (2026-09-20) saw every leg print the "Downloading torch-
  # 2.11.0+cpu... (148.1 MB)" banner then hang for 28+ min before the
  # 30-min job timeout cancelled the job. Split download from install so
  # a slow link can be retried with curl's byte-range resume (-C -) and
  # bounded by --max-time, and so pip installs a local wheel instead of
  # re-resolving through the aliyun find-links (pure-python deps still
  # come from PIP_INDEX_URL). Same hardening as projects/xtuner
  # (CI run 35483448757).
  TORCH_WHEEL_DIR=/tmp/torchtune-wheels
  mkdir -p "$TORCH_WHEEL_DIR"
  TORCH_WHEEL="$TORCH_WHEEL_DIR/torch-2.11.0+cpu-cp312-cp312-manylinux_2_28_aarch64.whl"
  if [[ ! -f "$TORCH_WHEEL" ]]; then
    curl -fsSL --retry 5 --retry-delay 5 --retry-all-errors --max-time 1500 \
      -C - -o "$TORCH_WHEEL" \
      "https://mirrors.aliyun.com/pytorch-wheels/cpu/torch-2.11.0%2Bcpu-cp312-cp312-manylinux_2_28_aarch64.whl" \
      || python -m pip install --find-links https://mirrors.aliyun.com/pytorch-wheels/cpu torch==2.11.0
  fi
  if [[ -f "$TORCH_WHEEL" ]]; then
    python -m pip install --find-links "$TORCH_WHEEL_DIR" torch==2.11.0 || \
      python -m pip install --find-links https://mirrors.aliyun.com/pytorch-wheels/cpu torch==2.11.0
  fi
  echo "installing torchvision==0.26.0+cpu (matches torch 2.11 per upstream compat table)"
  python -m pip install \
    --find-links https://mirrors.aliyun.com/pytorch-wheels/cpu \
    torchvision==0.26.0
  echo "installing torch_npu==${torch_npu_version} (Huawei ascend index)"
  pip_ascend "torch_npu==${torch_npu_version}"
}

# Copy CI fixture data files into the target root so that example
# recipes can load them via a local path under $TARGET_ROOT/fixtures/
# (engine contract: overlay_args uses ${TARGET_ROOT}/fixtures/...;
# same decoupling from the workflows-checkout subtree as peft).
prepare_fixtures() {
  local src="${FIXTURE_DIR:?FIXTURE_DIR is required}"
  local dst="$TARGET_ROOT/fixtures"
  echo "preparing fixtures from $src to $dst"
  # Fail-soft: we don't want to hard-fail if the project only ships a
  # custom-task YAML and no JSON/JSONL — the eleuther_eval recipe, for
  # example, ships fixtures/eleuther_tasks/*.yaml but no top-level JSON.
  # The actual presence/validity of the file is checked downstream when
  # the recipe opens it; a missing file at setup is a config bug that
  # surfaces as a real stack trace, not a setup-script one.
  if ! compgen -G "$src/*" >/dev/null; then
    echo "FATAL: fixture dir $src is empty" >&2
    exit 1
  fi
  mkdir -p "$dst"
  # Mirror the entire tree: flat *.json / *.jsonl for dataset recipes,
  # plus *.yaml subdirs (e.g. fixtures/eleuther_tasks/) for lm_eval
  # custom-task definitions consumed by recipes/eleuther_eval.py via
  # cfg.include_path = ${TARGET_ROOT}/fixtures/eleuther_tasks.
  cp -r "$src"/. "$dst"/
  echo "copied $(find "$dst" -type f | wc -l) fixture file(s) to $dst"
}

resolve_hub_paths() {
  # $@ = env var names from scripts/hub_cache.py ASSETS. Each resolves to
  # a snapshot in the shared HF hub cache, downloaded on a miss, and is
  # appended to GITHUB_ENV for the manifest overlay_args.
  python "$HERE/hub_cache.py" "$@"
}

# Install torchtune + its declared deps (transformers / omegaconf /
# tokenizers / safetensors / tqdm / pyyaml). Shared by every profile —
# the version selection / dynamic torchao probe are torchtune-checkout-
# dependent and have nothing to do with the torch_npu version split
# above. Probe runs against the installed source (non-editable, see
# Quick-start-Ascend.md:164-167), so it reflects whatever ref the
# engine checked out (release tag OR main HEAD).
install_torchtune_pkg() {
  echo "installing torchtune from $TARGET_ROOT (non-editable, see Quick-start-Ascend.md:164-167)"
  python -m pip install "$TARGET_ROOT"
  python -m pip install "transformers==4.57.1" "omegaconf>=2.3,<3" \
    tokenizers safetensors tqdm pyyaml

  # Probe which NF4Tensor import path the torchtune checkout uses, then
  # install the matching torchao exact pin.
  local import_path torchao_pin
  import_path="$(python -c "
import importlib.util, pathlib
p = pathlib.Path('$TARGET_ROOT') / 'torchtune' / 'modules' / 'common_utils.py'
src = p.read_text() if p.exists() else ''
for line in src.splitlines():
    s = line.strip()
    if s.startswith('from torchao') and 'NF4Tensor' in s:
        print(s)
        break
else:
    print('NF4Tensor_NOT_IMPORTED')
")"
  echo "torchtune common_utils.py NF4Tensor import: $import_path"
  case "$import_path" in
    "from torchao.dtypes.nf4tensor import NF4Tensor")
      torchao_pin="torchao==0.13.0"
      ;;
    "from torchao.quantization import NF4Tensor")
      # 0.18.0 re-exposed NF4Tensor under torchao.quantization; main
      # HEAD depends on the exact 0.18 series (later 0.18.x keep the
      # symbol but pin to whatever the upstream test grid currently
      # passes — 0.18.0 is the first stable release with the re-add).
      torchao_pin="torchao==0.18.0"
      ;;
    "NF4Tensor_NOT_IMPORTED")
      # Newer torchtune may drop NF4Tensor entirely. Default to a
      # neutral recent torchao and let setup proceed; recipe failures
      # downstream will surface real reasons rather than setup noise.
      echo "WARN: common_utils.py does not import NF4Tensor; skipping torchao pin"
      torchao_pin=""
      ;;
    *)
      echo "FATAL: unexpected NF4Tensor import line: $import_path" >&2
      exit 1
      ;;
  esac
  if [ -n "$torchao_pin" ]; then
    echo "pinning $torchao_pin (matched import path)"
    python -m pip install "$torchao_pin"
  fi

  # importlib.metadata.version returns the real install tag for both
  # editable installs (where __version__ is empty string) and wheel
  # installs. torchtune.__version__ is "" by default in the source
  # tree; reading it directly would print a confusing blank.
  python -c "
import importlib.metadata as md
import torchao, omegaconf, transformers
import torchtune  # noqa: just to confirm the import chain
print('torchtune', md.version('torchtune'), '/ torchao', torchao.__version__, '/ transformers', transformers.__version__)
"
}

# Profile: single-device recipes. torch_npu 2.11.0 is verified to
# work for these on coder npu-3.
setup_torchtune_single() {
  ensure_torch_stack 2.11.0
  install_torchtune_pkg

  # lm_eval is only needed for the eleuther_eval recipe (not declared as
  # an upstream dep of torchtune). The recipe's __init__ checks
  # `version("lm-eval") < "0.4.5"` (recipes/eleuther_eval.py:446) using
  # STRING comparison (importlib.metadata.version returns a string, no
  # version coercion), not packaging.version. So "0.4.13" < "0.4.5" is
  # True ("0.4.1" prefix beats "0.4.5" lexicographically) and 0.4.10-0.4.49
  # all get rejected — only 0.4.5-0.4.9 and 0.4.50+ pass. Pin exact 0.4.5
  # (lowest acceptable) to keep the floor obvious; the import chain only
  # uses evaluator/models/tasks/utils which is stable across 0.4.x.
  python -m pip install "lm-eval==0.4.5"

  # modelscope: hub_cache.py's fallback when HuggingFace fails.
  python -m pip install "modelscope==1.37.0"

  # Resolve model snapshot paths through the shared HF hub cache:
  # student / SFT base is Qwen2.5-0.5B-Instruct, KD teacher is
  # Qwen2.5-1.5B-Instruct.
  resolve_hub_paths TT_MODEL_PATH TT_TEACHER_PATH
}

# Profile: full_finetune_distributed — supported 里唯一的 distributed
# recipe。免补丁路径：backend 走 get_distributed_backend("npu") → hccl
# （torch_npu 2.11.0 registry），seed 走 manifest overlay seed=42 绕过
# _broadcast_tensor 的 CPU broadcast。teacher（1.5B）只有 KD single_device
# 用（torchtune_single profile），这里不 resolve。
setup_torchtune_distributed() {
  ensure_torch_stack 2.11.0
  install_torchtune_pkg
  python -m pip install "modelscope==1.37.0"
  resolve_hub_paths TT_MODEL_PATH
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

# recipes/eleuther_eval.py:311 hard-codes `super().__init__(pretrained="gpt2", ...)`
# which triggers transformers.AutoConfig.from_pretrained("gpt2"). This runner
# pool reaches huggingface.co directly, so keep the endpoint explicit: the
# residual gpt2 config download and hub_cache.py's HuggingFace-first lookup
# both go straight to huggingface.co, never through hf-mirror.
export HF_ENDPOINT=https://huggingface.co

source /usr/local/Ascend/ascend-toolkit/set_env.sh

select_pip_index
python -m pip install -U pip setuptools wheel
prepare_fixtures

"setup_${PROFILE}"