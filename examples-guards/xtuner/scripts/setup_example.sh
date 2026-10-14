#!/usr/bin/env bash
# Prepare the CI environment for one supported xtuner example.
# $1 is the manifest profile. Unknown profiles fail before any install.
# xtuner itself is installed from TARGET_ROOT (the release checkout
# under test), so the guarded tag is exactly the code that runs.
#
# The upstream requirements/runtime.txt is deliberately NOT installed
# as-is: it pins bitsandbytes==0.45.0 (no aarch64 NPU wheel) and
# torchvision without version (PyPI pulls CUDA build that mismatches
# +cpu torch). We install xtuner 0.2.0 from the release tag first,
# then overlay the same fixed dep line the Quick-start-Ascend.md
# doc exercises end-to-end (2026-09-14 verified on coder npu-3:
#   xtuner v0.2.0 + transformers==4.48.0 + peft>=0.14.0 + datasets 3.x
#   + torch 2.11.0 / torch_npu 2.11.0 + torchvision 0.26.0+cpu).
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

ensure_torch_stack() {
  # torch 2.11.0 + torch_npu 2.11.0 — pinned by Quick-start-Ascend.md
  # (CANN 9.1.0 pairing verified on coder npu-3 2026-09-14). Reuse the
  # image stack when it already matches, otherwise install.
  if python -c "
import torch, torch_npu
raise SystemExit(
    0 if torch.__version__.startswith('2.11.0')
    and torch_npu.__version__.startswith('2.11.0') else 1)
" 2>/dev/null; then
    echo "reusing image torch stack ($(python -c 'import torch; print(torch.__version__)'))"
    return
  fi
  echo "installing torch==2.11.0 torch_npu==2.11.0"
  # The 148MB torch+cpu aarch64 wheel is the long pole of setup — CI run
  # 35483448757 (2026-09-20) saw each leg print the "Downloading torch-
  # 2.11.0+cpu... (148.1 MB)" banner then hang for 28+ min before the
  # 30-min job timeout kicked in. Split download from install so a slow
  # link can be retried with curl's byte range resume, and so pip
  # install runs against a local wheel (no pip resolver round-trip).
  XTUNER_WHEEL_DIR=/tmp/xtuner-wheels
  mkdir -p "$XTUNER_WHEEL_DIR"
  if [[ ! -f "$XTUNER_WHEEL_DIR/torch-2.11.0+cpu-cp312-cp312-manylinux_2_28_aarch64.whl" ]]; then
    local wheel_url="https://mirrors.aliyun.com/pytorch-wheels/cpu/torch-2.11.0%2Bcpu-cp312-cp312-manylinux_2_28_aarch64.whl"
    curl -fsSL --retry 5 --retry-delay 5 --retry-all-errors --max-time 1500 \
      -C - -o "$XTUNER_WHEEL_DIR/torch-2.11.0+cpu-cp312-cp312-manylinux_2_28_aarch64.whl" \
      "$wheel_url" \
      || pip_ascend -f https://mirrors.aliyun.com/pytorch-wheels/cpu torch==2.11.0
  fi
  if [[ -f "$XTUNER_WHEEL_DIR/torch-2.11.0+cpu-cp312-cp312-manylinux_2_28_aarch64.whl" ]]; then
    # --no-deps so pip doesn't try to also pull filelock/networkx/jinja2
    # from --no-index (the local dir only has torch). Those deps come
    # along via the runtime-deps pip install below, which goes back to
    # the aliyun / cluster indexes.
    pip_ascend --no-deps --no-index --find-links "$XTUNER_WHEEL_DIR" torch==2.11.0 || \
      pip_ascend -f https://mirrors.aliyun.com/pytorch-wheels/cpu torch==2.11.0
  fi
  pip_ascend torch_npu==2.11.0
}

setup_xtuner-llm() {
  # scikit-image pulls GUI opencv-python 5.x as a transitive dep, whose
  # cv2.abi3.so links libxcb.so.1 / libGL.so.1 — missing from the lean
  # CANN runner image. Install the system libs before pip so any post-
  # install `import cv2` (mmengine pulls cv2 via naive_visualization_hook)
  # doesn't crash with libxcb.so.1 not found. Confirmed by CI run
  # 35050200236 (2026-09-16); same fix is in projects/xtuner/docs/
  # Quick-start-Ascend.md (apt-get install libgl1 libglib2.0-0).
  if command -v apt-get >/dev/null 2>&1; then
    echo "installing system libs (libgl1 libglib2.0-0) for opencv-python 5.x"
    apt-get update -qq && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 2>&1 | tail -3
  else
    echo "skipping apt-get (no apt-get on PATH); cv2 import may fail"
  fi
  # xtuner from the guarded release checkout, plus the verified dep
  # line from Quick-start-Ascend.md (2026-09-14 verified, coder npu-3
  # end-to-end run):
  #   xtuner 0.2.0 + transformers 4.48.0 + peft>=0.14.0 + datasets 3.x
  #   + scikit-image (libgl1 system dep) + torchvision 0.26.0+cpu
  # PIP_CONSTRAINT keeps CUDA metapackages out (see constraints-npu.txt).
  # Install from the guarded release checkout (`-e` always re-links
  # site-packages/xtuner to TARGET_ROOT/xtuner, so this overrides any
  # xtuner already in the image from a previous run; the code under
  # test is exactly what the release tag ships).
  python -m pip install --no-deps -e "$TARGET_ROOT" 2>&1 | tail -3
  # runtime deps — verbatim from Quick-start-Ascend.md `xtuner-install-binary`
  # block (applies on top of xtuner==0.2.0 too; matches verified stack).
  # torch / torch_npu already installed by ensure_torch_stack; do not
  # re-list them here, or pip would redownload the 148 MB torch wheel.
  # Use PIP_CONSTRAINT (set above) to keep mmengine from pulling a
  # fresh torch==2.x range marker and re-downloading torch.
  python -m pip install -f https://mirrors.aliyun.com/pytorch-wheels/cpu \
      'mmengine==0.10.6' 'transformers==4.48.0' 'peft>=0.14.0' \
      'datasets>=3.2.0,<4.0.0' einops loguru openpyxl 'scikit-image' scipy \
      SentencePiece tiktoken transformers_stream_generator cyclopts \
      'opencv-python-headless<=4.12.0.88' 'torchvision==0.26.0+cpu' \
      timm pyarrow pydantic tensorboard xxhash imageio 'py-libnuma' GitPython
  python -c "
import torch, torch_npu
assert torch.__version__.startswith('2.11.0'), f'torch drifted to {torch.__version__}'
assert torch_npu.__version__.startswith('2.11.0'), f'torch_npu drifted to {torch_npu.__version__}'
# xtuner does NOT expose __version__ on the top-level module (unlike
# peft/accelerate); it lives in xtuner/version.py. Confirmed by CI run
# 35043584383 (xtuner.__version__ raised AttributeError; using
# `from xtuner.version import __version__` is the correct probe).
from xtuner.version import __version__ as xtuner_ver
import xtuner, transformers, datasets, accelerate, peft
print('xtuner', xtuner_ver, '/ transformers', transformers.__version__,
      '/ datasets', datasets.__version__, '/ torch', torch.__version__)
print('peft', peft.__version__)
"

  # Pre-download the small LLM used by the supported HF trainer
  # example from ModelScope (China-reachable; HF download is gated).
  # The snapshot path is exported as LLM_MODEL_PATH for overlay_args
  # to consume.
  # Pinned: modelscope>=1.38 splits the hub code into modelscope-hub,
  # and the fresh 1.40.1 wheel's loose ">=0.4.2" floor breaks import
  # when the mirror lags on hub 0.4.3. 1.37.0 is the last pre-split
  # line, verified by the sibling quick-start docs.
  python -m pip install "modelscope==1.37.0"
  python - <<'PY'
import os
# Non-TTY CI logs: throttle tqdm refreshes instead of disabling.
os.environ.setdefault("TQDM_MININTERVAL", "15")
from modelscope import snapshot_download

MODEL_CACHE = os.environ.get("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope"))
local = snapshot_download("Qwen/Qwen2.5-0.5B", cache_dir=MODEL_CACHE)
with open(os.environ["GITHUB_ENV"], "a") as fh:
    fh.write(f"LLM_MODEL_PATH={local}\n")
print("LLM_MODEL_PATH=", local)
PY

  # Launcher for `examples/demo_data/*/config.py` (mmengine cfg recipes
  # with `with read_base(): from .map_fn import ...`). run_example.sh
  # resolves `manifest.entry.exec` against TARGET_ROOT then does
  # `python "$LAUNCH_PATH" "$@"` — but `python config.py` blows up on
  # the relative import (ImportError: attempted relative import with no
  # known parent package). This wrapper invokes
  # `python -m xtuner.tools.train` instead, which Config.fromfile()
  # happily reads the cfg tree. cwd = TARGET_ROOT so the cfg path is
  # resolved relative to the release checkout (same as the supported
  # train_hf.py entry, which is launched bare).
  # xtuner v0.2.0 release tree has no scripts/ dir (top-level layout is
  # .github/docs/examples/requirements/xtuner only); without mkdir the
  # heredoc hits "No such file or directory". Reproduced by CI run
  # 35201067734 (2026-09-17) on every xtuner-examples config + train_hf
  # job, because setup_xtuner-llm is shared.
  mkdir -p "$TARGET_ROOT/scripts"
  cat > "$TARGET_ROOT/scripts/xtuner_train_demo.sh" <<'SH'
#!/usr/bin/env bash
# Launcher used by supported demo_data/*/config.py entries. Manifest
# invokes it as `xtuner_train_demo.sh <cfg-path> <overlay args>`.
# run_example.sh's *.sh branch sets cwd to $TARGET_ROOT/scripts/ before
# invoking this launcher (matches torchtitan's `tune run` launcher
# pattern), but xtuner.tools.train reads the cfg via `osp.isfile(...)`
# which is cwd-relative. The manifest cfg path is relative to the
# release checkout (e.g. `examples/demo_data/multi_turn_2/config.py`),
# so this launcher must cd back to $TARGET_ROOT before invoking the
# train module. TARGET_ROOT is exported by run_example.sh.
cd "${TARGET_ROOT:?TARGET_ROOT is required}" || exit 1
exec python -m xtuner.tools.train "$@"
SH
  chmod +x "$TARGET_ROOT/scripts/xtuner_train_demo.sh"
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

source /usr/local/Ascend/ascend-toolkit/set_env.sh

select_pip_index
python -m pip install -U pip setuptools wheel
ensure_torch_stack

"setup_${PROFILE}"
