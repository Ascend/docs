#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"
# A profile selects the smallest dependency set needed by one example family;
# it is not a transformers feature flag. Keep unknown profiles failing before
# any package is installed so a manifest typo cannot run a partial setup.
# 只保留 lm：上游 v5.20.0 起删除 examples/pytorch/（PR #49357），generation /
# glue / small-training / seq2seq / vision / speech 六个 profile 连同它们需要的
# 模型快照与 fixture 一起退役。若这些用例回归，从本文件的 git 历史里取回。
case "$PROFILE" in
  lm)
    # tensorboard 不是可选项：run_greedy.py 把 report_to="tensorboard" 写死在
    # TrainingArguments 里且没有 CLI 开关，Trainer 每次都会构造
    # TensorBoardCallback，缺包时直接 RuntimeError 退出。
    DEPS=(accelerate datasets tensorboard)
    ;;
  *)
    echo "unknown profile: $PROFILE (supported: lm)" >&2
    exit 1
    ;;
esac

: "${TARGET_ROOT:?TARGET_ROOT is required}"
: "${GITHUB_ENV:?GITHUB_ENV is required}"
# Redirect the pip cache onto the shared persistent runner cache the pool
# mounts at ~/.cache/huggingface (docs/examples-guard-engine.md). Model
# snapshots land in the same shared HF cache (HF_HOME/hub) by default.
SHARED_CACHE_ROOT="${HF_HOME:-${HOME}/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SHARED_CACHE_ROOT}/pip}"
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export PIP_TRUSTED_HOST="repo.huaweicloud.com"
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python -m pip install -U pip
# The Ascend image normally contains a compatible torch/torch_npu pair. Reuse
# it when possible because these wheels are large; install only on a missing
# or unusable image stack.
if python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.device_count())"; then
  echo "reusing image torch stack"
else
  python -m pip install --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi \
    torch==2.9.0 torch_npu==2.9.0.post2
fi
# Install the current target checkout. Its dependencies (regex, tokenizers,
# huggingface-hub, safetensors, ...) are resolved by pip from the index; this is
# safe for the NPU torch stack because transformers does not depend on torch_npu
# and its torch requirement is already satisfied by the image build.
python -m pip install -e "$TARGET_ROOT"
# Pin datasets to a single release. The rebuilt 9.1.0 image no longer
# preinstalls datasets, so an unpinned resolve pulls 5.x. 3.6.0 is the release
# the deepspeed guard pins, and it keeps the two properties this example needs:
# `load_dataset` still resolves a Hub dataset from the parquet files in the
# dataset repo (wikitext ships no loading script), and the resolve avoids the
# pathological pip backtracking surface that once walked an unpinned install
# down to the datasets 0.0.9 sdist.
python -m pip install "${DEPS[@]}" "datasets==3.6.0"

# run_greedy.py 默认取 meta-llama/Llama-3.2-1B（gated，Runner 拉不到），且默认走
# AutoModelForCausalLM.from_config 从零训练，只需要 config.json 与 tokenizer
# 文件；tiny 快照两者都带。引擎不设 HF_ENDPOINT、也不禁用 Xet，下载直连
# huggingface.co。run_greedy.py 是唯一 supported 条目，下载失败直接让 job 失败，
# 不像以前那样降级成 warning。
if TINYLLAMA=$(python - <<'PY'
from huggingface_hub import snapshot_download

print(snapshot_download("hf-internal-testing/tiny-random-LlamaForCausalLM"))
PY
); then
  echo "TINYLLAMA_PATH=$TINYLLAMA" >> "$GITHUB_ENV"
else
  echo "error: tiny causal LM download failed; run_greedy.py is the only supported entry" >&2
  exit 1
fi
