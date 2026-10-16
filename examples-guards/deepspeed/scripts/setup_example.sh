#!/usr/bin/env bash
# Prepare the CI environment for one supported example.
# $1 is the manifest profile. Unknown profiles fail before any install.
# DeepSpeed source is installed from the main-repo checkout (TARGET_ROOT in
# split mode; DEEPSPEED_SOURCE_ROOT overrides). Examples run from the
# DeepSpeedExamples checkout (EXAMPLES_ROOT).
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"
export PIP_CONSTRAINT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/constraints-npu.txt"

ASCEND_PIP_INDEX=https://repo.huaweicloud.com/ascend/repos/pypi
FALLBACK_PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
CLUSTER_PIP_HOST=cache-service.nginx-pypi-cache.svc.cluster.local
export CLUSTER_PIP_INDEX="http://${CLUSTER_PIP_HOST}/pypi/simple"

pip_ascend() {
  python -m pip install --extra-index-url "$ASCEND_PIP_INDEX" "$@"
}

select_pip_index() {
  if python -c "
import urllib.error
import urllib.request
try:
    urllib.request.urlopen('${CLUSTER_PIP_INDEX}', timeout=3)
except urllib.error.HTTPError:
    pass
" 2>/dev/null; then
    export PIP_INDEX_URL="$CLUSTER_PIP_INDEX"
    export PIP_TRUSTED_HOST="$CLUSTER_PIP_HOST"
  else
    export PIP_INDEX_URL="$FALLBACK_PIP_INDEX"
    unset PIP_TRUSTED_HOST
  fi
  echo "pip index: $PIP_INDEX_URL"
}

ensure_torch_stack() {
  if python -c "
import torch, torch_npu
print('found torch', torch.__version__, 'torch_npu', torch_npu.__version__)
raise SystemExit(0 if torch.__version__.startswith('2.9.0') and torch_npu.__version__.startswith('2.9.0') else 1)
"; then
    echo "reusing image torch stack"
    return
  fi
  echo "installing torch==2.9.0 torch_npu==2.9.0.post2"
  pip_ascend torch==2.9.0 torch_npu==2.9.0.post2
}

# Install DeepSpeed from the main-repo checkout (TARGET_ROOT).
install_deepspeed_source() {
  local src="${DEEPSPEED_SOURCE_ROOT:-$TARGET_ROOT}"
  echo "installing DeepSpeed from source at $src"
  python -m pip install -e "$src"
  python -c "
import deepspeed
print('DeepSpeed version:', deepspeed.__version__)
"
  ds_report 2>&1 | grep -i 'npu' || {
    echo 'WARNING: ds_report did not list npu accelerator'
  }
  echo "installing MPI runtime for deepspeed.initialize distributed discovery"
  apt-get update && apt-get install -y libopenmpi-dev numactl
  python -m pip install mpi4py
}

# Download models from ModelScope and expose their local paths to the run step
# via GITHUB_ENV (same pattern as examples-guards/trl). overlay_args reference these
# variables so examples receive concrete local directories.
ms_download_models() {
  # modelscope>=1.38 splits hub code into modelscope-hub; pin the last pre-split
  # release because the runner mirror may only expose an older hub for the latest wheel.
  python -m pip install -q "modelscope==1.37.0" safetensors
  TQDM_MININTERVAL="${TQDM_MININTERVAL:-15}" \
    MODELSCOPE_DOWNLOAD_PARALLELS=1 MODELSCOPE_ENABLE_DEFAULT_HASH_VALIDATION=true \
    python - "$@" <<'PY'
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from modelscope import snapshot_download
from safetensors import safe_open


def validate_snapshot(local):
    root = Path(local)
    if not (root / 'config.json').is_file():
        raise RuntimeError(f'model config missing: {root}')
    json.loads((root / 'config.json').read_text())
    shards = sorted(root.glob('*.safetensors'))
    index = root / 'model.safetensors.index.json'
    if index.is_file():
        expected = set(json.loads(index.read_text())['weight_map'].values())
        if not expected.issubset({shard.name for shard in shards}):
            raise RuntimeError(f'model safetensors shards missing: {root}')
    for shard in shards:
        with safe_open(str(shard), framework='np') as weights:
            if not list(weights.keys()):
                raise RuntimeError(f'empty model safetensors: {shard}')
    if not shards and not list(root.glob('pytorch_model*.bin')):
        raise RuntimeError(f'no PyTorch model weights: {root}')
    print(f'validated model assets: {root}, {len(shards)} safetensors shard(s)', flush=True)


# A shared snapshot can change after validation while another job downloads
# into it. Keep every model in this setup's private cache through the run step.
MODEL_CACHE = tempfile.mkdtemp(prefix='deepspeed-modelscope-', dir='/tmp')
print('private ModelScope cache:', MODEL_CACHE, flush=True)
for pair in sys.argv[1:]:
    env_name, model_id = pair.split("=", 1)
    cache = MODEL_CACHE
    ignored = ['*.h5', '*.msgpack', '*.onnx', '*.ot', '*.ckpt', '*.tflite']
    # This mirror contains safetensors; don't fetch its redundant .bin weights.
    if model_id == 'AI-ModelScope/bert-base-cased':
        ignored.append('*.bin')
    for attempt in range(3):
        try:
            local = snapshot_download(model_id, cache_dir=cache,
                                      ignore_file_pattern=ignored, max_workers=2,
                                      enable_file_lock=True)
            validate_snapshot(local)
            break
        except Exception as exc:
            print(f'ModelScope {model_id} attempt {attempt + 1}/3 failed: {exc}', flush=True)
            if attempt == 2:
                raise
            # A fresh job-owned cache avoids trusting a corrupt cached shard.
            # Keep the old files untouched; never purge the runner-wide cache.
            cache = tempfile.mkdtemp(prefix='deepspeed-modelscope-retry-', dir='/tmp')
            time.sleep(2 * (attempt + 1))
    with open(os.environ["GITHUB_ENV"], "a") as fh:
        fh.write(f"{env_name}={local}\n")
PY
  # GITHUB_ENV is applied in the next step; expose only requested model names
  # in this setup process so fixture checks can use the downloaded tokenizer.
  local pair env_name local_path
  for pair in "$@"; do
    env_name="${pair%%=*}"
    [[ "$env_name" =~ ^[A-Z_][A-Z0-9_]*$ ]] || {
      echo "invalid model environment name: $env_name" >&2
      return 1
    }
    local_path="$(awk -v key="$env_name" '
      index($0, key "=") == 1 { value = substr($0, length(key) + 2) }
      END { print value }
    ' "$GITHUB_ENV")"
    [[ -d "$local_path" ]] || {
      echo "downloaded model directory missing for $env_name: $local_path" >&2
      return 1
    }
    export "$env_name=$local_path"
  done
}

# Redirect the renamed wikitext dataset at interpreter startup without editing
# the DeepSpeedExamples checkout. The shim is scoped to this CI job through
# GITHUB_ENV and leaves every other datasets.load_dataset call unchanged.
install_hello_dataset_shim() {
  local shim_dir="$GITHUB_WORKSPACE/.ci/deepspeed-sitecustomize"
  mkdir -p "$shim_dir"
  cat > "$shim_dir/sitecustomize.py" <<'PY'
import datasets as _datasets

_original_load_dataset = _datasets.load_dataset


def _patched_load_dataset(path, *args, **kwargs):
    if path == "wikitext":
        path = "Salesforce/wikitext"
    return _original_load_dataset(path, *args, **kwargs)


_datasets.load_dataset = _patched_load_dataset
PY
  echo "PYTHONPATH=$shim_dir${PYTHONPATH:+:$PYTHONPATH}" >> "$GITHUB_ENV"
  echo "installed wikitext runtime redirect in $shim_dir"
}

# Copy a fixture into the DeepSpeed-Chat data/ dir that local/jsonfile reads.
# $1 = fixture name under $FIXTURE_DIR.
plant_chat_fixture() {
  local fixture="$1"
  local chat_data="$EXAMPLES_ROOT/applications/DeepSpeed-Chat/data"
  mkdir -p "$chat_data"
  cp "$FIXTURE_DIR/$fixture" "$chat_data/train.json"
  cp "$FIXTURE_DIR/$fixture" "$chat_data/eval.json"
  echo "planted chat fixture $fixture -> $chat_data/{train,eval}.json"
}

# DeepSpeed-Chat's setup.py uses find_packages(include=['dschat']), while the
# dschat root is a PEP 420 namespace package without __init__.py. pip therefore
# installs distribution metadata but does not make the source package
# importable when the launcher changes into a training/step* directory. Expose
# the upstream source root explicitly without editing the examples checkout.
install_chat_source_path() {
  local chat_root="$EXAMPLES_ROOT/applications/DeepSpeed-Chat"
  case ":${PYTHONPATH:-}:" in
    *":$chat_root:"*) ;;
    *) export PYTHONPATH="$chat_root${PYTHONPATH:+:$PYTHONPATH}" ;;
  esac
  echo "PYTHONPATH=$PYTHONPATH" >> "$GITHUB_ENV"
  python - <<'PY'
import dschat

paths = list(dschat.__path__)
if not paths:
    raise SystemExit("dschat namespace package has no source path")
print("runtime dschat namespace:", paths)
PY
}

setup_deepspeed() {
  install_deepspeed_source
  echo "installing dependencies from the HelloDeepSpeed and CIFAR requirements baselines"
  PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple" \
    python -m pip install "tokenizers>=0.22.0,<0.23" "transformers<5" datasets \
      fire loguru "sh==1.14.2" tqdm pytz tensorboard \
      "torchvision==0.24.0" "pillow>=7.1.0" matplotlib
  # CIFAR profiles reuse these packages but never load wikitext.
  if [[ "$PROFILE" == deepspeed ]]; then
    install_hello_dataset_shim
  fi
}

# Pre-stage CIFAR-10 from a pinned ModelScope dataset revision. Both the
# single-card and MoE examples use torchvision with root=./data and
# download=True; a complete local tree makes torchvision skip its unreachable
# Toronto download URL. Verify both the archive SHA-256 and torchvision's
# official per-file MD5 list.
plant_cifar10() {
  local data_dir="$EXAMPLES_ROOT/training/cifar/data"
  CIFAR10_DATA_DIR="$data_dir" python - <<'PY'
import hashlib
import os
from pathlib import Path
import time
import urllib.request
import zipfile

from torchvision.datasets import CIFAR10

URL = (
    "https://modelscope.cn/api/v1/datasets/studyhard1/"
    "cifar10-dataset/repo?Revision="
    "9231e736fd8d53f7158165a07d801429b4414993&"
    "FilePath=cifar-10-batches-py.zip"
)
EXPECTED_SHA256 = "4f287e6733e987d5c1ab2af557413cf0f5bc78f293765d87dc5633788246e5d7"
data_dir = Path(os.environ["CIFAR10_DATA_DIR"]).resolve()
archive = data_dir / "cifar-10-batches-py.zip"
partial = archive.with_suffix(".zip.part")
data_dir.mkdir(parents=True, exist_ok=True)


def complete() -> bool:
    dataset = CIFAR10.__new__(CIFAR10)
    dataset.root = str(data_dir)
    return dataset._check_integrity()


if complete():
    print("CIFAR-10 fixture already passes torchvision integrity checks:", data_dir)
    raise SystemExit(0)

for attempt in range(1, 4):
    digest = hashlib.sha256()
    try:
        request = urllib.request.Request(URL, headers={"User-Agent": "cosdt-ci/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != EXPECTED_SHA256:
            raise RuntimeError(
                f"CIFAR-10 archive sha256 mismatch: expected {EXPECTED_SHA256}, got {actual}"
            )
        partial.replace(archive)
        break
    except Exception:
        partial.unlink(missing_ok=True)
        if attempt == 3:
            raise
        print(f"CIFAR-10 download attempt {attempt}/3 failed; retrying", flush=True)
        time.sleep(2 * attempt)

with zipfile.ZipFile(archive) as bundle:
    for member in bundle.infolist():
        destination = (data_dir / member.filename).resolve()
        if destination != data_dir and data_dir not in destination.parents:
            raise RuntimeError(f"unsafe CIFAR-10 archive member: {member.filename}")
    bundle.extractall(data_dir)
archive.unlink()

if not complete():
    raise SystemExit("extracted CIFAR-10 data failed torchvision integrity checks")
print("planted verified CIFAR-10 fixture in", data_dir)
PY
}

setup_ds_cifar() {
  setup_deepspeed
  plant_cifar10
}

# DS-Chat reloads the tokenizer from config._name_or_path even when its model
# argument is local. Keep that lookup local in a job-owned asset view, without
# changing the upstream code or the shared ModelScope snapshot.
prepare_chat_model_view() {
  python - <<'PY'
import json
import os
from pathlib import Path
import tempfile
from transformers import AutoTokenizer

snapshot = Path(os.environ['OPT_125M_PATH']).resolve()
views = Path(os.environ['GITHUB_WORKSPACE']) / '.ci' / 'deepspeed-chat-models'
views.mkdir(parents=True, exist_ok=True)
view = Path(tempfile.mkdtemp(prefix='opt-125m-', dir=views))
for asset in snapshot.iterdir():
    if asset.name != 'config.json':
        (view / asset.name).symlink_to(asset, target_is_directory=asset.is_dir())
config = json.loads((snapshot / 'config.json').read_text())
config['_name_or_path'] = str(view)
(view / 'config.json').write_text(json.dumps(config))
AutoTokenizer.from_pretrained(str(view), local_files_only=True)
with open(os.environ['GITHUB_ENV'], 'a') as env:
    env.write(f'OPT_125M_PATH={view}\n')
print('DS-Chat offline model/tokenizer view:', view)
PY
  local model_path
  model_path="$(awk '/^OPT_125M_PATH=/ { value = substr($0, 15) } END { print value }' "$GITHUB_ENV")"
  [[ -d "$model_path" ]] || { echo 'DS-Chat local model view missing' >&2; return 1; }
  export OPT_125M_PATH="$model_path"
}

# Shared DeepSpeed-Chat setup: DS source + transformers + opt-125m + fixture.
setup_ds_chat() {
  local fixture="$1"
  install_deepspeed_source
  # Install the declared dependencies explicitly so pip cannot replace the NPU
  # torch stack or the source DeepSpeed under test. The upstream editable
  # package is intentionally not used: its find_packages() call produces an
  # empty package for the PEP 420 dschat namespace.
  python -m pip install "transformers>=4.31.0,<5,!=4.33.2" \
    "datasets>=2.8.0" "accelerate>=0.15.0" "sentencepiece>=0.1.97" \
    "protobuf==3.20.3" tensorboard
  install_chat_source_path
  ms_download_models "OPT_125M_PATH=facebook/opt-125m"
  prepare_chat_model_view
  plant_chat_fixture "$fixture"
}

setup_ds_chat_sft()  { setup_ds_chat ci_sft_8.json; }
setup_ds_chat_rw()   { setup_ds_chat ci_rw_8.json; }
setup_ds_chat_dpo()  { setup_ds_chat ci_dpo_8.json; }
setup_ds_chat_rlhf() { setup_ds_chat ci_rlhf_8.json; }

# Evaluation entries use the same upstream Chat baseline with protected runtime
# dependencies. Leave the already-validated training profiles unchanged.
setup_ds_chat_eval_dependencies() {
  install_deepspeed_source
  install_example_dependencies "transformers>=4.31.0,<5,!=4.33.2" \
    "datasets>=2.8.0" "accelerate>=0.15.0" "sentencepiece>=0.1.97" \
    "protobuf==3.20.3" tensorboard
  install_chat_source_path
  ms_download_models "OPT_125M_PATH=facebook/opt-125m"
  prepare_chat_model_view
  python - <<'PY'
import importlib.util
import os
from pathlib import Path
import torch_npu
from transformers import AutoTokenizer

root = Path(os.environ["EXAMPLES_ROOT"]) / "applications/DeepSpeed-Chat/training"
for step, filename in (
    ("step1_supervised_finetuning", "prompt_eval.py"),
    ("step2_reward_model_finetuning", "rw_eval.py"),
):
    spec = importlib.util.spec_from_file_location(f"ds_ci_{filename[:-3]}", root / step / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
tokenizer = AutoTokenizer.from_pretrained(os.environ["OPT_125M_PATH"])
print("Chat evaluation local tokenizer:", type(tokenizer).__name__)
PY
}

setup_ds_chat_prompt_eval() {
  setup_ds_chat_eval_dependencies
  # The runner trains a genuine small SFT checkpoint before prompt comparison.
  plant_chat_fixture ci_sft_8.json
}

setup_ds_chat_reward_eval() {
  setup_ds_chat_eval_dependencies
  # rw_eval's default create_critic_model() constructs a fresh value head; this
  # entry is an honest reward-forward smoke, not trained checkpoint restoration.
}

setup_ds_infer() {
  install_deepspeed_source
  python -m pip install "transformers<5" accelerate
  ms_download_models "OPT_125M_PATH=facebook/opt-125m"
}

setup_ds_autotp_equivalence() {
  install_deepspeed_source
  # Qwen3 support is present in current 4.x transformers; keep the upper bound
  # below the next major release to avoid unreviewed API changes.
  python -m pip install "transformers>=4.51.0,<5" safetensors
  ms_download_models "QWEN3_06B_PATH=Qwen/Qwen3-0.6B"
}

# Install only each example's non-runtime requirements. Constraints also stop
# transitive dependencies from replacing the image NPU stack or source DS.
install_example_dependencies() {
  local constraint_dir="$GITHUB_WORKSPACE/.ci/deepspeed-dependencies"
  local constraint_file="$constraint_dir/protected-stack.txt"
  mkdir -p "$constraint_dir"
  python - "$constraint_file" <<'PY'
from importlib.metadata import version
from pathlib import Path
import os
import sys

protected = ("torch", "torch-npu", "deepspeed")
constraints = "".join(f"{name}=={version(name)}\n" for name in protected)
# Reuse the global scientific stack pins; keep SciPy and NumPy paired.
constraints += Path(os.environ['PIP_CONSTRAINT']).read_text(encoding='utf-8')
Path(sys.argv[1]).write_text(constraints)
print("preserving installed runtime dependencies:\n" + constraints)
PY
  python -m pip install --constraint "$constraint_file" "$@"
}

# New examples accept explicit fixture paths, so stage their data outside the
# examples checkout instead of changing any upstream source or dataset loader.
plant_ci_fixture() {
  local fixture="$1"
  local env_name="$2"
  local ci_dir="$GITHUB_WORKSPACE/.ci/deepspeed-fixtures"
  mkdir -p "$ci_dir"
  cp "$FIXTURE_DIR/$fixture" "$ci_dir/$fixture"
  export "$env_name=$ci_dir/$fixture"
  echo "$env_name=$ci_dir/$fixture" >> "$GITHUB_ENV"
  echo "staged $fixture -> $ci_dir/$fixture ($env_name)"
}

setup_ds_hf_autotp() {
  install_deepspeed_source
  # train.py imports Trainer directly; the extra evaluation/OpenAI packages in
  # the upstream requirements belong to utils.py, which this entry never imports.
  install_example_dependencies "transformers>=4.51.0,<5" "accelerate>=1.10.1,<2" \
    sentencepiece psutil numpy safetensors
  ms_download_models "OPT_125M_PATH=facebook/opt-125m"
  plant_ci_fixture ci_alpaca_16.json ALPACA_CI_PATH
  python - <<'PY'
import accelerate
import importlib.util
import os
from pathlib import Path
import psutil
import sys
import sentencepiece
import torch_npu
import transformers
from transformers import AutoTokenizer, Trainer

entry = Path(os.environ["EXAMPLES_ROOT"]) / "training/tensor_parallel/hf_integration/train.py"
spec = importlib.util.spec_from_file_location("ds_ci_hf_autotp", entry)
upstream = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = upstream
spec.loader.exec_module(upstream)
tokenizer = AutoTokenizer.from_pretrained(os.environ["OPT_125M_PATH"],
                                         model_max_length=128, padding_side="right", use_fast=False)
special_tokens = {}
for kind in ("pad", "eos", "bos", "unk"):
    if getattr(tokenizer, f"{kind}_token") is None:
        special_tokens[f"{kind}_token"] = getattr(upstream, f"DEFAULT_{kind.upper()}_TOKEN")
tokenizer.add_special_tokens(special_tokens)
# SupervisedDataset writes a fixed dataset_dict.pkl in cwd. Precompute it once
# before eight ranks start, in the output directory used by the project runner.
# The engine supplies CI_OUTPUT_DIR only to run, whose default is workspace/output.
output_root = Path(os.environ.get("CI_OUTPUT_DIR", str(Path(os.environ["GITHUB_WORKSPACE"]) / "output")))
work_dir = output_root / "hf-autotp-work"
work_dir.mkdir(parents=True, exist_ok=True)
cache = work_dir / "dataset_dict.pkl"
# A runner may retain output from a previous job. This file is our generated
# tokenization cache; rebuild it from this run's fixture before workers read it.
cache.unlink(missing_ok=True)
previous_cwd = Path.cwd()
try:
    os.chdir(work_dir)
    dataset = upstream.SupervisedDataset(os.environ["ALPACA_CI_PATH"], tokenizer)
finally:
    os.chdir(previous_cwd)
counts = [(labels != upstream.IGNORE_INDEX).sum().item() for labels in dataset.labels]
if len(dataset) != 16 or any(count == 0 for count in counts):
    raise SystemExit(f"HF AutoTP fixture lost training labels at length 128: {counts}")
if not cache.is_file() or cache.stat().st_size == 0:
    raise SystemExit(f"HF AutoTP pre-tokenized cache missing: {cache}")
print("HF AutoTP shared pre-tokenized cache:", cache)
print("HF AutoTP non-masked target tokens per fixture row:", counts)

print("HF AutoTP dependencies:", "transformers", transformers.__version__,
      "accelerate", accelerate.__version__, "psutil", psutil.__version__)
PY
}

setup_ds_hf_bench_length() {
  install_deepspeed_source
  # The sibling utils.py imports the legacy OpenAIObject API at module import.
  # This dependency enables ordinary imports only; this recipe makes no API call.
  install_example_dependencies "transformers>=4.51.0,<5" "accelerate>=1.10.1,<2" \
    "openai==0.28.1" sentencepiece psutil numpy safetensors
  ms_download_models "OPT_125M_PATH=facebook/opt-125m"
  plant_ci_fixture ci_alpaca_16.json ALPACA_CI_PATH
  python - <<'PY'
import accelerate
import importlib.util
import os
from pathlib import Path
import sys

import numpy
import openai
from openai import openai_object
import torch_npu
import transformers
from transformers import AutoTokenizer

entry = Path(os.environ["EXAMPLES_ROOT"]) / "training/tensor_parallel/hf_integration/train_bench_length.py"
sys.path.insert(0, str(entry.parent))
spec = importlib.util.spec_from_file_location("ds_ci_hf_bench_length", entry)
upstream = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = upstream
spec.loader.exec_module(upstream)
tokenizer = AutoTokenizer.from_pretrained(os.environ["OPT_125M_PATH"],
                                         model_max_length=128, padding_side="right", use_fast=False)
special_tokens = {}
for kind in ("pad", "eos", "bos", "unk"):
    if getattr(tokenizer, f"{kind}_token") is None:
        special_tokens[f"{kind}_token"] = getattr(upstream, f"DEFAULT_{kind.upper()}_TOKEN")
tokenizer.add_special_tokens(special_tokens)
output_root = Path(os.environ.get("CI_OUTPUT_DIR", str(Path(os.environ["GITHUB_WORKSPACE"]) / "output")))
work_dir = output_root / "hf-bench-length-work"
work_dir.mkdir(parents=True, exist_ok=True)
cache = work_dir / "dataset_dict128.pkl"
# Our generated shared cache is rebuilt before the eight readers start.
cache.unlink(missing_ok=True)
previous_cwd = Path.cwd()
try:
    os.chdir(work_dir)
    dataset = upstream.SupervisedDataset(os.environ["ALPACA_CI_PATH"], tokenizer)
finally:
    os.chdir(previous_cwd)
counts = [(labels != upstream.IGNORE_INDEX).sum().item() for labels in dataset.labels]
if len(dataset) != 16 or any(count == 0 for count in counts):
    raise SystemExit(f"fixed-length HF fixture lost training labels: {counts}")
for ids, labels in zip(dataset.input_ids, dataset.labels):
    if ids.numel() != 128 or labels.numel() != 128:
        raise SystemExit("HF benchmark cache does not contain fixed 128-token rows")
    if not bool((labels[ids == tokenizer.pad_token_id] == upstream.IGNORE_INDEX).all()):
        raise SystemExit("HF benchmark cache contains unmasked padding targets")
if not cache.is_file() or cache.stat().st_size == 0:
    raise SystemExit(f"fixed-length HF pre-tokenized cache missing: {cache}")
print("HF benchmark shared pre-tokenized cache:", cache)
print("HF benchmark non-masked target tokens:", counts)
print("HF benchmark dependencies:", "transformers", transformers.__version__,
      "accelerate", accelerate.__version__, "openai", openai.__version__, "numpy", numpy.__version__)
PY
}

setup_ds_superoffload() {
  install_deepspeed_source
  # finetune_zero3 imports wandb unconditionally, but the runner never enables
  # its network logging. CPU Adam is compiled up front rather than on every rank.
  install_example_dependencies "transformers>=4.56.1,<5" "accelerate>=1.10.1,<2" \
    "datasets>=4,<5" "numpy>=1.21.0" packaging psutil pyarrow \
    sentencepiece safetensors wandb ninja
  apt-get install -y build-essential
  ms_download_models "OPT_125M_PATH=facebook/opt-125m"
  plant_ci_fixture ci_alpaca_16.json ALPACA_CI_PATH
  export ALPACA_DATASET_DIR="$GITHUB_WORKSPACE/.ci/deepspeed-fixtures/alpaca-parquet"
  mkdir -p "$ALPACA_DATASET_DIR"
  echo "ALPACA_DATASET_DIR=$ALPACA_DATASET_DIR" >> "$GITHUB_ENV"
  python - <<'PY'
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys

import datasets
import torch_npu
import transformers
import wandb
from datasets import Dataset, load_dataset
from deepspeed.ops.op_builder import CPUAdamBuilder

rows = json.loads(Path(os.environ["ALPACA_CI_PATH"]).read_text())
if len(rows) != 16 or any(set(row) != {"instruction", "input", "output"} for row in rows):
    raise SystemExit("SuperOffload fixture must contain 16 Alpaca instruction/input/output rows")
dataset_dir = Path(os.environ["ALPACA_DATASET_DIR"])
Dataset.from_list(rows).to_parquet(str(dataset_dir / "train.parquet"))
# The upstream entry passes a directory directly to load_dataset(), not the
# JSON loader or load_from_disk(). Validate precisely that public loading path.
dataset = load_dataset(str(dataset_dir))
if len(dataset["train"]) != 16 or set(dataset["train"].column_names) != {"instruction", "input", "output"}:
    raise SystemExit("local parquet directory did not resolve to the 16-row train split")
entry = Path(os.environ["EXAMPLES_ROOT"]) / "training/DeepSpeed-SuperOffload/finetune_zero3.py"
spec = importlib.util.spec_from_file_location("ds_ci_superoffload", entry)
upstream = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = upstream
spec.loader.exec_module(upstream)
tokenizer = upstream.load_tokenizer(os.environ["OPT_125M_PATH"], logging.getLogger("ci-superoffload"))
for row in dataset["train"]:
    batch = upstream.preprocess_alpaca_example(row, tokenizer, max_length=128)
    if len(batch["input_ids"]) != 128 or len(batch["labels"]) != 128:
        raise SystemExit("SuperOffload fixture does not produce 128-token training rows")
    if sum(batch["attention_mask"]) < 2 or batch["labels"] != batch["input_ids"]:
        raise SystemExit("SuperOffload fixture lost the upstream causal language-model targets")
builder = CPUAdamBuilder()
print("checking SuperOffload CPU Adam extension:", type(builder).__name__)
builder.load(verbose=True)
print("SuperOffload CPU Adam extension loaded successfully")
print("SuperOffload local parquet train split:", len(dataset["train"]), dataset_dir)
print("SuperOffload upstream optimizer LR:", upstream.DEFAULT_OPTIMIZER_LR)
print("SuperOffload dependencies:", "datasets", datasets.__version__,
      "transformers", transformers.__version__, "wandb", wandb.__version__)
PY
}

setup_ds_variable_batch() {
  install_deepspeed_source
  # DeepSpeed's DataAnalyzer/data-sampling path uses these packages. The
  # example itself constructs its small model and synthetic dataset locally.
  install_example_dependencies numpy pandas
  python - <<'PY'
import numpy
import pandas
import torch_npu
from deepspeed.runtime.data_pipeline.data_sampling.variable_batch_size_and_lr import (
    get_dataloader_and_lr_scheduler_for_variable_batch_size_deepspeed,
)

print("dynamic batch dependencies:", "numpy", numpy.__version__,
      "pandas", pandas.__version__)
PY
}

setup_ds_zenflow() {
  install_deepspeed_source
  # benchmark/requirements.txt without its torch/deepspeed entries. CPU Adam
  # is imported here; its actual NPU/offload execution is checked by the run.
  install_example_dependencies "datasets>=2.14.1" "transformers>=4.37.2,<5" \
    "numpy>=1.21.0" tabulate pandas ninja
  echo "installing C++ build tools for upstream ZenFlow CPU Adam"
  apt-get install -y build-essential
  python - <<'PY'
import datasets
import numpy
import pandas
import tabulate
import torch_npu
import transformers
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.ops.op_builder import CPUAdamBuilder

builder = CPUAdamBuilder()
print("checking ZenFlow CPU Adam extension:", type(builder).__name__)
builder.load(verbose=True)
print("ZenFlow CPU Adam extension loaded successfully")

print("ZenFlow dependencies:", "datasets", datasets.__version__,
      "transformers", transformers.__version__, "numpy", numpy.__version__,
      "pandas", pandas.__version__, "tabulate", tabulate.__version__)
PY
}

setup_ds_rac_prune() {
  install_deepspeed_source
  # Local prompt JSONL takes the standard-library branch of the calibration
  # loader. No datasets, vLLM, evaluation backend or runtime patch is needed.
  install_example_dependencies "transformers>=4.45.0,<5" "accelerate>=0.30" \
    sentencepiece safetensors
  ms_download_models "OPT_125M_PATH=facebook/opt-125m"
  plant_ci_fixture ci_rac_8.jsonl RAC_CI_PATH
  python - <<'PY'
import os
from pathlib import Path
import sys

import accelerate
import torch_npu
import transformers

sys.path.insert(0, str(Path(os.environ["EXAMPLES_ROOT"]) /
                       "compression/reasoning_aware_compression"))
from rac import build_calibration_samples, sequential_prune
from rac.wanda import Wanda
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(os.environ["OPT_125M_PATH"])
samples = build_calibration_samples(
    "prompt", tokenizer, nsamples=4, seqlen=32, seed=0,
    dataset_name=os.environ["RAC_CI_PATH"], prompt_column="prompt",
    use_chat_template=False,
)
if len(samples) != 4 or any(sample.numel() != 32 for sample in samples):
    raise SystemExit("RAC fixture cannot fill the required 4 x 32 token windows")
print("RAC calibration fixture:", len(samples), "windows x", samples[0].numel(), "tokens")

print("Wanda pruning dependencies:", "transformers", transformers.__version__,
      "accelerate", accelerate.__version__)
PY
}

setup_ds_pin_memory() {
  install_deepspeed_source
  install_example_dependencies numpy ninja
  apt-get install -y build-essential
  # Adjust the shell so its Python/compiler subprocesses inherit the allowed
  # limit. The later run step independently makes the same bounded adjustment.
  local memlock_hard
  memlock_hard="$(ulimit -Hl)"
  ulimit -Sl "$memlock_hard"
  echo "pin-memory shell memlock soft/hard KiB: $(ulimit -Sl)/$memlock_hard"
  python - <<'PY'
import os
import resource

import torch
import torch_npu
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import CPUAdamBuilder, PinMemoryBuilder
from deepspeed.utils.pin_memory import get_native_pinned_memory

# Never increase the container's hard limit or require extra privileges.
soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
print("pin-memory RLIMIT_MEMLOCK soft/hard bytes:", soft, hard)
required = 4 * 1024 * 1024
if soft != resource.RLIM_INFINITY and soft < required:
    raise SystemExit(f"small native pin-memory CI recipes need at least {required} locked bytes; container limit is {soft}")
for builder in (CPUAdamBuilder(), PinMemoryBuilder()):
    print("preloading pin-memory benchmark extension:", type(builder).__name__)
    builder.load(verbose=True)
os.environ["DS_PIN_MEMORY_BACKEND"] = "native"
os.environ["DS_PIN_MEMORY_REGISTER_DEVICE"] = "0"
manager = get_native_pinned_memory()
raw = torch.arange(1024 * 1024 // 4, dtype=torch.float32)
locked = manager.pin(raw)
try:
    if not manager.is_pinned(locked) or not torch.equal(locked, raw):
        raise SystemExit("native page-locked allocation failed the pin/content preflight")
    accelerator = get_accelerator()
    device = locked.to(accelerator.current_device_name())
    accelerator.synchronize()
    if not torch.equal(device.cpu(), raw):
        raise SystemExit("native page-locked host/NPU copy lost tensor contents")
finally:
    manager.unpin(locked)
print("native mlock host allocation and NPU copy verified; this is not CUDA host registration")
PY
}

setup_ds_zenflow_finetune() {
  install_deepspeed_source
  install_example_dependencies "transformers==4.57.6" "datasets>=4,<5" \
    "accelerate>=1.10.1,<2" numpy pyarrow sentencepiece safetensors ninja
  apt-get install -y build-essential
  ms_download_models "OPT_125M_PATH=facebook/opt-125m"
  plant_ci_fixture ci_alpaca_16.json ALPACA_CI_PATH
  export ZENFLOW_WORK_DIR="${CI_OUTPUT_DIR:-$GITHUB_WORKSPACE/output}/zenflow-finetune-work"
  mkdir -p "$ZENFLOW_WORK_DIR/tatsu-lab/alpaca"
  echo "ZENFLOW_WORK_DIR=$ZENFLOW_WORK_DIR" >> "$GITHUB_ENV"
  echo "HF_DATASETS_OFFLINE=1" >> "$GITHUB_ENV"
  HF_DATASETS_OFFLINE=1 python - <<'PY'
import importlib.util
import json
import os
from pathlib import Path
import sys

import datasets
import torch_npu
import transformers
from datasets import Dataset, load_dataset
from deepspeed.ops.op_builder import CPUAdamBuilder
from transformers import AutoTokenizer, default_data_collator

rows = json.loads(Path(os.environ["ALPACA_CI_PATH"]).read_text())
if len(rows) != 16 or any(set(row) != {"instruction", "input", "output"} for row in rows):
    raise SystemExit("ZenFlow fixture must contain 16 original Alpaca-schema rows")
work = Path(os.environ["ZENFLOW_WORK_DIR"]).resolve()
Dataset.from_list(rows).to_parquet(str(work / "tatsu-lab/alpaca/train.parquet"))
os.chdir(work)
# Match the unmodified entry's hardcoded public API call precisely. The local
# directory is resolved natively; no dataset shim or source rewrite is used.
dataset = load_dataset("tatsu-lab/alpaca")
if len(dataset["train"]) != 16 or set(dataset["train"].column_names) != {"instruction", "input", "output"}:
    raise SystemExit("ZenFlow's exact hardcoded dataset name did not resolve to the local fixture")
entry = Path(os.environ["EXAMPLES_ROOT"]) / "training/DeepSpeed-ZenFlow/finetuning/finetune_llama.py"
spec = importlib.util.spec_from_file_location("ds_ci_zenflow_finetune", entry)
upstream = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = upstream
spec.loader.exec_module(upstream)
tokenizer = AutoTokenizer.from_pretrained(os.environ["OPT_125M_PATH"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# Keep the original string columns just as the entry does. The official
# collator ignores strings, so the engine receives only model input tensors.
tokenized = dataset["train"].map(lambda row: upstream.preprocess_alpaca(row, tokenizer), batched=False)
for row in tokenized:
    if len(row["input_ids"]) != 512 or row["labels"] != row["input_ids"] or sum(row["attention_mask"]) < 2:
        raise SystemExit("ZenFlow's original preprocessing produced invalid causal training targets")
batch = default_data_collator([tokenized[0], tokenized[1]])
if set(batch) != {"input_ids", "attention_mask", "labels"} or batch["input_ids"].shape != (2, 512):
    raise SystemExit(f"ZenFlow's original collator did not drop auxiliary strings: {list(batch)}")
CPUAdamBuilder().load(verbose=True)
print("ZenFlow local offline name resolution:", work / "tatsu-lab/alpaca", "rows:", len(tokenized))
print("ZenFlow finetuning dependencies:", "transformers", transformers.__version__, "datasets", datasets.__version__)
PY
}

setup_ds_opsd() {
  install_deepspeed_source
  install_example_dependencies "transformers==4.57.6" "accelerate>=1.10.1,<2" \
    numpy sentencepiece safetensors ninja
  apt-get install -y build-essential
  ms_download_models "QWEN25_STUDENT_PATH=Qwen/Qwen2.5-0.5B-Instruct" \
    "QWEN25_TEACHER_PATH=Qwen/Qwen2.5-0.5B"
  plant_ci_fixture ci_rac_8.jsonl OPSD_CI_PATH
  python - <<'PY'
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch_npu
import transformers
from deepspeed.ops.op_builder import CPUAdamBuilder
from deepspeed.runtime.rollout import RolloutConfig, build_rollout
from transformers import AutoConfig, AutoTokenizer

student = AutoTokenizer.from_pretrained(os.environ["QWEN25_STUDENT_PATH"], padding_side="left")
teacher = AutoTokenizer.from_pretrained(os.environ["QWEN25_TEACHER_PATH"], padding_side="left")
# KL compares the same token index in each logit tensor. Validate the entire
# mapping, including added tokens, rather than merely comparing its size.
if student.get_vocab() != teacher.get_vocab() or student.get_added_vocab() != teacher.get_added_vocab():
    raise SystemExit("OPSD student/teacher token IDs differ; token-wise distillation is invalid")
student_config = AutoConfig.from_pretrained(os.environ["QWEN25_STUDENT_PATH"])
teacher_config = AutoConfig.from_pretrained(os.environ["QWEN25_TEACHER_PATH"])
if student_config.vocab_size != teacher_config.vocab_size or max(student.get_vocab().values()) >= student_config.vocab_size:
    raise SystemExit("OPSD model logit vocabularies do not align with their token IDs")
if not student.chat_template:
    raise SystemExit("OPSD instruction student must provide its own chat template")
entry = Path(os.environ["EXAMPLES_ROOT"]) / "training/opsd/main.py"
sys.path.insert(0, str(entry.parent))
spec = importlib.util.spec_from_file_location("ds_ci_opsd", entry)
upstream = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = upstream
spec.loader.exec_module(upstream)
dataset = upstream.PromptDataset(os.environ["OPSD_CI_PATH"], tokenizer=student,
                                max_prompt_length=128, prompt_field="prompt")
if len(dataset) != 8 or any(not isinstance(dataset[index], str) or not dataset[index] for index in range(len(dataset))):
    raise SystemExit("OPSD fixture must produce eight nonempty native chat-templated prompts")
collator = upstream.LeftPaddedPromptCollator(student, max_prompt_length=128)
batch = collator([dataset[0], dataset[1]])
if batch["prompt_ids"].shape[0] != 2 or not bool((batch["prompt_attention_mask"].sum(dim=1) > 0).all()):
    raise SystemExit("OPSD native prompt collator lost valid prompt tokens")
CPUAdamBuilder().load(verbose=True)
print("OPSD shared token mapping verified:", len(student), "model vocab size:", student_config.vocab_size)
print("OPSD student/teacher use different checkpoint weights; student's chat/EOS policy drives rollout")
print("OPSD rollout API available:", RolloutConfig.__name__, build_rollout.__name__)
print("OPSD local prompt fixture:", len(dataset), "transformers:", transformers.__version__)
PY
}

verify_data_stack() {
  python - <<'PY'
import numpy as np
import pandas as pd
import pyarrow as pa
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix

versions = {'numpy': np.__version__, 'scipy': scipy.__version__,
            'pyarrow': pa.__version__, 'pandas': pd.__version__}
print('scientific/data stack:', versions)
expected = {'numpy': '1.26.4', 'scipy': '1.16.3',
            'pyarrow': '20.0.0', 'pandas': '2.2.3'}
if versions != expected:
    raise SystemExit(f'incompatible scientific/data stack: {versions}; expected {expected}')
matrix = csr_matrix(np.eye(2))
assert matrix.sum() == 2
rows, columns = linear_sum_assignment([[0, 1], [1, 0]])
assert rows.tolist() == columns.tolist() == [0, 1]
table = pa.Table.from_pandas(pd.DataFrame({'value': [1, 2]}))
assert table.column('value').to_pylist() == [1, 2]
print('SciPy sparse/optimize and Arrow/Pandas checks passed')
PY
}

verify_installed_runtime() {
  python - "$PROFILE" <<'PY'
import os
from pathlib import Path
import sys

import deepspeed
import torch
import torch_npu
from deepspeed.accelerator import get_accelerator

source_root = Path(os.environ["TARGET_ROOT"]).resolve()
deepspeed_file = Path(deepspeed.__file__).resolve()
print("runtime torch:", torch.__version__)
print("runtime torch_npu:", torch_npu.__version__)
print("runtime deepspeed:", deepspeed.__version__, deepspeed_file)
if sys.argv[1] in {"ds_cifar", "ds_hf_autotp", "ds_variable_batch", "ds_zenflow", "ds_rac_prune",
                  "ds_chat_prompt_eval", "ds_chat_reward_eval", "ds_hf_bench_length", "ds_superoffload",
                  "ds_pin_memory", "ds_zenflow_finetune", "ds_opsd", "ds_finetune_demo",
                  "ds_sd_distil", "ds_opsd_decode", "ds_legacy_inference_bench", "ds_hf_ds_compare",
                  "ds_fill_mask_bert", "ds_fill_mask_electra", "ds_fill_mask_roberta",
                  "ds_t5_translation", "ds_hybrid_rollout", "ds_asr_ctc"}:
    accelerator = get_accelerator()._name
    available = torch.npu.is_available()
    print("runtime accelerator:", accelerator, "NPU available:", available)
    if accelerator != "npu" or not available:
        raise SystemExit(f"new DeepSpeed NPU example requires available npu accelerator, got {accelerator}")
    # Availability alone does not exercise the CANN Python compiler imports.
    probe = torch.ones(2, device=get_accelerator().device_name()).sum()
    if probe.item() != 2:
        raise SystemExit('NPU scientific-stack preflight produced an unexpected result')
    print('NPU allocation/reduction preflight passed')
try:
    deepspeed_file.relative_to(source_root)
except ValueError as exc:
    raise SystemExit(
        f"DeepSpeed was not imported from target source {source_root}: "
        f"{deepspeed_file}") from exc
PY
}

# Optional profiles use the same protected dependency and source-install helpers.
setup_inference_dependencies() {
  install_deepspeed_source
  # 4.51+ resets pipeline.device to a CPU-loaded model.device when distributed
  # is already initialized; these older recipes require the pre-change behavior.
  install_example_dependencies 'transformers==4.44.2' 'accelerate>=0.30,<2' \
    sentencepiece protobuf safetensors numpy
}

plant_inference_alias() {
  local snapshot="$1" alias="$2"
  local work="$GITHUB_WORKSPACE/.ci/deepspeed-inference/$PROFILE"
  [[ -d "$snapshot" && "$alias" != /* && "$alias" != *'..'* ]] || {
    echo "invalid local inference alias: $alias -> $snapshot" >&2
    return 1
  }
  mkdir -p "$work/$(dirname "$alias")"
  if [[ -e "$work/$alias" || -L "$work/$alias" ]]; then
    [[ -L "$work/$alias" && "$(readlink "$work/$alias")" == "$snapshot" ]] || {
      echo "refusing to replace existing inference asset: $work/$alias" >&2
      return 1
    }
  else
    ln -s "$snapshot" "$work/$alias"
  fi
  export INFERENCE_CI_WORK="$work"
  printf 'INFERENCE_CI_WORK=%s\nHF_HUB_OFFLINE=1\nTRANSFORMERS_OFFLINE=1\n' "$work" >> "$GITHUB_ENV"
}

setup_ds_legacy_inference_bench() {
  setup_inference_dependencies
  case "${EXEC:-${EXAMPLE_PATH:-}}" in
    benchmarks/inference/bert-bench.py) ms_download_models 'BERT_BASE_CASED_PATH=AI-ModelScope/bert-base-cased' ;;
    benchmarks/inference/gpt-bench.py) ms_download_models 'OPT_125M_PATH=facebook/opt-125m' ;;
    *) echo 'unknown legacy inference benchmark entry' >&2; return 1 ;;
  esac
}

setup_ds_hf_ds_compare() {
  setup_inference_dependencies
  ms_download_models 'OPT_125M_PATH=facebook/opt-125m'
}

setup_ds_fill_mask_bert() {
  setup_inference_dependencies
  ms_download_models 'BERT_LARGE_CASED_PATH=AI-ModelScope/bert-large-cased'
  plant_inference_alias "$BERT_LARGE_CASED_PATH" 'bert-large-cased'
}

setup_ds_fill_mask_electra() {
  setup_inference_dependencies
  ms_download_models 'ELECTRA_GENERATOR_PATH=google/electra-base-generator'
  plant_inference_alias "$ELECTRA_GENERATOR_PATH" 'google/electra-base-generator'
}

setup_ds_fill_mask_roberta() {
  setup_inference_dependencies
  ms_download_models 'ROBERTA_LARGE_PATH=AI-ModelScope/roberta-large'
  plant_inference_alias "$ROBERTA_LARGE_PATH" 'roberta-large'
}

setup_ds_t5_translation() {
  setup_inference_dependencies
  ms_download_models 'T5_BASE_PATH=AI-ModelScope/t5-base'
  # A separate asset view bounds generation, without writing ModelScope cache
  # or changing the upstream script. Weights/tokenizer files remain symlinks.
  local work="$GITHUB_WORKSPACE/.ci/deepspeed-inference/$PROFILE"
  mkdir -p "$work/t5-base"
  python - "$T5_BASE_PATH" "$work/t5-base" <<'PY'
import json
import sys
from pathlib import Path
import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, GenerationConfig

snapshot, view = map(Path, sys.argv[1:])
for source in snapshot.iterdir():
    if source.name in ('generation_config.json', 'config.json'):
        continue
    target = view / source.name
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise SystemExit(f'unexpected T5 asset alias: {target}')
    elif target.exists():
        raise SystemExit(f'refusing to replace T5 asset: {target}')
    else:
        target.symlink_to(source, target_is_directory=source.is_dir())
config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
if config.model_type != 't5' or config.num_heads % 2:
    raise SystemExit('translation requires a T5 model with heads divisible by TP=2')
with torch.device('meta'):
    model = AutoModelForSeq2SeqLM.from_config(config)
names = [name for name, _ in model.named_modules()]
for suffix in ('SelfAttention.o', 'EncDecAttention.o', 'DenseReluDense.wo'):
    if not any(name.endswith(suffix) for name in names):
        raise SystemExit(f'T5 injection target missing: {suffix}')
generation = GenerationConfig.from_model_config(config)
generation.max_new_tokens = 8
generation.do_sample = False
# Pipeline applies legacy task_specific_params after generation_config loading.
# Keep its translation task semantics, but supply a legal short min_length.
generation.min_length = 0
for name, task in (config.task_specific_params or {}).items():
    if name.startswith('translation'):
        task['min_length'] = 0
        task['max_length'] = 16
config.save_pretrained(view)
generation.save_pretrained(view)
print('T5 TP policy checked; local generation_config limits output to 8 new tokens')
PY
  export INFERENCE_CI_WORK="$work"
  printf 'INFERENCE_CI_WORK=%s\nHF_HUB_OFFLINE=1\nTRANSFORMERS_OFFLINE=1\n' "$work" >> "$GITHUB_ENV"
}

setup_ds_asr_ctc() {
  setup_inference_dependencies
  # datasets 4 returns Column objects for result['text']; this original script
  # and jiwer 3 expect ordinary lists. Keep the supported list-valued API.
  install_example_dependencies 'datasets==3.6.0' 'jiwer>=3,<4' soundfile pyarrow
  ms_download_models 'WAV2VEC2_PATH=AI-ModelScope/wav2vec2-base-960h'
  plant_inference_alias "$WAV2VEC2_PATH" 'facebook/wav2vec2-base-960h'
  # Native local-directory resolution satisfies the upstream's fixed
  # load_dataset("librispeech_asr", "clean", split="test") call. These are
  # synthetic waveforms, deliberately a CTC forward smoke, not LibriSpeech WER.
  python - "$INFERENCE_CI_WORK" <<'PY'
import json
import math
from pathlib import Path
import struct
import sys
import wave
from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq

work = Path(sys.argv[1])
dataset = work / 'librispeech_asr'
dataset.mkdir(exist_ok=True)
rows = []
for index in range(2):
    audio = dataset / f'ci-wave-{index}.wav'
    values = [int(2500 * math.sin(2 * math.pi * (220 + index * 110) * i / 16000)) for i in range(16000)]
    with wave.open(str(audio), 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(struct.pack('<' + 'h' * len(values), *values))
    rows.append({'file': str(audio), 'text': 'HELLO WORLD'})
pq.write_table(pa.Table.from_pylist(rows), dataset / 'test.parquet')
(dataset / 'README.md').write_text('---\nconfigs:\n- config_name: clean\n  data_files:\n  - split: test\n    path: test.parquet\n---\nSynthetic CTC CI inputs, not LibriSpeech.\n')
# Validate the exact original loader contract, not an alternative builder.
import os
os.chdir(work)
loaded = load_dataset('librispeech_asr', 'clean', split='test')
if len(loaded) != 2 or not all(Path(row['file']).is_file() for row in loaded):
    raise SystemExit('native clean/test CTC fixture loader contract failed')
if not isinstance(loaded['text'], list):
    raise SystemExit('CTC original jiwer call requires a list-valued dataset column')
print('prepared two synthetic 16 kHz CTC inputs; not a LibriSpeech accuracy benchmark')
PY
}
setup_ds_finetune_demo() {
  install_deepspeed_source
  # The upstream rotary reset requires base/dim but sets max_seq_len_cached=None.
  # Llama 4.42 uses base/dim and position_ids without a cached-length comparison;
  # newer Llama/Qwen and older Qwen rotary implementations do not meet both rules.
  install_example_dependencies 'transformers==4.42.4' 'accelerate>=1.0,<2' \
    'datasets>=4,<5' safetensors sentencepiece wandb
  ms_download_models 'SMOLLM2_135M_PATH=HuggingFaceTB/SmolLM2-135M'
  plant_ci_fixture ci_alpaca_16.json ALPACA_CI_PATH
  export ALPACA_FINETUNE_DIR="$GITHUB_WORKSPACE/.ci/deepspeed-fixtures/finetune-demo"
  echo "ALPACA_FINETUNE_DIR=$ALPACA_FINETUNE_DIR" >> "$GITHUB_ENV"
  python - <<'PY'
import importlib.util
import json
import os
from pathlib import Path
import sys

from datasets import Dataset, load_dataset
import torch
import torch_npu
import transformers
from transformers import AutoConfig, AutoTokenizer, LlamaConfig, LlamaForCausalLM

if transformers.__version__ != '4.42.4':
    raise SystemExit('finetune demo requires Transformers 4.42.4 rotary semantics')
config = AutoConfig.from_pretrained(os.environ['SMOLLM2_135M_PATH'], local_files_only=True)
if config.model_type != 'llama' or config.rope_scaling is not None:
    raise SystemExit('finetune demo requires the original SmolLM2 Llama/default-RoPE config')
entry = Path(os.environ['EXAMPLES_ROOT']) / 'training/deepspeed_finetune_demo/finetune_llama.py'
sys.path.insert(0, str(entry.parent))
spec = importlib.util.spec_from_file_location('ds_ci_finetune_demo', entry)
upstream = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = upstream
spec.loader.exec_module(upstream)
# Execute the exact upstream reset, then a real CPU forward/backward. This
# preflight detects deterministic dependency/API incompatibilities before NPU launch.
with torch.device('cpu'):
    tiny = LlamaForCausalLM(LlamaConfig(vocab_size=128, hidden_size=32,
                                    intermediate_size=64, num_hidden_layers=2,
                                    num_attention_heads=4, num_key_value_heads=2,
                                    rope_scaling=None, max_position_embeddings=128))
upstream._reset_rotary_embeddings(tiny)
ids = torch.arange(16, device='cpu').reshape(1, 16)
loss = tiny(input_ids=ids, labels=ids).loss
if not torch.isfinite(loss):
    raise SystemExit('upstream rotary-reset CPU preflight produced nonfinite loss')
loss.backward()
print('real upstream Llama rotary-reset CPU forward/backward:', loss.item())
tokenizer = AutoTokenizer.from_pretrained(os.environ['SMOLLM2_135M_PATH'], local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
rows = json.loads(Path(os.environ['ALPACA_CI_PATH']).read_text())
directory = Path(os.environ['ALPACA_FINETUNE_DIR'])
directory.mkdir(parents=True, exist_ok=True)
Dataset.from_list(rows).to_parquet(directory / 'train.parquet')
dataset = load_dataset(str(directory))['train']
if len(dataset) != 16:
    raise SystemExit('finetune demo local dataset must contain exactly 16 rows')
for row in dataset:
    encoded = upstream.preprocess_alpaca(row, tokenizer, max_length=128)
    if len(encoded['input_ids']) != 128 or not any(x != -100 for x in encoded['labels'][1:]):
        raise SystemExit('finetune demo fixture lost shifted training labels at length 128')
print('finetune demo native local Parquet and labels:', directory, len(dataset))
PY
}

download_sd15_ci_model() {
  install_example_dependencies 'modelscope==1.37.0' safetensors
  TQDM_MININTERVAL="${TQDM_MININTERVAL:-15}" \
    MODELSCOPE_DOWNLOAD_PARALLELS=1 MODELSCOPE_ENABLE_DEFAULT_HASH_VALIDATION=true \
    python - <<'PY'
import json
import os
from pathlib import Path
import tempfile
import time
from modelscope import snapshot_download
from safetensors import safe_open

# Full Diffusers components, but not duplicate .bin/root .ckpt weights. The
# final unchanged entry reloads the safety checker when constructing its pipeline.
patterns = [
    'model_index.json', 'feature_extractor/preprocessor_config.json',
    'scheduler/scheduler_config.json', 'tokenizer/*',
    'text_encoder/config.json', 'text_encoder/model.safetensors',
    'unet/config.json', 'unet/diffusion_pytorch_model.safetensors',
    'vae/config.json', 'vae/diffusion_pytorch_model.safetensors',
    'safety_checker/config.json', 'safety_checker/model.safetensors',
]
required = [name for name in patterns if '*' not in name]
required += ['tokenizer/vocab.json', 'tokenizer/merges.txt', 'tokenizer/tokenizer_config.json']


def validate_components(local):
    for name in required:
        asset = Path(local) / name
        if not asset.is_file() or not asset.stat().st_size:
            raise RuntimeError(f'SD15 complete local Diffusers asset missing: {asset}')
        if asset.suffix == '.json':
            json.loads(asset.read_text())
        elif asset.suffix == '.safetensors':
            with safe_open(str(asset), framework='np') as weights:
                if not list(weights.keys()):
                    raise RuntimeError(f'SD15 empty component weights: {asset}')
    print('validated SD15 tokenizer and all four weight components:', local, flush=True)


cache = tempfile.mkdtemp(prefix='deepspeed-sd15-', dir='/tmp')
print('private SD15 ModelScope cache:', cache, flush=True)
for attempt in range(3):
    try:
        path = snapshot_download('AI-ModelScope/stable-diffusion-v1-5',
                                 allow_patterns=patterns, cache_dir=cache,
                                 enable_file_lock=True, max_workers=2)
        validate_components(path)
        break
    except Exception as exc:
        print(f'SD15 attempt {attempt + 1}/3 failed: {exc}', flush=True)
        if attempt == 2:
            raise
        cache = tempfile.mkdtemp(prefix='deepspeed-sd15-retry-', dir='/tmp')
        time.sleep(2 * (attempt + 1))
with open(os.environ['GITHUB_ENV'], 'a') as handle:
    handle.write(f'SD15_PATH={path}\n')
print('complete ModelScope SD15 Diffusers components:', path)
PY
  local model_path
  model_path="$(awk 'index($0,"SD15_PATH=")==1 { value=substr($0,11) } END {print value}' "$GITHUB_ENV")"
  [[ -d "$model_path" ]] || { echo "SD15 download did not expose a local directory" >&2; return 1; }
  export SD15_PATH="$model_path"
}

setup_ds_sd_distil() {
  install_deepspeed_source
  # Teacher UNet is an unwrapped FP32 module. Run the entire recipe in FP32;
  # BF16 VAE output would otherwise meet FP32 teacher convolution weights.
  install_example_dependencies 'transformers==4.44.2' 'diffusers==0.30.3' \
    'accelerate==1.10.1' 'datasets>=4,<5' 'pillow>=10' 'torchvision==0.24.0' \
    safetensors sentencepiece numpy tensorboard
  download_sd15_ci_model
  python - <<'PY'
import importlib.util
import io
import json
import os
from pathlib import Path
import sys

from datasets import Dataset, Features, Image as DatasetImage, Value, load_dataset
from PIL import Image, ImageDraw
import torch_npu
from transformers import AutoTokenizer

output = Path(os.environ.get('CI_OUTPUT_DIR', str(Path(os.environ['GITHUB_WORKSPACE']) / 'output')))
work = output / 'sd-distil-work'
directory = work / 'poloclub/diffusiondb'
directory.mkdir(parents=True, exist_ok=True)
prompts, images = [], []
for index in range(8):
    image = Image.new('RGB', (64, 64), (16 * index, 64, 128))
    draw = ImageDraw.Draw(image)
    draw.rectangle((8 + index, 8, 48, 48), fill=(192, 32 * index, 64))
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    images.append({'bytes': buffer.getvalue(), 'path': None})
    prompts.append(f'a colorful geometric rectangle number {index}')
features = Features({'image': DatasetImage(), 'prompt': Value('string')})
Dataset.from_dict({'image': images, 'prompt': prompts}, features=features).to_parquet(directory / 'train.parquet')
(directory / 'README.md').write_text(
    '---\nconfigs:\n- config_name: 2m_first_10k\n  data_files:\n'
    '  - split: train\n    path: train.parquet\n---\nLocal eight-image CI fixture.\n'
)
previous = Path.cwd()
try:
    os.chdir(work)
    # Native local-directory resolution, not a datasets.load_dataset wrapper.
    dataset = load_dataset('poloclub/diffusiondb', '2m_first_10k')['train']
finally:
    os.chdir(previous)
if len(dataset) != 8 or any(row['image'].mode != 'RGB' or row['image'].size != (64, 64)
                           or not row['prompt'] for row in dataset):
    raise SystemExit('SD native local dataset must decode eight RGB 64x64 images/prompts')
entry = Path(os.environ['EXAMPLES_ROOT']) / 'training/stable_diffusion/train_sd_distil_lora.py'
sys.path.insert(0, str(entry.parent))
spec = importlib.util.spec_from_file_location('ds_ci_sd_distil', entry)
upstream = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = upstream
spec.loader.exec_module(upstream)
# The original dataset reads the original entry's args global, as its normal
# __main__ does. Populate it through the original parser, without changing APIs.
upstream.args = upstream.parse_args([
    '--pretrained_model_name_or_path', os.environ['SD15_PATH'],
    '--default_prompt', 'a colorful geometric shape', '--train_batch_size', '1',
    '--resolution', '64', '--max_train_steps', '3', '--mixed_precision', 'no', '--report_to', 'none',
])
tokenizer = AutoTokenizer.from_pretrained(os.environ['SD15_PATH'], subfolder='tokenizer', local_files_only=True)
native = upstream.DreamBoothDataset(dataset['prompt'], dataset['image'], tokenizer, size=64, center_crop=True)
batch = upstream.collate_fn([native[0]], with_prior_preservation=False)
if tuple(batch['pixel_values'].shape) != (1, 3, 64, 64) or batch['input_ids'].shape[0] != 1:
    raise SystemExit('SD original DreamBooth dataset/collator produced invalid CI tensors')
model_index = json.loads((Path(os.environ['SD15_PATH']) / 'model_index.json').read_text())
if model_index['_class_name'] != 'StableDiffusionPipeline':
    raise SystemExit('SD CI requires a complete StableDiffusionPipeline snapshot')
print('SD native fixture/collator verified:', directory.resolve(), len(native), tuple(batch['pixel_values'].shape))
print('SD teacher CFG distillation trains full UNet, not LoRA; dtype remains FP32')
PY
}

setup_ds_opsd_decode() {
  install_deepspeed_source
  install_example_dependencies 'transformers==4.57.6' 'accelerate>=1.10.1,<2' safetensors numpy
  ms_download_models 'QWEN3_06B_PATH=Qwen/Qwen3-0.6B'
  python - <<'PY'
import os
import torch
import torch_npu
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout, HybridEngineRolloutConfig
from transformers import AutoTokenizer

index = get_accelerator().current_device()
# Torch 2.9 maps integer devices to the registered accelerator (PrivateUse1
# first). Verify the exact tensor protocols the unchanged benchmark uses.
tensor = torch.empty(1, device='cpu').to(index)
random = torch.randint(10, 1000, (1, 2), device=index)
if tensor.device.type != 'npu' or random.device.type != 'npu':
    raise SystemExit(f'OPSD decode integer device did not select NPU: {tensor.device}, {random.device}')
tokenizer = AutoTokenizer.from_pretrained(os.environ['QWEN3_06B_PATH'], local_files_only=True)
print('OPSD decode integer-device preflight:', index, tensor.device, random.device, type(tokenizer).__name__)
print('OPSD rollout APIs:', HybridEngineRollout.__name__, HybridEngineRolloutConfig.__name__)
PY
}

# vLLM examples run in their own prebuilt image, not the training image.
# Preserve the image's torch/NPU/vLLM runtime; prepare a compatible data stack
# and select the tokenizer API required by each native client.
verify_vllm_runtime() {
  python - <<'PY'
from importlib.metadata import version
import torch
import torch_npu
from vllm.platforms import current_platform

for name, expected in {'vllm': '0.23.0', 'vllm-ascend': '0.23.0',
                       'torch': '2.10.0', 'torch-npu': '2.10.0.post4'}.items():
    actual = version(name).split('+')[0]
    print(f'vLLM image runtime: {name}={actual}')
    if actual != expected:
        raise SystemExit(f'{name}: expected {expected}, got {actual}; do not replace the image stack')
if current_platform.device_type != 'npu' or not torch.npu.is_available():
    raise SystemExit(f'vLLM must select available NPU, got {current_platform.device_type}')
if torch.ones(2, device='npu').sum().item() != 2:
    raise SystemExit('vLLM NPU allocation/reduction preflight failed')
print('vLLM Ascend platform and NPU preflight passed')
PY
}

prepare_vllm_image() {
  verify_vllm_runtime
  local constraint_dir="$GITHUB_WORKSPACE/.ci/deepspeed-vllm"
  mkdir -p "$constraint_dir"
  export PIP_CONSTRAINT="$constraint_dir/protected-serving-stack.txt"
  python - "$PIP_CONSTRAINT" "$PROFILE" "${2:-}" <<'PY'
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

protected = ('torch', 'torch-npu', 'vllm', 'vllm-ascend',
             'triton', 'triton-ascend', 'safetensors')
if not sys.argv[3]:
    protected += ('transformers', 'huggingface-hub', 'tokenizers')
lines = []
for name in protected:
    try:
        lines.append(f'{name}=={version(name)}\n')
    except PackageNotFoundError:
        pass
if sys.argv[3]:
    lines += [f'transformers=={sys.argv[3]}\n', 'huggingface-hub>=0.34.0,<1.0\n',
              'tokenizers>=0.22.0,<=0.23.0\n']
numpy_version = '2.2.6' if sys.argv[2] == 'ds_math500' else '1.26.4'
lines += [f'numpy=={numpy_version}\n', 'scipy==1.16.3\n',
          'pyarrow==20.0.0\n', 'pandas==2.2.3\n']
Path(sys.argv[1]).write_text(''.join(lines))
print('Serving runtime and compatible data constraints:\n' + ''.join(lines))
PY
  install_deepspeed_source
  # The original GSM8K/MMLU calls include trust_remote_code; use datasets 3.x
  # with native local-directory loading, without replacing load_dataset.
  install_example_dependencies numpy scipy pyarrow pandas transformers "${1:-datasets==3.6.0}"
  python -c 'import numpy, scipy, pandas, pyarrow; print("vLLM data stack:", numpy.__version__, scipy.__version__, pandas.__version__, pyarrow.__version__)'
  ms_download_models 'VLLM_CI_MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct'
  python - <<'PY'
import os
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(os.environ['VLLM_CI_MODEL_PATH'], local_files_only=True)
print('vLLM CI local tokenizer:', type(tokenizer).__name__)
PY
}

setup_ds_vllm_eval() {
  prepare_vllm_image
  export VLLM_CI_DATA_ROOT="$GITHUB_WORKSPACE/.ci/deepspeed-vllm/data"
  python - "$FIXTURE_DIR" "$VLLM_CI_DATA_ROOT" <<'PY'
import json
import os
from pathlib import Path
import sys
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
from datasets import Dataset, load_dataset

fixture_root, data_root = map(Path, sys.argv[1:])
previous = Path.cwd()
for repo, config, fixture, fields in (
    ('openai/gsm8k', 'main', 'ci_gsm8k_8.jsonl', {'question', 'answer'}),
    ('cais/mmlu', 'all', 'ci_mmlu_8.jsonl', {'question', 'choices', 'answer'}),
):
    rows = [json.loads(line) for line in (fixture_root / fixture).read_text().splitlines() if line.strip()]
    if len(rows) != 8 or any(set(row) != fields for row in rows):
        raise SystemExit(f'{fixture}: expected eight native-schema rows')
    directory = data_root / repo
    directory.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(directory / 'test.parquet')
    (directory / 'README.md').write_text(
        f'---\nconfigs:\n- config_name: {config}\n  data_files:\n'
        '  - split: test\n    path: test.parquet\n---\nLocal eight-question CI fixture.\n'
    )
    try:
        os.chdir(data_root)
        # Same repository/config/split call as the unchanged original entry.
        loaded = load_dataset(repo, config, split='test', trust_remote_code=True)
    finally:
        os.chdir(previous)
    if loaded.to_list() != rows:
        raise SystemExit(f'{repo}: native local-directory resolution changed CI data')
    print(f'Native local dataset verified: {repo}/{config}, {len(loaded)} rows')
with open(os.environ['GITHUB_ENV'], 'a') as output:
    output.write(f'VLLM_CI_DATA_ROOT={data_root.resolve()}\n')
    output.write('HF_HUB_OFFLINE=1\nHF_DATASETS_OFFLINE=1\n')
PY
}

setup_ds_vllm_traces() {
  prepare_vllm_image
  plant_ci_fixture ci_rac_8.jsonl RAC_CI_PATH
}

setup_ds_vllm_bench() {
  if [[ "${EXAMPLE_PATH:-}" == benchmarks/inference/mii/run_benchmark.py ]]; then
    # This unchanged client calls batch_encode_plus, removed in Transformers 5.
    prepare_vllm_image 'datasets==3.6.0' '4.57.6'
  else
    prepare_vllm_image
  fi
  install_example_dependencies loguru tabulate requests 'pydantic>=2,<3'
  local work="$GITHUB_WORKSPACE/.ci/deepspeed-vllm/benchmark"
  mkdir -p "$work"
  ln -s "$VLLM_CI_MODEL_PATH" "$work/ci-model"
  echo "VLLM_CI_BENCH_ROOT=$work" >> "$GITHUB_ENV"
}

setup_ds_aio() {
  install_deepspeed_source
  apt-get install -y libaio-dev build-essential
  python -c 'from deepspeed.ops.op_builder import AsyncIOBuilder; AsyncIOBuilder().load(verbose=True)'
}

setup_ds_math500() {
  prepare_vllm_image 'datasets>=4,<5'
  # This lighteval commit carries PR #1211 (tokenizer/TokensPrompt APIs for
  # vLLM >=0.19) plus PR #1402, which encodes strings before xxhash 4.x hashing.
  # Do not install its vllm extra or replace the image engine.
  install_example_dependencies \
    'lighteval[math] @ git+https://github.com/huggingface/lighteval.git@b634749d160b080071b9a8e3f5e5a7153dd130c0' \
    ray more_itertools
  local work="$GITHUB_WORKSPACE/.ci/deepspeed-math500"
  python - "$work" "$FIXTURE_DIR/ci_gsm8k_8.jsonl" <<'PY'
import json
from pathlib import Path
import sys

work, fixture = map(Path, sys.argv[1:])
(work / 'data').mkdir(parents=True, exist_ok=True)
rows = [json.loads(line) for line in fixture.read_text().splitlines()]
with (work / 'data/test.jsonl').open('w') as output:
    for row in rows:
        answer = row['answer'].split('####')[-1].strip()
        output.write(json.dumps({'problem': row['question'], 'solution': answer, 'answer': answer}) + '\n')
(work / 'task.py').write_text(
    'import os\nfrom dataclasses import replace\n'
    'from lighteval.tasks.tasks.math_500 import math_500\n'
    'TASKS_TABLE = [replace(math_500, name="ci_math500", '
    'hf_repo=os.environ["MATH500_DATA_PATH"], generation_size=16)]\n'
)
PY
  printf 'MATH500_CI_WORK=%s\nMATH500_DATA_PATH=%s/data\n' "$work" "$work" >> "$GITHUB_ENV"
}

setup_ds_autotune() {
  local model="$1" script="$2"
  install_deepspeed_source
  install_example_dependencies 'transformers==4.57.6' 'datasets==3.6.0' \
    'accelerate>=1.10.1,<2' 'evaluate==0.4.6' scikit-learn sentencepiece protobuf tabulate
  ms_download_models "AUTOTUNE_MODEL_PATH=$model"
  local work="$GITHUB_WORKSPACE/.ci/deepspeed-autotune"
  mkdir -p "$work"
  curl -fL "https://raw.githubusercontent.com/huggingface/transformers/v4.57.6/examples/pytorch/$script" \
    -o "$work/train.py"
  if [[ "$script" == text-classification/* ]]; then
    cp "$FIXTURE_DIR/ci_mnli_8.jsonl" "$work/train.json"
    mkdir -p "$work/accuracy"
    curl -fL https://raw.githubusercontent.com/huggingface/evaluate/v0.4.6/metrics/accuracy/accuracy.py \
      -o "$work/accuracy/accuracy.py"
  else
    cp "$FIXTURE_DIR/ci_rac_8.jsonl" "$work/train.json"
  fi
  # Autotuner's native scheduler uses localhost SSH/pdsh for experiment cleanup.
  # Keys and configuration belong only to this CI container, never the host.
  apt-get install -y pdsh openssh-server
  ssh-keygen -q -t ed25519 -N '' -f "$work/key"
  ssh-keygen -q -t ed25519 -N '' -f "$work/hostkey"
  mkdir -p /run/sshd
  /usr/sbin/sshd -f /dev/null -p 2222 -h "$work/hostkey" \
    -o ListenAddress=127.0.0.1 -o AuthorizedKeysFile="$work/key.pub" \
    -o PermitRootLogin=prohibit-password -o StrictModes=no -o PasswordAuthentication=no \
    -o UsePAM=yes -o KbdInteractiveAuthentication=no -o PidFile="$work/sshd.pid"
  # The launcher preflight and each tuner experiment call plain ssh, not pdsh.
  # Ubuntu's client config includes this container-local fragment for all three.
  mkdir -p /etc/ssh/ssh_config.d
  printf 'Host localhost 127.0.0.1\n  HostName 127.0.0.1\n  Port 2222\n  IdentityFile %s/key\n  IdentitiesOnly yes\n  StrictHostKeyChecking no\n  UserKnownHostsFile /dev/null\n' \
    "$work" > /etc/ssh/ssh_config.d/deepspeed-ci.conf
  ssh -o BatchMode=yes localhost true
  printf 'AUTOTUNE_WORK=%s\n' "$work" >> "$GITHUB_ENV"
}

setup_ds_autotune_bert() { setup_ds_autotune AI-ModelScope/bert-base-cased text-classification/run_glue.py; }
setup_ds_autotune_deberta() { setup_ds_autotune microsoft/deberta-v3-small text-classification/run_glue.py; }
setup_ds_autotune_distilbert() { setup_ds_autotune AI-ModelScope/distilbert-base-uncased text-classification/run_glue.py; }
setup_ds_autotune_gpt2() { setup_ds_autotune AI-ModelScope/gpt2 language-modeling/run_clm.py; }

supported_profiles() {
  declare -F | awk '/^declare -f setup_/ { sub(/^declare -f setup_/, ""); print }' | paste -sd' ' -
}

if ! declare -F "setup_${PROFILE}" >/dev/null 2>&1; then
  echo "unknown profile: ${PROFILE} (supported: $(supported_profiles))" >&2
  exit 1
fi

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
EXAMPLES_ROOT="${EXAMPLES_ROOT:-$TARGET_ROOT}"
FIXTURE_DIR="${FIXTURE_DIR:-$GITHUB_WORKSPACE/workflows/examples-guards/deepspeed/fixtures}"
GITHUB_WORKSPACE="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
GITHUB_ENV="${GITHUB_ENV:?GITHUB_ENV is required}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh

select_pip_index
# Do not run the training stack installer or its fixed scientific-stack
# checks for these independently scheduled vLLM image jobs.
case "$PROFILE" in
  ds_vllm_eval|ds_vllm_traces|ds_vllm_bench|ds_math500)
    "setup_${PROFILE}"
    verify_vllm_runtime
    verify_installed_runtime
    exit 0
    ;;
esac
python -m pip install -U pip setuptools wheel
ensure_torch_stack
python -m pip install numpy scipy pyarrow pandas
verify_data_stack

"setup_${PROFILE}"
verify_data_stack
verify_installed_runtime
