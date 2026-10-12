#!/usr/bin/env bash
# Prepare the CI environment for one supported ROLL example.
# $1 is the manifest profile. Unknown profiles fail before any install.
# The manifest uses a domestic CANN base image.  This script installs the
# version-matched torch_npu / vLLM-Ascend stack following the v0.3.0
# Ascend environment guide,
# then installs ROLL itself from TARGET_ROOT (the upstream checkout under
# test). ModelScope snapshots are private to the current CI job.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"
export CI_OUTPUT_DIR="${CI_OUTPUT_DIR:-${GITHUB_WORKSPACE:?}/output}"
mkdir -p "$CI_OUTPUT_DIR"
export PIP_CONSTRAINT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/constraints-npu.txt"

ASCEND_PIP_INDEX=https://repo.huaweicloud.com/ascend/repos/pypi
FALLBACK_PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
HUAWEICLOUD_PIP_INDEX=https://repo.huaweicloud.com/repository/pypi/simple
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

# Vendor CANN/ATB env scripts assume a login shell and reference optional
# variables (e.g. $ZSH_VERSION) without ${VAR:-} guards. Under this
# project's `set -u` they die with "unbound variable"; relax strict mode
# only while sourcing vendor code, then restore it.
source_vendor_env() {
  local vendor_file="$1"
  if [[ ! -f "$vendor_file" ]]; then
    echo "vendor env script not found, skipping: $vendor_file"
    return 0
  fi
  set +eu
  # shellcheck disable=SC1090
  source "$vendor_file"
  set -eu
}

# The engine no longer sources CANN for project jobs; bring it into this
# step and persist the runtime variables that ROLL's system_envs does not
# cover so the Run step inherits them.
prepare_ascend_env() {
  source_vendor_env /usr/local/Ascend/ascend-toolkit/set_env.sh
  source_vendor_env /usr/local/Ascend/nnal/atb/set_env.sh
  # Canonical ROLL NPU knobs (ascend_npu_env_config.md), tightened to smoke
  # timeouts. VLLM_ASCEND_ENABLE_NZ stays off for vLLM stability.
  export HCCL_NPU_SOCKET_PORT_RANGE="auto"
  export HCCL_CONNECT_TIMEOUT="600"
  export HCCL_EXEC_TIMEOUT="600"
  export HCCL_DETERMINISTIC="false"
  export HCCL_OP_EXPANSION_MODE="AIV"
  export HCCL_WHITELIST_DISABLE="1"
  export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
  export TASK_QUEUE_ENABLE="2"
  export COMBINED_ENABLE="1"
  export VLLM_ASCEND_ENABLE_NZ="0"
  export ACL_OP_COMPILER_CACHE_MODE="enable"
  export ACL_OP_COMPILER_CACHE_DIR="/tmp/npu_cache"
  export MODEL_DOWNLOAD_TYPE="MODELSCOPE"
  export USE_MODELSCOPE="1"
  for VAR in HCCL_NPU_SOCKET_PORT_RANGE HCCL_CONNECT_TIMEOUT HCCL_EXEC_TIMEOUT              HCCL_DETERMINISTIC HCCL_OP_EXPANSION_MODE HCCL_WHITELIST_DISABLE              PYTORCH_NPU_ALLOC_CONF TASK_QUEUE_ENABLE COMBINED_ENABLE              VLLM_ASCEND_ENABLE_NZ ACL_OP_COMPILER_CACHE_MODE              ACL_OP_COMPILER_CACHE_DIR MODEL_DOWNLOAD_TYPE USE_MODELSCOPE; do
    echo "${VAR}=${!VAR}" >> "$GITHUB_ENV"
  done
}

ensure_roll_installed() {
  local target_root="${TARGET_ROOT:?TARGET_ROOT is required}"
  # Both the current main (pyproject.toml without a pip build-system and a
  # legacy setup.py fallback) and older refs like v0.3.0 (setup.py only)
  # install cleanly via setuptools; --no-build-isolation keeps the in-image
  # setuptools path while --no-deps protects the pinned torch/npu stack.
  echo "=> installing ROLL from ${target_root} (editable, no deps, no isolation)"
  python -m pip install -q --no-deps --no-build-isolation -e "$target_root"
  python - "$target_root" <<'PY'
import os, sys, pathlib
import roll
target = pathlib.Path(sys.argv[1]).resolve()
source = pathlib.Path(roll.__path__[0]).resolve()
try:
    source.relative_to(target)
except ValueError:
    raise SystemExit(
        f"roll resolves to {source}, expected a path inside {target}; "
        "a stale image install has taken precedence over TARGET_ROOT"
    )
print(f"roll import path ok: {source}")
PY
}

install_npu_vllm_dependencies() {
  local requirements_file="$1"
  # --no-deps wheel installs do not provide packaging. Bootstrap the parser
  # in this interpreter before using it, even on a minimal CANN image.
  python -m pip install -q "packaging==25.0"
  python - <<'PY'
from importlib.metadata import version
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
assert canonicalize_name(Requirement('torch_npu>=2.10').name) == 'torch-npu'
print('vLLM metadata parser ready: packaging', version('packaging'))
PY
  # Read metadata without importing vLLM. Its PyPI wheel declares CUDA
  # kernels and torch 2.11; Ascend uses its own pinned torch/kernel stack.
  python - "$requirements_file" <<'PY'
from importlib.metadata import requires
from pathlib import Path
import sys
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

managed = {'torch', 'torch-npu', 'torchvision', 'torchaudio', 'triton',
           'triton-ascend', 'vllm', 'transformers'}
cuda_only = {'flashinfer-python', 'flashinfer-cubin', 'flash-attn',
             'humming-kernels', 'quack-kernels', 'tokenspeed-mla',
             'tilelang', 'fastsafetensors', 'helion'}
selected = []
for package in ('vllm', 'vllm-ascend'):
    for value in requires(package) or []:
        requirement = Requirement(value)
        if requirement.marker and not requirement.marker.evaluate({'extra': ''}):
            continue
        name = canonicalize_name(requirement.name)
        if name in managed or name in cuda_only or name.startswith('nvidia-'):
            print(f'NPU recipe omits {package} dependency: {requirement}')
            continue
        if name == 'opencv-python-headless':
            # Video inputs are not supported by this project's text recipes.
            requirement = Requirement('opencv-python-headless==4.11.0.86')
        elif name == 'mistral-common':
            # The image extra pulls NumPy-2-only OpenCV. Text tokenizers remain.
            requirement.extras.discard('image')
        selected.append(str(requirement))
selected += ['transformers==4.57.6', 'scipy==1.16.3',
             'opencv-python-headless==4.11.0.86']
Path(sys.argv[1]).write_text('\n'.join(sorted(set(selected))) + '\n', encoding='utf-8')
print('NPU vLLM dependency recipe:', sys.argv[1])
PY
  python -m pip install -q -r "$requirements_file"
}

install_rollout_stack() {
  local target_root="${TARGET_ROOT:?TARGET_ROOT is required}"
  local requirements_file
  # Keep the filtered file beside requirements_common.txt: pip resolves
  # its relative references from the file location.
  requirements_file="$target_root/.ci-requirements-common.txt"

  echo "=> installing matched torch 2.10 / torch_npu 2.10 runtime"
  python -m pip install -q \
    "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0" \
    "numpy==1.26.4" "scipy==1.16.3"
  pip_ascend -q --no-deps "torch-npu==2.10.0.post4"

  echo "=> installing vLLM 0.23.0 / vLLM-Ascend 0.23.0rc1 from domestic indexes"
  # Never resolve the CUDA dependency graph from these generic wheels.
  # Install their non-CUDA dependencies explicitly using the Ascend recipe.
  python -m pip install -q --no-deps --index-url "$HUAWEICLOUD_PIP_INDEX" \
    "vllm==0.23.0" "vllm-ascend==0.23.0rc1"
  install_npu_vllm_dependencies "$target_root/.ci-requirements-vllm-npu.txt"
  python -m pip uninstall -y -q triton triton-ascend || true
  python -m pip install -q --index-url "$HUAWEICLOUD_PIP_INDEX" \
    "triton==3.5.0"
  python -m pip install -q --no-deps --index-url "$ASCEND_PIP_INDEX" \
    "triton-ascend==3.2.1"

  # Generic vLLM metadata can select a CUDA torch build.  Restore the exact
  # NPU pair after installing vLLM, exactly as the v0.3.0 Ascend guide
  # requires.
  python -m pip install -q \
    "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0"
  pip_ascend -q --no-deps "torch-npu==2.10.0.post4"

  echo "=> installing ROLL common + Sokoban dependencies"
  # Text-only Sokoban rollout needs only requirements_common. Skip:
  #   - requirements_vision.txt: pycocotools / qwen_vl_utils / decord /
  #     rouge_score are video / detection-only; decord has no py3.12
  #     aarch64 wheel.
  #   - ./mcore_adapter / Megatron adapter: never exercises on the
  #     FSDP2/vLLM rollout path.
  grep -vE '^[[:space:]]*gem-llm' \
    "$target_root/requirements_common.txt" \
    | grep -vE '\./mcore_adapter|requirements_vision\.txt' \
    > "$requirements_file"
  (
    cd "$target_root"
    # A truncated index/wheel response can escape pip's own download retries.
    for attempt in 1 2 3; do
      echo "ROLL common dependency installation: attempt $attempt/3"
      if python -m pip install --progress-bar off -r "$requirements_file"; then
        exit 0
      else
        pip_status=$?
      fi
      if [[ "$attempt" == 3 ]]; then exit "$pip_status"; fi
      sleep 5
    done
  )
  # gem-llm metadata pulls an antlr runtime incompatible with Hydra's pin;
  # v0.3.0 requirements pin antlr4-python3-runtime 4.9.x, which gem works
  # with.
  (cd "$target_root" && python -m pip install -q \
    --ignore-requires-python --no-deps "gem-llm==0.0.4")
  # gem-llm 0.0.4 hard-requires reasoning-gym==0.1.23, which
  # --no-deps drops; the agentic env import chain breaks without it.
  python -m pip install -q \
    "numpy==1.26.4" "transformers==4.57.6" "tensorboard==2.20.0" \
    "antlr4-python3-runtime==4.9.3" "modelscope==1.37.0" \
    "reasoning-gym==0.1.23" "gym_sokoban" "gymnasium[toy-text]"

  python - <<'PY'
import tempfile
from pathlib import Path

import numpy, pandas, pyarrow, datasets, scipy
from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment
from datasets import Dataset, load_dataset
print('data stack:', 'numpy', numpy.__version__, 'pandas', pandas.__version__,
      'pyarrow', pyarrow.__version__, 'datasets', datasets.__version__,
      'scipy', scipy.__version__)
assert scipy.__version__ == '1.16.3'
assert csr_matrix(numpy.eye(2)).sum() == 2
rows, columns = linear_sum_assignment([[0, 1], [1, 0]])
assert rows.tolist() == columns.tolist() == [0, 1]
with tempfile.TemporaryDirectory(prefix='roll-data-probe-') as work:
    path = Path(work) / 'train.parquet'
    Dataset.from_list([{'text': 'ROLL NPU CI dependency probe'}]).to_parquet(str(path))
    loaded = load_dataset('parquet', data_files={'train': str(path)}, split='train')
    if loaded[0]['text'] != 'ROLL NPU CI dependency probe':
        raise SystemExit('ROLL data stack Parquet round-trip failed')

import torch, torch_npu, vllm, vllm_ascend, triton
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("vllm", vllm.__version__)
print("triton", triton.__version__)
print("NPU available", torch.npu.is_available(), "count", torch.npu.device_count())
from vllm.platforms import current_platform
from vllm_ascend.platform import NPUPlatform
if not isinstance(current_platform, NPUPlatform):
    raise SystemExit(f'vLLM did not select Ascend NPU: {type(current_platform)}')
from transformers import AutoModelForCausalLM, Trainer
print('vLLM NPU platform and Transformers imports passed')
PY
  rm -f "$requirements_file"
}

ms_download_model() {
  local model_id="${1:?model id required}"
  local path_variable="${2:-ROLL_MODEL_PATH}"
  local model_layout="${3:-transformers}"
  if python -c "import modelscope" 2>/dev/null; then
    echo "reusing image modelscope"
  else
    echo "=> installing modelscope"
    python -m pip install -q "modelscope==1.37.0"
  fi
  echo "=> downloading ${model_id} from ModelScope"
  MODEL_ID="$model_id" MODEL_PATH_VARIABLE="$path_variable" MODEL_LAYOUT="$model_layout" \
    TQDM_MININTERVAL="${TQDM_MININTERVAL:-15}" MODELSCOPE_DOWNLOAD_PARALLELS=1 \
    MODELSCOPE_ENABLE_DEFAULT_HASH_VALIDATION=true python - <<'PY'
import json
import os
from pathlib import Path
import tempfile
import time
from modelscope import snapshot_download
from safetensors import safe_open


def validate_snapshot(local):
    root = Path(local)
    components = [root]
    if os.environ['MODEL_LAYOUT'] == 'diffusers':
        for name in ('model_index.json', 'tokenizer/tokenizer_config.json',
                     'scheduler/scheduler_config.json'):
            json.loads((root / name).read_text())
        components = [root / name for name in ('transformer', 'text_encoder', 'vae')]
    for component in components:
        json.loads((component / 'config.json').read_text())
        shards = sorted(component.glob('*.safetensors'))
        for index in component.glob('*.safetensors.index.json'):
            expected = set(json.loads(index.read_text())['weight_map'].values())
            if not expected.issubset({shard.name for shard in shards}):
                raise RuntimeError(f'model weight shards missing: {component}')
        for shard in shards:
            with safe_open(str(shard), framework='np') as weights:
                if not list(weights.keys()):
                    raise RuntimeError(f'empty model weights: {shard}')
        if not shards and not list(component.glob('pytorch_model*.bin')):
            raise RuntimeError(f'model weights missing: {component}')
    print(f'validated model snapshot: {root}', flush=True)


model_id = os.environ["MODEL_ID"]
# Do not reuse or modify a potentially corrupt runner-wide snapshot. Ray
# workers in this job can read this private directory during the run step.
cache = tempfile.mkdtemp(prefix='roll-modelscope-', dir='/tmp')
print('private ModelScope cache:', cache, flush=True)
for attempt in range(3):
    try:
        local = snapshot_download(model_id, cache_dir=cache,
                                  enable_file_lock=True, max_workers=2)
        validate_snapshot(local)
        break
    except Exception as exc:
        print(f'{model_id} attempt {attempt + 1}/3 failed: {exc}', flush=True)
        if attempt == 2:
            raise
        cache = tempfile.mkdtemp(prefix='roll-modelscope-retry-', dir='/tmp')
        time.sleep(2 * (attempt + 1))
path_variable = os.environ["MODEL_PATH_VARIABLE"]
with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as handle:
    handle.write(f"{path_variable}={local}\n")
print(f"{path_variable}={local}")
PY
}

setup_wan_dmd() {
  local work="${GITHUB_WORKSPACE:?}/.ci/roll-wan"
  mkdir -p "$work"
  # VeOmni v0.1.11 uses transformers.initialization (5.x). Only the Wan
  # container changes that pin; all torch/NPU and data ABI constraints remain.
  python - "$PIP_CONSTRAINT" "$work/constraints.txt" <<'PY'
from pathlib import Path
import sys
text = Path(sys.argv[1]).read_text().replace('transformers==4.57.6', 'transformers==5.2.0')
Path(sys.argv[2]).write_text(text, encoding='utf-8')
PY
  export PIP_CONSTRAINT="$work/constraints.txt"
  python -m pip install -q 'torch==2.10.0' 'torchvision==0.25.0' 'torchaudio==2.10.0' \
    'numpy==1.26.4' 'scipy==1.16.3' 'pyarrow==20.0.0' 'pandas==2.2.3'
  pip_ascend -q --no-deps 'torch-npu==2.10.0.post4'
  # Wan uses VeOmni rather than vLLM. Keep common orchestration libraries,
  # replacing the text recipe's datasets/PEFT pins with the diffusion recipe.
  grep -vE '^[[:space:]]*(gem-llm|datasets|peft)|\./mcore_adapter|requirements_vision\.txt' \
    "${TARGET_ROOT:?}/requirements_common.txt" > "$work/common.txt"
  python -m pip install -q -r "$work/common.txt"
  python -m pip install -q 'transformers==5.2.0' 'peft==0.18.1' \
    'diffusers==0.37.0' 'datasets==2.21.0' 'torchdata>=0.8,<1' \
    'blobfile>=3' 'tiktoken>=0.9' 'einops>=0.8.1' 'packaging==25.0' psutil timm matplotlib ftfy \
    'modelscope==1.37.0' 'sentencepiece' 'tensorboard==2.20.0' 'safetensors>=0.5'
  python -m pip install -q --no-deps --no-build-isolation \
    'veomni @ git+https://github.com/ByteDance-Seed/VeOmni.git@f90b3dc6fbb0ce693745223cc7a94064123dbf4d'
  ensure_roll_installed
  ms_download_model 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers' ROLL_WAN_MODEL_PATH diffusers
  python - "$work" <<'PY'
import os
from pathlib import Path
import sys
import numpy as np
from datasets import Dataset
from veomni.arguments import OpsImplementationConfig
from veomni.models.auto import build_foundation_model
from veomni.distributed.torch_parallelize import build_parallelize_model
from roll.pipeline.diffusion.dmd.dmd_algorithm import latent_shape_from_vae_config
from diffusers import AutoencoderKLWan

work = Path(sys.argv[1])
values = dict(line.split('=', 1) for line in
              Path(os.environ['GITHUB_ENV']).read_text().splitlines()
              if line.startswith('ROLL_WAN_MODEL_PATH='))
vae_config = AutoencoderKLWan.load_config(values['ROLL_WAN_MODEL_PATH'], subfolder='vae')
# Diffusers fills the native VAE defaults in its constructor/config.
from types import SimpleNamespace
vae_config.setdefault('scale_factor_spatial', 8)
vae_config.setdefault('scale_factor_temporal', 4)
shape = latent_shape_from_vae_config(batch_size=1, num_frames=5, height=64, width=64,
                                   vae_config=SimpleNamespace(**vae_config))[1:]
prompts = ['A red ball moves on a plain white background.',
           'A blue cube slowly rotates on a table.'] * 4
prompt_path = work / 'prompts.txt'
prompt_path.write_text('\n'.join(prompts) + '\n', encoding='utf-8')
gan_path = work / 'real_latents.parquet'
rng = np.random.default_rng(42)
Dataset.from_list([{'text': prompt, 'latent': rng.standard_normal(shape).astype('float32').tolist()}
                   for prompt in prompts]).to_parquet(str(gan_path))
with open(os.environ['GITHUB_ENV'], 'a', encoding='utf-8') as handle:
    handle.write(f'ROLL_WAN_PROMPT_PATH={prompt_path}\nROLL_WAN_GAN_PATH={gan_path}\n')
print('Wan DMD native dependencies ready; GAN latent shape:', shape)
PY
}

# Only new profiles use these probes; existing 19 recipes and their assets stay unchanged.
verify_ci_assets() {
  ROLL_CI_PROFILE="$PROFILE" python - <<'PY'
import json
import os
from pathlib import Path

profile = os.environ['ROLL_CI_PROFILE']
if profile == 'agentic_gem_env_npu':
    import gem
    cases = [
        ('game:GuessTheNumber-v0', dict(min_number=1, max_number=3, max_turns=2)),
        ('rg:letter_counting', dict(size=8, seed=42)),
    ]
    for env_id, kwargs in cases:
        env = gem.make(env_id, **kwargs)
        observation, info = env.reset(seed=42)
        assert isinstance(observation, str) and observation
        env.step(r'\boxed{1}')
        print('GEM environment reset/step passed:', env_id)
else:
    from transformers import AutoTokenizer
    # GITHUB_ENV exports are visible in the next step, not this setup process.
    values = dict(line.split('=', 1) for line in
                  Path(os.environ['GITHUB_ENV']).read_text(encoding='utf-8').splitlines()
                  if line.startswith(('ROLL_MODEL_PATH=', 'ROLL_TEACHER_PATH=')))
    tokenizer = AutoTokenizer.from_pretrained(values['ROLL_MODEL_PATH'], local_files_only=True)
    fixture = Path(os.environ['FIXTURE_DIR']) / (
        'ci_opsd_8.jsonl' if profile == 'opd_npu' else 'ci_supervised_8.jsonl')
    rows = [json.loads(line) for line in fixture.read_text(encoding='utf-8').splitlines() if line.strip()]
    assert len(rows) == 8
    for row in rows:
        question = row.get('instruction', row.get('prompt'))
        answer = row.get('output', row.get('ground_truth'))
        assert question and answer
        ids = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}],
            tokenize=True, add_generation_prompt=False)
        assert 0 < len(ids) < 256, 'CI sample exceeds token budget'
        if profile == 'dpo_npu':
            assert row['chosen'] != row['rejected']
    if profile in ('distill_npu', 'opd_npu'):
        teacher = AutoTokenizer.from_pretrained(values['ROLL_TEACHER_PATH'], local_files_only=True)
        assert tokenizer.get_vocab() == teacher.get_vocab(), 'Student/teacher token IDs must match'
    print('CI fixture/tokenizer checks passed:', profile, len(rows), 'rows')
PY
}

prepare_judge_dataset() {
  python - <<'PY'
import json
import os
from pathlib import Path

source = Path(os.environ['FIXTURE_DIR']) / 'ci_math_8.jsonl'
path = Path(os.environ['CI_OUTPUT_DIR']) / 'judge.jsonl'
rows = [dict(json.loads(line), tag='RLVR') for line in source.read_text().splitlines()]
path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
with open(os.environ['GITHUB_ENV'], 'a', encoding='utf-8') as handle:
    handle.write(f'ROLL_JUDGE_PATH={path}\n')
PY
}

prepare_vlm_datasets() {
  # Native image datasets shared by RLVR, offline distillation and Geo3K.
  python - <<'PY'
import os
from pathlib import Path
from datasets import Dataset
from PIL import Image, ImageDraw

work = Path(os.environ['GITHUB_WORKSPACE']) / '.ci/roll-vlm'
work.mkdir(parents=True, exist_ok=True)
rows = []
for index, (left, right) in enumerate(((2, 3), (1, 4), (3, 4), (2, 6), (5, 1), (4, 4), (1, 2), (3, 6))):
    path = work / f'problem-{index}.png'
    image = Image.new('RGB', (112, 56), 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((4, 4, 48, 52), outline='black', width=2)
    draw.rectangle((64, 4, 108, 52), outline='black', width=2)
    draw.text((22, 20), str(left), fill='black')
    draw.text((82, 20), str(right), fill='black')
    image.save(path)
    question = 'Add the two numbers shown in the image. Return the sum.'
    answer = str(left + right)
    rows.append({'images': [str(path)], 'image': [str(path)],
                 'prompt': '<image>\n' + question, 'problem': question,
                 'solution': answer, 'answer': answer, 'data_source': 'mm_math',
                 'reward_model': {'ground_truth': answer}})
path = work / 'train.parquet'
Dataset.from_list(rows).to_parquet(str(path))
with open(os.environ['GITHUB_ENV'], 'a', encoding='utf-8') as handle:
    handle.write(f'ROLL_VLM_PATH={path}\n')
print('VLM fixture:', path, len(rows), 'image problems')
PY
}

prepare_gem_datasets() {
  python - <<'PY'
import json
import os
from pathlib import Path
from datasets import Dataset

math_path = (Path(os.environ['FIXTURE_DIR']) / 'ci_gem_math_8.jsonl').resolve()
rows = [json.loads(line) for line in math_path.read_text(encoding='utf-8').splitlines() if line.strip()]
qa_path = Path(os.environ['CI_OUTPUT_DIR']).resolve() / 'gem-qa'
qa_path.mkdir(parents=True, exist_ok=True)
qa_rows = [
    {
        'problem': f"Context: Student {index} solved task {index}. Task {index} asks: {row['problem']} "
                   f"The answer to task {index} is {row['answer']}. "
                   f"Question: What is the answer to the task solved by student {index}?",
        'answer': str(row['answer']),
    }
    for index, row in enumerate(rows, 1)
]
Dataset.from_list(qa_rows).to_parquet(str(qa_path / 'train.parquet'))
with open(os.environ['GITHUB_ENV'], 'a', encoding='utf-8') as handle:
    handle.write(f'ROLL_GEM_MATH_PATH={math_path}\nROLL_GEM_QA_PATH={qa_path}\n')
print(f'GEM datasets: math={math_path}, qa={qa_path}')
PY
}

check_npu_devices() {
  local minimum="$1"
  local devices
  devices=$(python - <<'PY'
import torch, torch_npu
print(int(torch.npu.device_count()))
PY
)
  echo "NPU available: $(python -c "import torch, torch_npu; print(torch.npu.is_available())"), devices: $devices"
  if [[ "$devices" -lt "$minimum" ]]; then
    echo "FATAL: profile $PROFILE needs at least $minimum NPUs, runner exposes $devices" >&2
    exit 1
  fi
}

prepare_ascend_env
select_pip_index
export PYTHONNOUSERSITE=1
if [[ "$PROFILE" == wan_dmd_npu ]]; then
  setup_wan_dmd
else
  install_rollout_stack
  ensure_roll_installed
  case "$PROFILE" in
    vlm_rlvr_npu|vlm_distill_npu|vlm_agentic_npu)
      # The native MM collator's load_images imports qwen_vl_utils.fetch_image;
      # the text recipe skips requirements_vision.txt, so install it for VLM.
      python -m pip install -q qwen_vl_utils
      ms_download_model "Qwen/Qwen2.5-VL-3B-Instruct"
      ;;
    *) ms_download_model "Qwen/Qwen2.5-0.5B-Instruct" ;;
  esac
fi

case "$PROFILE" in
  agentic_rollout_npu)
    echo "profile: agentic_rollout_npu (1 NPU: vLLM rollouts only)"
    check_npu_devices 1
    export ASCEND_RT_VISIBLE_DEVICES="0"
    ;;
  agentic_train_npu|opsd_lora_npu)
    echo "profile: agentic_train_npu (2 NPUs: FSDP2 train + vLLM rollout)"
    check_npu_devices 2
    export ASCEND_RT_VISIBLE_DEVICES="0,1"
    ;;
  agentic_async_npu|rlvr_judge_npu|vlm_rlvr_npu|vlm_distill_npu|vlm_agentic_npu|wan_dmd_npu)
    echo "profile: $PROFILE (4 NPUs)"
    check_npu_devices 4
    export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3"
    case "$PROFILE" in
      rlvr_judge_npu) prepare_judge_dataset ;;
      vlm_rlvr_npu|vlm_distill_npu|vlm_agentic_npu) prepare_vlm_datasets ;;
    esac
    ;;
  agentic_gem_env_npu)
    echo "profile: agentic_gem_env_npu (2 NPUs: native GEM env + FSDP2/vLLM)"
    check_npu_devices 2
    export ASCEND_RT_VISIBLE_DEVICES="0,1"
    case "${EXAMPLE_PATH:-}" in
      examples/qwen3_agentic_gem/gem_math_*) prepare_gem_datasets ;;
      *) verify_ci_assets ;;
    esac
    ;;
  sft_npu)
    echo "profile: sft_npu (1 NPU: FSDP2 supervised training)"
    check_npu_devices 1
    export ASCEND_RT_VISIBLE_DEVICES="0"
    verify_ci_assets
    ;;
  dpo_npu)
    echo "profile: dpo_npu (2 NPUs: FSDP2 actor + HF reference)"
    check_npu_devices 2
    export ASCEND_RT_VISIBLE_DEVICES="0,1"
    verify_ci_assets
    ;;
  distill_npu)
    echo "profile: distill_npu (2 NPUs: FSDP2 student + HF teacher, Ray logits transfer)"
    check_npu_devices 2
    export ASCEND_RT_VISIBLE_DEVICES="0,1"
    ms_download_model "Qwen/Qwen2.5-1.5B-Instruct" ROLL_TEACHER_PATH
    verify_ci_assets
    ;;
  opd_npu)
    echo "profile: opd_npu (4 NPUs: student train/rollout + independent teacher)"
    check_npu_devices 4
    export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3"
    ms_download_model "Qwen/Qwen2.5-1.5B-Instruct" ROLL_TEACHER_PATH
    if [[ "${EXAMPLE_PATH:-}" == examples/distill/on_policy/llm/multi_teacher_onpolicy_distill_config.yaml ]]; then
      echo "ROLL_MULTITEACHER_PATH=$(realpath "$FIXTURE_DIR/ci_multiteacher_8.jsonl")" >> "$GITHUB_ENV"
    fi
    verify_ci_assets
    ;;
  rlvr_npu)
    echo "profile: rlvr_npu (4 NPUs: FSDP2 train x2 + vLLM + reference)"
    check_npu_devices 4
    export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3"
    ;;
  *)
    echo "FATAL: unknown profile '$PROFILE'" >&2
    exit 2
    ;;
esac

echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}" >> "$GITHUB_ENV"

echo "setup complete for profile $PROFILE"
