#!/usr/bin/env bash
# Prepare the CI environment for one supported diffusers example.
# $1 is the manifest profile. Unknown profiles fail before any install.
#
# diffusers itself is installed from TARGET_ROOT (the release checkout
# under test), so the guarded tag is exactly the code that runs.
#
# Contract: examples-guards/README.md "项目数据" (scripts/setup_example.sh).
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"

# Validate the profile before installing anything (contract: unknown
# profile must exit non-zero before any install). This array is the single
# profile registry; a new profile also needs a PROFILE_* table entry or a
# bespoke setup_<profile>() function (see the Profiles section).
SUPPORTED_PROFILES=(
  diffusers-sdxl diffusers-sd15 diffusers-dreambooth diffusers-instruct-pix2pix
  diffusers-kandinsky diffusers-research diffusers-research-plain diffusers-t2i-adapter
  diffusers-text-to-image diffusers-textual-inversion diffusers-unconditional
  diffusers-vqgan diffusers-sdxl-online diffusers-flux diffusers-sana diffusers-lumina2
  diffusers-z-image diffusers-qwen-image diffusers-amused diffusers-cogvideo
  diffusers-cogvideo-i2v diffusers-lcm diffusers-lcm-sdxl diffusers-controlnet
  diffusers-controlnet-sdxl diffusers-llada2 diffusers-gligen diffusers-controlnet-wds
  diffusers-orpo-wds diffusers-sdxl-full diffusers-lcm-zero3
  diffusers-kandinsky-prior diffusers-multi-subject-inpaint diffusers-pixart
  diffusers-realfill diffusers-flux2-klein
)
profile_known=0
for _p in "${SUPPORTED_PROFILES[@]}"; do
  if [[ "$_p" == "$PROFILE" ]]; then
    profile_known=1
  fi
done
if [[ "$profile_known" != 1 ]]; then
  echo "unknown profile: ${PROFILE} (supported: ${SUPPORTED_PROFILES[*]})" >&2
  exit 1
fi

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
  # Same torch line as diffusers quick-start (CANN 9.1.0 pairing):
  # reuse the image stack when it already matches, otherwise install.
  if python -c "
import torch, torch_npu
raise SystemExit(
    0 if torch.__version__.startswith('2.9.0')
    and torch_npu.__version__.startswith('2.9.0') else 1)
"; then
    echo "reusing image torch stack ($(python -c 'import torch; print(torch.__version__)'))"
    return
  fi
  echo "installing torch==2.9.0 torch_npu==2.9.0.post2"
  pip_ascend torch==2.9.0 torch_npu==2.9.0.post2
}

prepare_fixtures() {
  # Optional: copy project fixtures into the target root so examples can
  # read them under $TARGET_ROOT/fixtures/. No-op when FIXTURE_DIR is
  # absent or empty (diffusers examples currently pull their datasets
  # from ModelScope / HuggingFace during setup instead).
  local src="${FIXTURE_DIR:-}"
  local dst="$TARGET_ROOT/fixtures"
  if [[ -z "$src" || ! -d "$src" ]]; then
    echo "no fixtures to prepare ($src)"
    return
  fi
  shopt -s nullglob
  local files=("$src"/*)
  shopt -u nullglob
  if ((${#files[@]} == 0)); then
    echo "no fixtures to prepare (empty $src)"
    return
  fi
  echo "preparing fixtures from $src to $dst"
  mkdir -p "$dst"
  cp -r "${files[@]}" "$dst/"
}

# ---------------------------------------------------------------
# Shared steps
# ---------------------------------------------------------------

install_example_stack() {
  echo "installing diffusers from $TARGET_ROOT"
  python -m pip install -e "$TARGET_ROOT"
  # Example stack. transformers 5.x is required by diffusers 0.40+
  # (huggingface-hub>=1.23 vs transformers 4.x's hub<1.0 conflict);
  # modelscope pinned to 1.37.0 (hub split started at 1.38 and 1.40.1's
  # "modelscope-hub>=0.4.2" floor dies on DEFAULT_CREDENTIALS_PATH);
  # torchvision pinned to the torch 2.9.0 pairing; prodigyopt is imported
  # by the advanced scripts' --optimizer=prodigy and is not in
  # examples/advanced_diffusion_training/requirements.txt.
  # datasets<4.0.0: diffusers examples (e.g. controlnet's load_dataset path)
  # break on datasets 4.x; the whole project stays below 4.
  # wandb: every training script does `if is_wandb_available(): import wandb`
  # and several default --report_to to wandb; installed once here, kept
  # offline by the global WANDB_MODE=disabled.
  python -m pip install \
    "transformers>=5.0,<6.0" "accelerate>=1.0,<2.0" "peft>=0.6" \
    "datasets<4.0.0" ftfy tensorboard Jinja2 sentencepiece "torchvision==0.24.0" \
    prodigyopt "modelscope==1.37.0" wandb
  python -c "import diffusers, transformers, accelerate, peft; print('diffusers', diffusers.__version__, '/ transformers',\
    transformers.__version__, '/ accelerate', accelerate.__version__, '/ peft', peft.__version__)"
}

# download_assets <what>: comma-separated tokens from
# {sdxl, sdxl-vae, sd15, sd15-inpaint, 3d-icon, cogvideo, sana-sprint}. Each token
# exports a path to GITHUB_ENV under the name overlay_args reference:
#   sdxl         -> SDXL_BASE_PATH      (AI-ModelScope/stable-diffusion-xl-base-1.0)
#   sdxl-vae     -> SDXL_VAE_PATH       (AI-ModelScope/sdxl-vae-fp16-fix)
#   sd15         -> SD15_MODEL_PATH     (AI-ModelScope/stable-diffusion-v1-5)
#   sd15-inpaint -> SD15_INPAINT_PATH   (AI-ModelScope/stable-diffusion-inpainting)
#   3d-icon      -> THREE_D_ICON_PATH   (HF linoyts/3d_icon)
#   cogvideo     -> COGVIDEOX_MODEL_PATH + COGVIDEOX_DATASET_DIR
download_assets() {
  DIFFUSERS_DOWNLOAD="$1" python - <<'PY'
import os
import sys
from pathlib import Path

os.environ.setdefault("TQDM_MININTERVAL", "15")

from modelscope import snapshot_download

WORKSPACE = Path(os.environ["GITHUB_WORKSPACE"])
ENV_FILE = os.environ["GITHUB_ENV"]
MODEL_CACHE = Path(os.environ.get("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope")))
MODEL_CACHE.mkdir(parents=True, exist_ok=True)
WANT = {item.strip() for item in os.environ.get("DIFFUSERS_DOWNLOAD", "").split(",") if item.strip()}

# ModelScope's snapshot_download gives up on a single file once its own
# retries are exhausted (e.g. "1 file(s) failed to download out of 16"),
# which fails the whole setup. Re-run it a few times: each pass resumes /
# re-fetches only what is still missing.
DOWNLOAD_ATTEMPTS = 3

exports: dict[str, str] = {}
failures: list[str] = []


def export(name: str, path: str) -> None:
    exports[name] = path
    print(f"{name}={path}", flush=True)


def corrupt_weights(root: Path) -> list[Path]:
    # snapshot_download trusts files that already exist, so a truncated one is
    # never re-fetched on its own: detect and delete the bad files here so the
    # retry loop re-downloads them.
    #   * .safetensors -> safe_open fails on a half-written header
    #     ("incomplete metadata, file not fully covered");
    #   * .bin -> diffusers' .bin checkpoints are torch zip archives, so a
    #     truncated file is no longer a readable zip (stable-diffusion-inpainting
    #     ships .bin only, and a partial download broke `from_pretrained`).
    import zipfile

    from safetensors import safe_open

    bad: list[Path] = []
    for path in sorted(root.rglob("*.safetensors")):
        try:
            with safe_open(path, framework="pt") as handle:
                handle.keys()
        except Exception as exc:  # noqa: BLE001
            print(f"corrupt {path}: {exc}", flush=True)
            bad.append(path)
    for path in sorted(root.rglob("*.bin")):
        if not zipfile.is_zipfile(path):
            print(f"corrupt {path}: not a readable zip archive (truncated?)", flush=True)
            bad.append(path)
    return bad


def snapshot(name: str, ms_id: str, **kwargs) -> None:
    last: Exception | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            local = Path(snapshot_download(ms_id, cache_dir=str(MODEL_CACHE), **kwargs))
            bad = corrupt_weights(local)
            if bad:
                for path in bad:
                    path.unlink()
                raise RuntimeError(f"{len(bad)} corrupt weight file(s) removed; re-downloading")
            export(name, str(local))
            return
        except Exception as exc:  # noqa: BLE001 - retry, then report
            last = exc
            print(f"retry {attempt}/{DOWNLOAD_ATTEMPTS} {ms_id}: {type(exc).__name__}: {exc}", flush=True)
    failures.append(f"{ms_id}: {type(last).__name__}: {last}")
    print(f"FAIL {ms_id}: {last}", flush=True)


def hf_snapshot(repo_id: str, **kwargs) -> str:
    from huggingface_hub import snapshot_download as hf_snapshot_download

    last: Exception | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            return hf_snapshot_download(repo_id, **kwargs)
        except Exception as exc:  # noqa: BLE001 - retry, then report
            last = exc
            print(f"retry {attempt}/{DOWNLOAD_ATTEMPTS} {repo_id}: {type(exc).__name__}: {exc}", flush=True)
    raise last


# SDXL base components only. The repo also carries sd_xl_base_1.0.safetensors
# (6.9 GB, ComfyUI single-file) and fp16 duplicates (~11 GB) that
# from_pretrained does not need; the script loads the fp32 component weights
# and casts to bf16 itself.
if "sdxl" in WANT:
    snapshot(
        "SDXL_BASE_PATH",
        "AI-ModelScope/stable-diffusion-xl-base-1.0",
        allow_file_pattern=[
            "*.json", "*.txt", "*.model",
            "unet/*", "vae/*",
            "text_encoder/*", "text_encoder_2/*",
            "tokenizer/*", "tokenizer_2/*", "scheduler/*",
        ],
        # .bin is the duplicate of .safetensors; skip both fp16 and .bin to
        # halve the download (and the .bin download is what fails).
        ignore_file_pattern=["*.fp16.*", "*.bin"],
    )

# SDXL fp16-safe VAE. Only entries that pass
# --pretrained_vae_model_name_or_path need it (the advanced SDXL entry does
# not; the LCM SDXL distillation does).
if "sdxl-vae" in WANT:
    snapshot("SDXL_VAE_PATH", "AI-ModelScope/sdxl-vae-fp16-fix")

# SD 1.5 diffusers components only; skip the single-file checkpoints
# (v1-5-pruned*.safetensors, ~4 GB each). Also the LCM SD teacher.
if "sd15" in WANT:
    snapshot(
        "SD15_MODEL_PATH",
        "AI-ModelScope/stable-diffusion-v1-5",
        allow_file_pattern=[
            "*.json", "*.txt", "*.model",
            "unet/*", "vae/*",
            "text_encoder/*", "tokenizer/*", "scheduler/*",
        ],
        # Same as SDXL: the repo carries both .bin and .safetensors; only the
        # latter is needed.
        ignore_file_pattern=["*.fp16.*", "*.bin"],
    )

# SD 1.5 inpainting (9-channel unet: 4 latent + 4 masked-latent + 1 mask).
# Only entries that train an inpainting unet need it. Same component-only
# filter as SD1.5, plus safety_checker: realfill's final
# StableDiffusionInpaintPipeline.from_pretrained only overrides unet/text_encoder
# (no safety_checker=None), so that component must be on disk too.
if "sd15-inpaint" in WANT:
    snapshot(
        "SD15_INPAINT_PATH",
        "AI-ModelScope/stable-diffusion-inpainting",
        allow_file_pattern=[
            "*.json", "*.txt", "*.model",
            "unet/*", "vae/*",
            "text_encoder/*", "tokenizer/*",
            "scheduler/*", "feature_extractor/*", "safety_checker/*",
        ],
        # This repo ships only .bin + *.fp16.safetensors (no full-precision
        # safetensors), so keep .bin and drop only the fp16 duplicates.
        ignore_file_pattern=["*.fp16.*"],
    )

# SD 1.4 (the dreambooth examples' README base model). Same component-only
# filter as SD1.5.
if "sd14" in WANT:
    snapshot(
        "SD14_MODEL_PATH",
        "AI-ModelScope/stable-diffusion-v1-4",
        allow_file_pattern=[
            "*.json", "*.txt", "*.model",
            "unet/*", "vae/*",
            "text_encoder/*", "tokenizer/*", "scheduler/*",
        ],
        ignore_file_pattern=["*.fp16.*", "*.bin"],
    )

# CogVideoX-2b (transformer + T5 text_encoder + VAE). The ModelScope repo
# layout is already clean (component dirs only, no single-file/fp16 dupes).
# Dataset: Wild-Heart/Disney-VideoGeneration-Dataset (69 videos, ~25 MB,
# Steamboat Willie clips) already in the README's first format:
# prompt.txt + videos.txt + videos/. No ModelScope mirror, so pull via
# HuggingFace directly (docs runners reach huggingface.co; the engine
# leaves HF_ENDPOINT / Xet at their defaults, so Xet-backed repos
# download natively).
def download_disney_dataset() -> None:
    try:
        dataset_dir = WORKSPACE / "datasets" / "disney"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        hf_snapshot(
            "Wild-Heart/Disney-VideoGeneration-Dataset",
            repo_type="dataset",
            local_dir=str(dataset_dir),
        )
        export("COGVIDEOX_DATASET_DIR", str(dataset_dir))
    except Exception as exc:  # noqa: BLE001
        failures.append(f"Wild-Heart/Disney-VideoGeneration-Dataset: {type(exc).__name__}: {exc}")
        print(f"FAIL Wild-Heart/Disney-VideoGeneration-Dataset: {exc}", flush=True)


# cogvideo: CogVideoX-2b (ModelScope) + the Disney dataset.
if "cogvideo" in WANT:
    snapshot(
        "COGVIDEOX_MODEL_PATH",
        "AI-ModelScope/CogVideoX-2b",
        allow_file_pattern=[
            "*.json", "*.model",
            "scheduler/*", "text_encoder/*", "tokenizer/*",
            "transformer/*", "vae/*",
        ],
    )
    download_disney_dataset()

# cogvideo-dataset: the Disney dataset only (the I2V entry pulls its own
# CogVideoX-5b-I2V from HuggingFace at run time, so no ModelScope pre-download).
if "cogvideo-dataset" in WANT:
    download_disney_dataset()

# 3d_icon dataset (only the advanced dreambooth entries use it): no
# ModelScope mirror (205 MB imagefolder + metadata.jsonl), so pull it from
# HuggingFace into the workspace and point --dataset_name at the local directory.
if "3d-icon" in WANT:
    try:
        dataset_dir = WORKSPACE / "datasets" / "3d_icon"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        hf_snapshot(
            "linoyts/3d_icon",
            repo_type="dataset",
            local_dir=str(dataset_dir),
            ignore_patterns=[".gitattributes"],
        )
        export("THREE_D_ICON_PATH", str(dataset_dir))
    except Exception as exc:  # noqa: BLE001
        failures.append(f"linoyts/3d_icon: {type(exc).__name__}: {exc}")
        print(f"FAIL linoyts/3d_icon: {exc}", flush=True)

# sana-sprint: the SANA-Sprint teacher model for research_projects/sana. The
# script reads <path>/transformer/diffusion_pytorch_model.safetensors directly
# (load_file, not from_pretrained), so --pretrained_model_name_or_path must be
# a LOCAL directory; pre-download it here.
if "sana-sprint" in WANT:
    try:
        dest = WORKSPACE / "sana_sprint_teacher"
        dest.mkdir(parents=True, exist_ok=True)
        hf_snapshot(
            "Efficient-Large-Model/SANA_Sprint_1.6B_1024px_teacher_diffusers",
            local_dir=str(dest),
        )
        export("SANA_SPRINT_TEACHER_PATH", str(dest))
    except Exception as exc:  # noqa: BLE001
        failures.append(f"SANA_Sprint_1.6B: {type(exc).__name__}: {exc}")
        print(f"FAIL SANA_Sprint_1.6B: {exc}", flush=True)

if exports:
    with open(ENV_FILE, "a", encoding="utf-8") as handle:
        for key, value in exports.items():
            handle.write(f"{key}={value}\n")

if failures:
    print(f"setup incomplete: {failures}", file=sys.stderr, flush=True)
    raise SystemExit(1)
PY
}

# ---------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------
#
# install_example_stack() is the common stack; a profile declares only its
# delta on top of it in the tables below, and profiles absent from all
# three tables get the base stack alone. When adding a new example, diff
# its requirements.txt against the base, add the profile to the
# SUPPORTED_PROFILES registry at the top, and extend the tables (or add a
# setup_<profile>() function for genuinely bespoke steps); never
# `pip install -r` it verbatim (it pins to git main / loose versions and
# pulls CUDA-only deps).

# deepspeed serves the accelerate-deepspeed (ZeRO-3) launcher; decord2 is the
# Ascend/aarch64 build of decord (plain decord has no NPU wheel).
declare -A PROFILE_PIP=(
  [diffusers-text-to-image]="deepspeed>=0.18.2"
  [diffusers-flux]="deepspeed>=0.18.2 webdataset braceexpand"
  [diffusers-vqgan]="timm"
  [diffusers-lcm]="webdataset braceexpand"
  [diffusers-lcm-sdxl]="webdataset braceexpand"
  [diffusers-cogvideo]="decord2 imageio imageio-ffmpeg"
  [diffusers-cogvideo-i2v]="decord2 imageio imageio-ffmpeg"
  [diffusers-lumina2]="deepspeed>=0.18.2"
  [diffusers-z-image]="deepspeed>=0.18.2"
  [diffusers-qwen-image]="deepspeed>=0.18.2"
  [diffusers-sdxl-full]="deepspeed>=0.18.2"
  [diffusers-lcm-zero3]="deepspeed>=0.18.2 webdataset braceexpand"
  [diffusers-controlnet-sdxl]="deepspeed>=0.18.2"
  [diffusers-kandinsky-prior]="deepspeed>=0.18.2"
  # controlnet-wds runs with launcher=accelerate-deepspeed (ZeRO-2), hence
  # deepspeed; the rest mirrors its vendored-webdataset requirements.
  [diffusers-controlnet-wds]="webdataset==0.2.86 braceexpand opencv-python-headless deepspeed>=0.18.2"
  [diffusers-orpo-wds]="webdataset braceexpand"
  # hf_transfer: parallel hub downloads for the pre-downloaded models.
  [diffusers-pixart]="hf_transfer"
  [diffusers-flux2-klein]="hf_transfer"
)

# download_assets tokens (comma-separated) for the profile's models/datasets.
declare -A PROFILE_ASSETS=(
  [diffusers-sd15]="sd15,3d-icon"
  [diffusers-dreambooth]="sd15"
  [diffusers-instruct-pix2pix]="sd15"
  [diffusers-research]="sd15"
  [diffusers-text-to-image]="sd15"
  [diffusers-textual-inversion]="sd15"
  [diffusers-controlnet]="sd15"
  [diffusers-lcm]="sd15"
  [diffusers-cogvideo]="cogvideo"
  [diffusers-cogvideo-i2v]="cogvideo-dataset"
  [diffusers-sana]="sana-sprint"
  [diffusers-multi-subject-inpaint]="sd15-inpaint"
  [diffusers-realfill]="sd15-inpaint"
)

# fusing/fill50k is a script-based dataset, so the controlnet examples need it.
declare -A PROFILE_ENV=(
  [diffusers-controlnet]="HF_DATASETS_TRUST_REMOTE_CODE=1"
  [diffusers-controlnet-sdxl]="HF_DATASETS_TRUST_REMOTE_CODE=1"
)

# ---------------------------------------------------------------
# Shared helper for bespoke profiles: pre-download a hub repo into
# <workspace>/<dest_subdir> and export the local path as <env_var> for the
# run step (local paths also avoid from_pretrained's model_info round-trip).
# allow/ignore are comma-separated glob lists; "-" means "not used".
# Goes straight to huggingface.co (the engine leaves HF_ENDPOINT
# unset); speeds up with hf_transfer when the
# profile pulled it in via PROFILE_PIP (run_profile sets the switch).
# ---------------------------------------------------------------
hub_snapshot() {
  local repo="$1" dest="$2" env_var="$3" allow="$4" ignore="$5"
  HUB_SNAPSHOT_REPO="$repo" HUB_SNAPSHOT_DEST="$dest" HUB_SNAPSHOT_VAR="$env_var" \
    HUB_SNAPSHOT_ALLOW="$allow" HUB_SNAPSHOT_IGNORE="$ignore" python3 - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download

repo = os.environ["HUB_SNAPSHOT_REPO"]
dest = Path(os.environ["GITHUB_WORKSPACE"]) / os.environ["HUB_SNAPSHOT_DEST"]
dest.mkdir(parents=True, exist_ok=True)
kwargs = {"local_dir": str(dest)}
allow = os.environ.get("HUB_SNAPSHOT_ALLOW", "-")
ignore = os.environ.get("HUB_SNAPSHOT_IGNORE", "-")
if allow != "-":
    kwargs["allow_patterns"] = [p for p in allow.split(",") if p]
if ignore != "-":
    kwargs["ignore_patterns"] = [p for p in ignore.split(",") if p]
snapshot_download(repo, **kwargs)
with open(os.environ["GITHUB_ENV"], "a") as fh:
    fh.write(f"{os.environ['HUB_SNAPSHOT_VAR']}={dest}\n")
print(f"pre-downloaded {repo} -> {dest}", flush=True)
PY
}

# diffusers-gligen: research_projects/gligen text-box training. Base stack only:
# dataset.py needs just torch/PIL/torchvision, and make_datasets.py's
# GroundingDINO + BLIP2 + CLIP pipeline is not used. The model
# (masterful/gligen-1-4-generation-text-box) is fetched online from
# HuggingFace by
# the example itself, so only the dataset is synthesized here: a COCO-style
# torch.load .pth (list of {file_path, captions, annos}) plus the image dir it
# references. Each anno is {bbox: xyxy pixels, text_embeddings_before_projection:
# 768-d}; that embedding is a constant conditioning input (never optimized), so
# zeros are fine. A 512x512 image makes --resolution 512 a no-op.
setup_diffusers_gligen() {
  python3 - <<'PY'
import os

import torch
from PIL import Image

root = os.path.join(os.environ["GITHUB_WORKSPACE"], "areal_data", "gligen")
image_dir = os.path.join(root, "images")
os.makedirs(image_dir, exist_ok=True)

samples = []
for i in range(4):
    file_name = f"sample{i}.png"
    Image.new("RGB", (512, 512), (110 + i * 20, 130, 150)).save(os.path.join(image_dir, file_name))
    samples.append(
        {
            "file_path": file_name,
            "captions": ["a gray square on a plain background"],
            "annos": [
                {
                    "bbox": [100.0, 100.0, 400.0, 400.0],
                    "text_embeddings_before_projection": torch.zeros(768),
                }
            ],
        }
    )

data_path = os.path.join(root, "gligen_train.pth")
torch.save(samples, data_path)

with open(os.environ["GITHUB_ENV"], "a") as fh:
    fh.write(f"GLIGEN_DATA_PTH={data_path}\n")
    fh.write(f"GLIGEN_IMAGE_DIR={image_dir}\n")
print(f"prepared gligen dataset at {data_path} (images in {image_dir})", flush=True)
PY
}

# controlnet-webdataset: webdataset pinned to 0.2.86 (the example's vendored
# tar iterators predate the 0.2.96 shard-end separator and KeyError on it);
# its dataloader hardcodes a 1000-sample shuffle buffer, so the shard must
# carry >1000 filter-passing samples - synthesized here because the upstream
# laion URL is unreachable. Each sample: jpg+png (image and control_image
# rename from image extensions), txt, and a json with original_width/height
# >= 512 + pwatermark for WebdatasetFilter. NB pwatermark must be truthy AND
# <= 0.5: the filter computes `(x or 1.0)`, so 0.0 evaluates to 1.0 and the
# sample is rejected - a rejected sample never enters the shuffle buffer and
# the run hangs silently (pwatermark=0.0 here cost a 55-min CI hang; 0.1
# passes). opencv-headless for the canny control image.
setup_diffusers_controlnet_wds() {
  python3 - <<'PY'
import io
import json
import os
import tarfile

from PIL import Image, ImageDraw

num_samples = 1024
size = 512
root = os.path.join(os.environ["GITHUB_WORKSPACE"], "controlnet_wds")
os.makedirs(root, exist_ok=True)
tar_path = os.path.join(root, "train-00000.tar")
meta = json.dumps({"original_width": size, "original_height": size, "pwatermark": 0.1}).encode()


def make_image(i):
    image = Image.new("RGB", (size, size), (60 + i % 180, 120, 160))
    draw = ImageDraw.Draw(image)
    draw.rectangle([64, 64, size - 64, size - 64], outline=(240, 240, 240), width=6)
    draw.line([0, i % size, size, (i * 7) % size], fill=(250, 200, 100), width=4)
    return image


def add(tf, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


with tarfile.open(tar_path, "w") as tf:
    for i in range(num_samples):
        key = f"{i:06d}"
        image = make_image(i)
        for ext, fmt in (("jpg", "JPEG"), ("png", "PNG")):
            buf = io.BytesIO()
            image.save(buf, format=fmt)
            add(tf, f"{key}.{ext}", buf.getvalue())
        add(tf, f"{key}.txt", f"a synthetic line drawing scene {i}".encode())
        add(tf, f"{key}.json", meta)

with open(os.environ["GITHUB_ENV"], "a") as fh:
    fh.write(f"CONTROLNET_WDS_SHARDS={tar_path}\n")
print(f"prepared {num_samples}-sample webdataset shard at {tar_path}", flush=True)
PY
}

# orpo-webdataset: the upstream preference dataset (diffusion-preference-opt on
# S3) went private, so the shard is synthesized here. Sample layout mirrors
# get_dataset()'s rename: {key}.original_prompt.txt / jpg_0.jpg / jpg_1.jpg /
# label_0.txt / label_1.txt (base_plus_ext groups by the first dot). The
# dataloader's shuffle buffer is 690, so the shard carries 1024 samples.
# Unlike controlnet-webdataset this uses the stock wds pipeline - no version
# pin needed. label txt values decode to str, so the == 0.5/== 0 comparisons
# never fire and the pair order is deterministically [jpg_0, jpg_1].
setup_diffusers_orpo_wds() {
  python3 - <<'PY'
import io
import os
import tarfile

from PIL import Image, ImageDraw

num_samples = 1024
size = 512
root = os.path.join(os.environ["GITHUB_WORKSPACE"], "orpo_wds")
os.makedirs(root, exist_ok=True)
tar_path = os.path.join(root, "train-00000.tar")


def make_image(i):
    image = Image.new("RGB", (size, size), (50 + i % 200, 110, 170))
    draw = ImageDraw.Draw(image)
    draw.ellipse([64 + i % 64, 64, size - 64, size - 64 - i % 64], outline=(240, 240, 240), width=6)
    return image


def add(tf, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


with tarfile.open(tar_path, "w") as tf:
    for i in range(num_samples):
        key = f"{i:06d}"
        for name, image in (("jpg_0", make_image(i)), ("jpg_1", make_image(i + 256))):
            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            add(tf, f"{key}.{name}.jpg", buf.getvalue())
        add(tf, f"{key}.original_prompt.txt", f"a synthetic ellipse scene {i}".encode())
        add(tf, f"{key}.label_0.txt", b"1.0")
        add(tf, f"{key}.label_1.txt", b"0.0")

with open(os.environ["GITHUB_ENV"], "a") as fh:
    fh.write(f"ORPO_WDS_SHARDS={tar_path}\n")
print(f"prepared {num_samples}-sample orpo webdataset shard at {tar_path}", flush=True)
PY
}

# sana (sprint distill): teacher from ModelScope/HuggingFace, plus a local parquet
# fixture. The script's __getitem__ does Image.open(io.BytesIO(item["image"])),
# and the upstream dataset's image column is auto-decoded to PIL by datasets
# (TypeError). A hand-built parquet keeps "image" as plain binary so datasets
# returns bytes. Columns must be exactly llava (str) + image (bytes).
setup_diffusers_sana() {
  python3 - <<'PY'
import io
import os

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

num_samples = 64
size = 512
root = os.path.join(os.environ["GITHUB_WORKSPACE"], "sana_fixture")
os.makedirs(root, exist_ok=True)
path = os.path.join(root, "sana_train.parquet")

prompts = []
images = []
for i in range(num_samples):
    image = Image.new("RGB", (size, size), (60 + i % 180, 120, 160))
    draw = ImageDraw.Draw(image)
    draw.rectangle([64, 64, size - 64, size - 64], outline=(240, 240, 240), width=6)
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    images.append(buf.getvalue())
    prompts.append(f"a synthetic line drawing scene {i}")

table = pa.table(
    {
        "llava": pa.array(prompts, type=pa.string()),
        "image": pa.array(images, type=pa.binary()),
    }
)
pq.write_table(table, path)
with open(os.environ["GITHUB_ENV"], "a") as fh:
    fh.write(f"SANA_FIXTURE_PARQUET={path}\n")
print(f"prepared {num_samples}-sample sana fixture parquet at {path}", flush=True)
PY
}

setup_diffusers_multi_subject_inpaint() {
  # multi_subject_dreambooth_inpainting 的 Dataset 用 load_dataset(path) 读本地
  # 目录（不是 load_from_disk），固定取 train/test 两个 split，且列里要有一个
  # 名字含 "mask" 的列。所以仿 Hub 数据集的落盘布局：平铺的
  # train-*.parquet / test-*.parquet，load_dataset 按扩展名推断 parquet builder、
  # 按文件名推断 split，features（Image）存在 schema metadata 里，读回即 PIL。
  # 模型必须是 9 通道的 inpainting 版（不是基础版 SD1.5），由 PROFILE_ASSETS
  # 表预下。
  python3 - <<'PY'
import os
from pathlib import Path

from datasets import Dataset, Features, Image, Value
from PIL import Image as PILImage
from PIL import ImageDraw

size = 512
num_samples = 4
prompts = [f"a photo of sks dog in scene {i}" for i in range(num_samples)]
images = []
masks = []
for i in range(num_samples):
    image = PILImage.new("RGB", (size, size), (60 + i % 180, 120, 160))
    ImageDraw.Draw(image).rectangle([64, 64, size - 64, size - 64], outline=(240, 240, 240), width=6)
    images.append(image)
    mask = PILImage.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rectangle([64, 64, size - 64, size - 64], fill=255)
    masks.append(mask)

features = Features({"image": Image(), "prompt": Value("string"), "mask": Image()})
split = Dataset.from_dict({"image": images, "prompt": prompts, "mask": masks}, features=features)

root = Path(os.environ["GITHUB_WORKSPACE"]) / "multi_subject_inpaint_fixture"
root.mkdir(parents=True, exist_ok=True)
split.to_parquet(str(root / "train-00000-of-00001.parquet"))
split.to_parquet(str(root / "test-00000-of-00001.parquet"))
with open(os.environ["GITHUB_ENV"], "a") as fh:
    fh.write(f"MULTI_SUBJECT_INPAINT_DATASET={root}\n")
print(f"prepared multi-subject inpainting fixture (train+test parquet) at {root}", flush=True)
PY
}

setup_diffusers_realfill() {
  # train_realfill.py 的 RealFillDataset 认死一个目录结构：
  #   <root>/ref/*.png + <root>/target/target.png + <root>/target/mask.png
  # 缺任一就直接 ValueError（:445）。这里现场合成一个最小 fixture（纯合成图，零下载）。
  # 模型是 SD1.5-inpainting（同样 9 通道），由 PROFILE_ASSETS 表预下。
  python3 - <<'PY'
import os
from pathlib import Path

from PIL import Image, ImageDraw

size = 512
root = Path(os.environ["GITHUB_WORKSPACE"]) / "realfill_fixture"
(ref := root / "ref").mkdir(parents=True, exist_ok=True)
(target := root / "target").mkdir(parents=True, exist_ok=True)


def scene(seed):
    img = Image.new("RGB", (size, size), (40 + seed % 160, 110, 150))
    draw = ImageDraw.Draw(img)
    draw.rectangle([80, 80, size - 80, size - 80], outline=(240, 240, 240), width=8)
    draw.ellipse([180, 180, 330, 330], fill=(210, 90, 60))
    return img


for i in range(3):
    scene(i).save(ref / f"ref{i}.png")
scene(9).save(target / "target.png")
mask = Image.new("L", (size, size), 0)
ImageDraw.Draw(mask).rectangle([80, 80, size - 80, size - 80], fill=255)
mask.save(target / "mask.png")

with open(os.environ["GITHUB_ENV"], "a") as fh:
    fh.write(f"REALFILL_DATASET={root}\n")
print(f"prepared realfill fixture at {root}", flush=True)
PY
}

setup_diffusers_pixart() {
  # PixArt-alpha/PixArt-XL-2-512x512 在 ModelScope 无镜像，只能走 HuggingFace 直连。
  # 它的 text_encoder 是 T5-XXL：脚本按默认（不带 --variant）加载，只会用 fp32
  # 权重（4 分片 ~19GB），另有一套 fp16 分片（~9GB）用不上——预下时跳过 *.fp16.*
  # 省掉一半，且只下组件目录（asset/ 里的 README 图/样例文本下载慢、易超时）。
  hub_snapshot "PixArt-alpha/PixArt-XL-2-512x512" pixart_model PIXART_MODEL_PATH \
    "model_index.json,scheduler/*,tokenizer/*,text_encoder/*,transformer/*,vae/*" \
    "*.fp16.*,*.bin,*.ckpt"
}

setup_diffusers_flux2_klein() {
  # FLUX.2-klein-4B：BFL 的开放权重档（实测非 gated、匿名可取）。预下只取组件
  # 目录（跳过顶层 ~7.2GB 单文件 checkpoint 重复）。img2img 条目的
  # DreamBoothDataset 会把整个数据集急切物化进宿主内存（fill50k 的 1000 对
  # 图+条件图直接把 RAM 吃穿 → 内核 SIGKILL），脚本又没有 max_train_samples
  # 切片参数，所以现场合成 8 对小数据集（列 image/conditioning_image/text）。
  hub_snapshot "black-forest-labs/FLUX.2-klein-4B" flux2_klein_model FLUX2_KLEIN_MODEL_PATH \
    "model_index.json,scheduler/*,tokenizer/*,text_encoder/*,transformer/*,vae/*" -
  python3 - <<'PY'
import os
from pathlib import Path

from datasets import Dataset, Features, Image as ImageFeature, Value
from PIL import Image, ImageDraw

size = 512
images, conds, captions = [], [], []
for i in range(8):
    img = Image.new("RGB", (size, size), (60 + i % 180, 120, 160))
    ImageDraw.Draw(img).rectangle([64, 64, size - 64, size - 64], outline=(240, 240, 240), width=6)
    cond = Image.new("RGB", (size, size), (30, 30, 30))
    ImageDraw.Draw(cond).line([0, (i * 61) % size, size, (i * 97) % size], fill=(255, 255, 255), width=8)
    images.append(img)
    conds.append(cond)
    captions.append(f"a synthetic line drawing scene {i}")

features = Features(
    {
        "image": ImageFeature(),
        "conditioning_image": ImageFeature(),
        "text": Value("string"),
    }
)
ds = Dataset.from_dict(
    {"image": images, "conditioning_image": conds, "text": captions}, features=features
)
data_dir = Path(os.environ["GITHUB_WORKSPACE"]) / "flux2_klein_img2img_dataset"
data_dir.mkdir(parents=True, exist_ok=True)
ds.to_parquet(str(data_dir / "train-00000-of-00001.parquet"))

with open(os.environ["GITHUB_ENV"], "a") as fh:
    fh.write(f"FLUX2_KLEIN_DATASET={data_dir}\n")
print(f"prepared img2img dataset at {data_dir}", flush=True)
PY
}

# Base stack + the three tables ALWAYS apply; a bespoke setup_<profile>()
# function, when present, runs ON TOP of them for the genuinely bespoke steps
# (hub model pre-downloads, fixture synthesis) instead of replacing everything.
# This keeps pip/env/assets registration in one place: a profile that needs
# deepspeed just gets a PROFILE_PIP row - no bespoke function can forget it.
run_profile() {
  install_example_stack
  if [[ -n "${PROFILE_PIP[$PROFILE]:-}" ]]; then
    python -m pip install ${PROFILE_PIP[$PROFILE]}
  fi
  # hf_transfer (pulled in via PROFILE_PIP above): parallel hub downloads for
  # both the setup-time hub_snapshot calls and the run step.
  if [[ "${PROFILE_PIP[$PROFILE]:-}" == *hf_transfer* ]] && python -c "import hf_transfer" >/dev/null 2>&1; then
    export HF_XET_HIGH_PERFORMANCE=1
    echo "HF_XET_HIGH_PERFORMANCE=1" >> "$GITHUB_ENV"
  fi
  if [[ -n "${PROFILE_ASSETS[$PROFILE]:-}" ]]; then
    download_assets "${PROFILE_ASSETS[$PROFILE]}"
  fi
  local var
  for var in ${PROFILE_ENV[$PROFILE]:-}; do
    echo "$var" >> "$GITHUB_ENV"
  done
  local fn="setup_${PROFILE//-/_}"
  if declare -F "$fn" >/dev/null 2>&1; then
    "$fn"
  fi
}

if [[ -z "${TARGET_ROOT:-}" || -z "${GITHUB_WORKSPACE:-}" || -z "${GITHUB_ENV:-}" ]]; then
  echo "TARGET_ROOT / GITHUB_WORKSPACE / GITHUB_ENV must be set" >&2
  exit 2
fi

HERE=$(cd "$(dirname "$0")" && pwd)
export PIP_CONSTRAINT="$(cd "$HERE/.." && pwd)/constraints-npu.txt"

source /usr/local/Ascend/ascend-toolkit/set_env.sh

select_pip_index
python -m pip install -U pip setuptools wheel
ensure_torch_stack
prepare_fixtures

# All examples: keep wandb offline. Some entrypoints default --report_to to
# wandb; WANDB_MODE=disabled prevents any network reporting and is harmless
# for the ones that use tensorboard. Persisted to the run step via GITHUB_ENV.
echo "WANDB_MODE=disabled" >> "$GITHUB_ENV"

# All examples: let HCCL pick a free socket port range (required on some
# Ascend setups to avoid "address already in use" on multi-process init).
echo "HCCL_NPU_SOCKET_PORT_RANGE=auto" >> "$GITHUB_ENV"

run_profile
