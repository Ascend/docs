#!/usr/bin/env bash
# Prepare one example from the exact release checked out by examples-template.
set -euo pipefail

profile="${1:-}"
case "$profile" in
  vision_gallery_npu|vision_classification_npu|vision_segmentation_npu|\
  vision_flow_npu|vision_video_npu) ;;
  *) echo "unknown TorchVision example profile: ${profile:-<missing>}" >&2; exit 2 ;;
esac

: "${TARGET_ROOT:?TARGET_ROOT is required}"
: "${PROJECT_ROOT:?PROJECT_ROOT is required}"
source /usr/local/Ascend/ascend-toolkit/set_env.sh

resolved_tag="$(git -C "$TARGET_ROOT" describe --tags --exact-match 2>/dev/null || true)"
echo "TorchVision tested release: ${resolved_tag:-<tag unavailable in shallow checkout>}"
echo "TorchVision tested commit: $(git -C "$TARGET_ROOT" rev-parse HEAD)"

# Use the same public Ascend-compatible wheel sources as this project's
# quick-start. The source install below uses --no-deps and FORCE_CUDA=0 so
# torchvision cannot replace torch_npu's matching PyTorch with a CUDA wheel.
#
# torch_npu needs no --extra-index-url: 2.14.0 is on PyPI. torch must
# come from aliyun's pytorch-wheels/cpu find-links repo — PyPI's
# torch==2.14.0 aarch64 wheel is the CUDA build.
python -m pip install uv
uv pip install --system -f https://mirrors.aliyun.com/pytorch-wheels/cpu 'torch==2.14.0+cpu'
uv pip install --system 'torch_npu==2.14.0'
uv pip install --system 'pillow>=10.0' numpy
FORCE_CUDA=0 uv pip install --system --no-build-isolation --no-deps -e "$TARGET_ROOT"

python - <<'PY'
import os
from pathlib import Path

import torch
import torch_npu
import torchvision

target = Path(os.environ["TARGET_ROOT"]).resolve()
loaded = Path(torchvision.__file__).resolve()
declared = (target / "version.txt").read_text(encoding="utf-8").strip()
print(f"torch={torch.__version__} torch_npu={torch_npu.__version__}")
print(f"torchvision={torchvision.__version__} source={loaded}")
if not torch.__version__.startswith("2.14.0") or not torch_npu.__version__.startswith("2.14.0"):
    raise SystemExit("torch and torch_npu must match the declared CANN 9.1 stack")
if not loaded.is_relative_to(target) or not torchvision.__version__.startswith(declared):
    raise SystemExit("torchvision must be installed from this release checkout")
if not torch.npu.is_available() or torch.npu.device_count() < 1:
    raise SystemExit("NPU unavailable; refusing CPU fallback")
PY

if [[ "$profile" == vision_classification_npu ]]; then
  : "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
  export VISION_DATA_ROOT="$GITHUB_WORKSPACE/vision-classification-fixture"
  python - <<'PY'
import os
from pathlib import Path
from PIL import Image

root = Path(os.environ["VISION_DATA_ROOT"])
for split, count in (("train", 2), ("val", 1)):
    for cls in range(5):  # upstream evaluation always calculates top-5
        folder = root / split / f"class_{cls}"
        folder.mkdir(parents=True, exist_ok=True)
        for item in range(count):
            image = Image.new("RGB", (80, 80), (25 + cls * 35, 15 + item * 80, 180 - cls * 20))
            image.save(folder / f"{item}.png")
print(f"five-class ImageFolder fixture: {root}")
PY
  if [[ -n "${GITHUB_ENV:-}" ]]; then
    printf 'VISION_DATA_ROOT=%s\n' "$VISION_DATA_ROOT" >> "$GITHUB_ENV"
  fi
fi

if [[ "$profile" == vision_gallery_npu ]]; then
  uv pip install --system matplotlib
  export MPLBACKEND=Agg
  [[ -z "${GITHUB_ENV:-}" ]] || printf 'MPLBACKEND=Agg\n' >> "$GITHUB_ENV"
fi

if [[ "$profile" == vision_flow_npu || "$profile" == vision_video_npu ]]; then
  : "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
  export VISION_REFERENCE_DATA_ROOT="$GITHUB_WORKSPACE/vision-$profile-fixture"
  kind=flow
  if [[ "$profile" == vision_video_npu ]]; then
    kind=video
    # fixture encoder only (PyAV bundles its own FFmpeg); the train script
    # never goes through it.
    uv pip install --system 'av>=14,<17'
    # torchvision v0.29 removed video decoding: Kinetics/VideoClips imports
    # torchcodec unconditionally. torchcodec >=0.12 builds against torch's
    # stable ABI (torch >=2.11; this stack runs 2.14.0). PyPI's aarch64
    # wheel is the CUDA build - libtorchcodec_image.so links
    # libnvrtc.so.13 (run 39894257350) - so fetch the +cpu variant by
    # direct URL from download.pytorch.org/whl/cpu, same as peft's setup
    # does for torch/torchvision +cpu wheels. --no-deps keeps the
    # resolver from ever seeing a CUDA-flavored torch.
    CP_ABI=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
    uv pip install --system --no-deps \
      "https://download.pytorch.org/whl/cpu/torchcodec-0.17.0%2Bcpu-${CP_ABI}-${CP_ABI}-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl"
    # torchcodec dlopen()s system FFmpeg (4-9) at decode time; install the
    # shared libs unconditionally (fast no-op when the image already has
    # them) so a missing libav* surfaces here instead of mid-training.
    apt-get update -qq && apt-get install -y -qq --no-install-recommends ffmpeg
  fi
  python "$PROJECT_ROOT/scripts/prepare_ci_assets.py" "$kind" "$VISION_REFERENCE_DATA_ROOT"
  if [[ "$profile" == vision_video_npu ]]; then
    # fail fast when torchcodec cannot load FFmpeg or decode the fixture
    python - "$VISION_REFERENCE_DATA_ROOT" <<'PY'
import sys

from torchcodec.decoders import VideoDecoder

decoder = VideoDecoder(f"{sys.argv[1]}/ci_template.avi")
frame = decoder[0]
print(f"torchcodec decode ok: {tuple(frame.shape)}")
PY
  fi
  [[ -z "${GITHUB_ENV:-}" ]] || printf 'VISION_REFERENCE_DATA_ROOT=%s\n' "$VISION_REFERENCE_DATA_ROOT" >> "$GITHUB_ENV"
fi

if [[ "$profile" == vision_segmentation_npu ]]; then
  # train.py imports coco_utils at module load even for --dataset voc;
  # coco_utils imports pycocotools unconditionally. Keep this optional
  # reference-script dependency scoped to the segmentation profile.
  uv pip install --system pycocotools
  python -c 'from pycocotools import mask; print("pycocotools mask import OK")'

  : "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
  export VISION_SEG_DATA_ROOT="$GITHUB_WORKSPACE/vision-segmentation-fixture"
  python - <<'PY'
import os
from pathlib import Path
from PIL import Image, ImageDraw

root = Path(os.environ["VISION_SEG_DATA_ROOT"]) / "VOCdevkit" / "VOC2012"
images = root / "JPEGImages"
masks = root / "SegmentationClass"
splits = root / "ImageSets" / "Segmentation"
for folder in (images, masks, splits):
    folder.mkdir(parents=True, exist_ok=True)

for split, names in (("train", ("ci_train_0", "ci_train_1")), ("val", ("ci_val_0",))):
    (splits / f"{split}.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    for idx, name in enumerate(names):
        Image.new("RGB", (80, 80), (40 + idx * 50, 90, 130)).save(images / f"{name}.jpg")
        mask = Image.new("L", (80, 80), 0)
        ImageDraw.Draw(mask).rectangle((16, 16, 63, 63), fill=idx + 1)
        mask.save(masks / f"{name}.png")
print(f"generated three-image VOC segmentation fixture at {root}")
PY
  if [[ -n "${GITHUB_ENV:-}" ]]; then
    printf 'VISION_SEG_DATA_ROOT=%s\n' "$VISION_SEG_DATA_ROOT" >> "$GITHUB_ENV"
  fi
fi
