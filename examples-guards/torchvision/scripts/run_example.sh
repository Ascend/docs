#!/usr/bin/env bash
# Run the original upstream script and require an observable NPU result.
set -euo pipefail

: "${TARGET_ROOT:?TARGET_ROOT is required}"
: "${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"
entry="${1:-}"
case "$entry" in
  gallery/transforms/plot_custom_transforms.py|\
  gallery/transforms/plot_custom_tv_tensors.py|\
  gallery/transforms/plot_tv_tensors.py|\
  gallery/transforms/plot_keypoints_transforms.py|\
  references/classification/train.py|\
  references/segmentation/train.py|\
  references/optical_flow/train.py|\
  references/video_classification/train.py) ;;
  *) echo "unsupported TorchVision example entry: ${entry:-<missing>}" >&2; exit 2 ;;
esac

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
mkdir -p "$CI_OUTPUT_DIR"

python - "$entry" <<'PY'
import json
import os
from pathlib import Path
import runpy
import sys

import torch
import torch_npu
import torchvision

entry = sys.argv[1]
root = Path(os.environ.get("EXAMPLES_ROOT") or os.environ["TARGET_ROOT"]).resolve()
script = (root / entry).resolve()
if not script.is_relative_to(root) or not script.is_file():
    raise SystemExit(f"upstream example missing or outside checkout: {entry}")

overlay = json.loads(os.environ.get("OVERLAY_ARGS") or "[]")
if not isinstance(overlay, list) or not all(isinstance(arg, str) for arg in overlay):
    raise SystemExit("OVERLAY_ARGS must be a JSON string array")
if not torch.npu.is_available() or torch.npu.device_count() < 1:
    raise SystemExit("NPU unavailable; refusing CPU fallback")
torch.npu.set_device(0)

if entry.startswith("gallery/"):
    if overlay:
        raise SystemExit("gallery entry does not take CLI overlay arguments")
    torch.set_default_device("npu:0")
    sys.argv = [str(script)]
elif entry == "references/classification/train.py":
    data_root = os.environ.get("VISION_DATA_ROOT")
    if not data_root or not Path(data_root, "train").is_dir() or not Path(data_root, "val").is_dir():
        raise SystemExit("five-class ImageFolder fixture missing")
    if "--device" not in overlay or "npu:0" not in overlay:
        raise SystemExit("classification entry requires --device npu:0")
    output = Path(os.environ["CI_OUTPUT_DIR"]) / "classification"
    output.mkdir(parents=True, exist_ok=True)
    sys.argv = [str(script), *overlay, "--data-path", data_root, "--output-dir", str(output)]
elif entry == "references/segmentation/train.py":
    data_root = os.environ.get("VISION_SEG_DATA_ROOT")
    voc_root = Path(data_root or "") / "VOCdevkit" / "VOC2012"
    if not data_root or not (voc_root / "ImageSets" / "Segmentation" / "train.txt").is_file():
        raise SystemExit("three-image VOC segmentation fixture missing")
    if "--dataset" not in overlay or "voc" not in overlay or "--device" not in overlay or "npu:0" not in overlay:
        raise SystemExit("segmentation entry requires --dataset voc --device npu:0")
    output = Path(os.environ["CI_OUTPUT_DIR"]) / "segmentation"
    output.mkdir(parents=True, exist_ok=True)
    sys.argv = [str(script), *overlay, "--data-path", data_root, "--output-dir", str(output)]
else:
    data_root = os.environ.get("VISION_REFERENCE_DATA_ROOT")
    if not data_root or not Path(data_root).is_dir():
        raise SystemExit("local flow/video reference fixture missing")
    if "--device" not in overlay or "npu:0" not in overlay:
        raise SystemExit("new reference entry requires --device npu:0")
    if any(key in os.environ for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "SLURM_PROCID")):
        raise SystemExit("new reference recipes require plain single-process launch; distributed CUDA path is not selected")
    kind = "optical-flow" if "optical_flow" in entry else "video-classification"
    output = Path(os.environ["CI_OUTPUT_DIR"]) / kind
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / ("ci_flow_0.pth" if kind == "optical-flow" else "model_0.pth")
    checkpoint.unlink(missing_ok=True)  # do not accept a prior run's checkpoint
    data_flag = "--dataset-root" if kind == "optical-flow" else "--data-path"
    sys.argv = [str(script), *overlay, data_flag, data_root, "--output-dir", str(output)]

if entry == "references/video_classification/train.py":
    # CANN 9.1.0's fp32 Conv3DBackpropInput kernel rejects stride_d >
    # kernel_d (R3D-18 downsample 1x1x1 conv / stride (2,2,2); run
    # 39895381622; allow_tf32 and allow_internal_format do not route
    # around it). The fp16 kernel handles the shape, so the manifest
    # overlays upstream's --amp. train.py gates that on
    # torch.cuda.amp.autocast, a no-op on the +cpu torch build - this
    # shim (same as peft's cuda->npu sitecustomize) maps it to NPU
    # autocast. Verified on coder npu-3 (CANN 9.1.0): autocast output
    # fp16, backward + GradScaler + optimizer step all green.
    import torch_npu.contrib.transfer_to_npu  # noqa: F401

os.chdir(script.parent)
sys.path.insert(0, str(script.parent))
print(f"running unchanged upstream {entry} with torchvision {torchvision.__version__}")
namespace = runpy.run_path(str(script), run_name="__main__")

def require_npu(name):
    value = namespace.get(name)
    if not isinstance(value, torch.Tensor) or value.device.type != "npu":
        raise SystemExit(f"{entry}: {name} did not finish on NPU: {getattr(value, 'device', None)}")

if entry.endswith("plot_custom_transforms.py"):
    require_npu("out_img")
    require_npu("out_bboxes")
elif entry.endswith("plot_custom_tv_tensors.py"):
    # The v0.29 tutorial leaves the custom tensor and hflip result in
    # namespace variables; do not rely on an earlier release's examples.
    require_npu("my_dp")
    require_npu("wrapped")
    require_npu("wrapped_dog")
elif entry.endswith("plot_tv_tensors.py"):
    # The tutorial's first Image is a zero-copy wrapper around this tensor.
    require_npu("tensor")
elif entry.endswith("plot_keypoints_transforms.py"):
    require_npu("orig_pts")
    # Source retains PIL images for documentation rendering; point geometry is
    # the genuine NPU workload, including rotation/perspective/crop operations.
    for group in ("rotated_imgs", "padded_imgs_and_points", "resized_imgs", "perspective_imgs", "center_crops_and_points"):
        values = namespace.get(group)
        if not values or len(values) != 4:
            raise SystemExit(f"keypoint tutorial result group missing: {group}")
        for _, points in values:
            if not isinstance(points, torch.Tensor) or points.device.type != "npu" or not bool(torch.isfinite(points).all()):
                raise SystemExit(f"keypoint geometry did not complete on NPU: {group}")
elif entry in {"references/optical_flow/train.py", "references/video_classification/train.py"}:
    if not checkpoint.is_file():
        raise SystemExit(f"{entry}: did not finish the original epoch and save a fresh checkpoint")
    if torch.npu.max_memory_allocated() <= 0:
        raise SystemExit(f"{entry}: did not allocate NPU memory")
else:
    if not (output / "model_0.pth").is_file():
        raise SystemExit(f"{entry}: did not finish an epoch and save a checkpoint")
    if torch.npu.max_memory_allocated() <= 0:
        raise SystemExit(f"{entry}: did not allocate NPU memory")

torch.npu.synchronize()
print(f"NPU example passed: {entry}")
PY
