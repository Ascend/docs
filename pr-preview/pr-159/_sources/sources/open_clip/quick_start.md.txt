# 快速开始

在单卡昇腾 NPU 上运行 [open_clip](https://github.com/mlfoundations/open_clip)
官方 README 的预训练图文相似度示例，并用上游测试已有的 synthetic 数据方式
完成一次最小训练：先确认 `ViT-B-32` 图片最匹配 `a diagram`，再验证 RN50
能够在 `npu:0` 上执行前向、反向和优化器更新。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，至少有一张可用
的 Ascend NPU，并已完成物理机或容器中的设备与驱动配置。CI 使用
`linux-aarch64-a2-1` Runner，由 Runner 自动提供一张配置完好的 NPU。

### 基础软件

在运行本文档之前，机器上需要已经安装并可用：

- Linux aarch64 操作系统；
- Python 3.12；
- CANN toolkit 和驱动；
- 与 CANN 匹配的 `torch`、`torchvision` 和 `torch_npu`；
- 能访问 open_clip 预训练权重，或者已经准备好 Hugging Face 本地缓存。

### 本文档示例使用的版本

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| torch | 2.9.0+cpu |
| torchvision | 0.24.0 |
| torch_npu | 2.9.0.post2 |
| open_clip | 工作流注入的最新 release 源码 |
| 模型 | `ViT-B-32` / `laion2b_s34b_b79k` |
| NPU | Ascend 910B4 × 1 |

### 检查前置条件

检查 Python：

```shell #test id="check-python"
python --version
```

```shell #test-result id="check-python" fuzzy="xxx"
Python 3.12.xxx
```

检查 PyTorch-NPU 和可见设备：

```shell #test id="check-torch"
python - <<'PY'
import torch
import torch_npu

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("npu_available:", torch.npu.is_available())
print("npu_count:", torch.npu.device_count())
PY
```

```shell #test-result id="check-torch" fuzzy="xxx"
torch: 2.9.0xxx
torch_npu: 2.9.0.post2
npu_available: True
npu_count: 1
```

## 安装 open_clip

下面安装工作流解析出的最新 GitHub release，而不是静态固定 open_clip 版本。

<!--
```shell #test-setup store="upstream_ref"
printf '%s\n' "$UPSTREAM_REF"
```
-->

```shell #test id="install-open-clip" load="upstream_ref>>ref"
git clone --depth 1 --branch <ref> https://github.com/mlfoundations/open_clip.git open-clip-src
cd open-clip-src
uv pip install -r requirements-training.txt
uv pip install -e . --no-deps
python -c "import open_clip; print('open_clip', open_clip.__version__)"
```

```shell #test-result id="install-open-clip" fuzzy="xxx"
open_clip xxx
```

## Quick Start：单卡预训练图文推理

本例来自 open_clip README 的 `ViT-B-32` 推理流程。与 CUDA 示例相比，
只增加 `torch_npu` 导入、`device="npu:0"`，并把输入移动到同一张 NPU。

```shell #test id="npu-inference"
python - <<'PY'
import torch
import torch_npu
from PIL import Image
import open_clip

device = "npu:0"
labels = ["a diagram", "a dog", "a cat"]

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k",
    device="npu:0",
)
model.eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")

image = preprocess(Image.open("open-clip-src/docs/CLIP.png")).unsqueeze(0).to("npu:0")
text = tokenizer(labels).to("npu:0")

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    probabilities = (100.0 * image_features @ text_features.T).softmax(dim=-1)

top_label = labels[probabilities.argmax(dim=-1).item()]
assert next(model.parameters()).device.type == "npu"
assert image_features.device.type == "npu"
assert text_features.device.type == "npu"
assert torch.isfinite(probabilities).all()
assert top_label == "a diagram"

print("device:", next(model.parameters()).device)
print("top label:", top_label)
print("NPU inference PASSED")
PY
```

```shell #test-result id="npu-inference"
device: npu:0
top label: a diagram
NPU inference PASSED
```

## 单卡最小训练验证

本步骤来自上游 `tests/test_training_simple.py` 的 RN50 synthetic training 用例。
它不下载训练数据，只把上游测试规模缩小为 4 个样本，并增加
`--device npu:0` 和 `--precision fp32`；因此会真实执行损失计算、反向传播和
优化器更新，同时保持 Quick Start 足够轻量。

```shell #test id="npu-training"
set -euo pipefail
cd open-clip-src
python -m open_clip_train.main \
    --dataset-type synthetic \
    --train-num-samples 4 \
    --batch-size 2 \
    --epochs 1 \
    --workers 0 \
    --model RN50 \
    --device npu:0 \
    --precision fp32 \
    --warmup 1 \
    --lr 1e-3 \
    --wd 0.1 \
    --save-frequency 0 \
    --zeroshot-frequency 0 \
    --logs none 2>&1
echo "NPU training PASSED"
```

```shell #test-result id="npu-training" fuzzy="xxx"
xxxRunning with a single process. Device npu:0.xxx
xxxTrain Epoch: 0xxx
NPU training PASSED
```

该示例覆盖真实权重单卡推理和 synthetic 数据单卡训练，不覆盖 HCCL 多卡训练、
FSDP、音频模型、CoCa、INT8 或 `torch.compile`。
