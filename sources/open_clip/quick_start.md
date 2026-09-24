# open_clip

在单张昇腾 NPU 上，使用 [open_clip](https://github.com/mlfoundations/open_clip)
的预训练模型比较图片与文本的相似度，再用合成数据运行一次 RN50 训练。
完成后，可以确认模型推理和训练都能在 `npu:0` 上运行。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，至少有一张可用
的 Ascend NPU，并已完成物理机或容器中的设备与驱动配置。

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
| open_clip | 最新 release |
| 模型 | `ViT-B-32` / `laion2b_s34b_b79k` |
| NPU | Ascend 910B4 × 1 |

### 检查前置条件

检查 Python：

```shell #test id="check-python"
python --version
```

输出结果如下：

```text #test-result id="check-python" fuzzy="xxx"
Python 3.12.xxx
```

检查 PyTorch-NPU 和可见设备：

```python #test id="check-torch"
import torch
import torch_npu

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("npu_available:", torch.npu.is_available())
print("npu_count:", torch.npu.device_count())
```

输出结果如下：

```text #test-result id="check-torch"
torch: 2.9.0+cpu
torch_npu: 2.9.0.post2
npu_available: True
npu_count: 1
```

## 安装 open_clip

从 [open_clip Releases](https://github.com/mlfoundations/open_clip/releases) 选择
最新正式版本，将命令中的 `<ref>` 替换为对应的标签，再安装 open_clip。

<!--
```shell #test-setup store="upstream_ref"
printf '%s\n' "$UPSTREAM_REF"
```
-->

```shell #test id="install-open-clip" load="upstream_ref>>ref"
git clone --depth 1 --branch <ref> https://github.com/mlfoundations/open_clip.git open-clip-src
cd open-clip-src
python -m pip install -U uv
uv pip install -r requirements-training.txt
uv pip install -e . --no-deps
python -c "import open_clip; print('open_clip', open_clip.__version__)"
```

输出结果如下：

```text #test-result id="install-open-clip" fuzzy="xxx"
open_clip xxx
```

## 单卡预训练图文推理

使用预训练 `ViT-B-32` 模型，分别提取图片和三个候选文本的特征，
选择与图片最匹配的描述。模型与输入都放在同一张 NPU 上。

```python #test id="npu-inference"
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
```

输出结果如下：

```text #test-result id="npu-inference"
device: npu:0
top label: a diagram
NPU inference PASSED
```

## 单卡最小训练验证

使用 4 个自动生成的样本运行一次 RN50 单卡训练，无需下载训练数据。
命令会在 `npu:0` 上以 FP32 精度完成损失计算、反向传播和模型参数更新。

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

输出结果如下：

```text #test-result id="npu-training" fuzzy="xxx"
xxxRunning with a single process. Device npu:0.xxx
xxxTrain Epoch: 0xxx
NPU training PASSED
```

看到 `NPU training PASSED` 表示单卡训练命令已运行完成。多卡训练和其他模型
不在本文范围内。
