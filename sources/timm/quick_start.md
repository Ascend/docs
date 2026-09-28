# pytorch-image-models

在昇腾 NPU 上安装 timm，并用预训练 ResNet-18 跑通图像分类推理和多尺度特征提取。

## 前置条件

### 硬件

Atlas 900 A2 / A3 或 Ascend 950 系列服务器（Ascend 910B），并按需完成物理机或容器内的设备挂载。

### 基础软件

在运行本文档示例之前，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 与 CANN 匹配的 `torch` + `torch_npu`（参考 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)）

本文档示例在 Python 3.12、CANN 9.1.0、torch 2.9.0、torch_npu 2.9.0.post2 环境下验证通过。

## 加载 CANN 环境

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 安装 timm

从 PyPI 安装最新 release，并确认可以正常导入：

```shell #test id="install-timm"
pip install timm
python -c "import timm; print('timm', timm.__version__)"
```

输出结果如下，其中 `xxx` 表示实际版本号：

```shell #test-result id="install-timm" fuzzy='...' fuzzy='xxx'
...
timm xxx
```

## 快速开始

### 示例 1：图像分类推理

timm 的入门示例是 `timm.create_model('resnet18')` 创建模型并前向推理。详见[官方文档](https://timm.fast.ai/#create-a-model)。

首次运行自动从 ModelScope 下载 ResNet-18 预训练权重（来源 [timm/resnet18.a1_in1k](https://modelscope.cn/models/timm/resnet18.a1_in1k)），输入随机图像，输出 1000 类 logits，运行下面的 Python 脚本：

```python #test id="inference"
import os

import safetensors.torch
import timm
import torch
import torch_npu

try:
    from modelscope.hub.snapshot_download import snapshot_download

    model_dir = snapshot_download("timm/resnet18.a1_in1k")
    model = timm.create_model("resnet18")
    model.load_state_dict(
        safetensors.torch.load_file(os.path.join(model_dir, "model.safetensors"))
    )
except Exception:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    model = timm.create_model("resnet18", pretrained=True)

model = model.to("npu:0").eval()
x = torch.randn(1, 3, 224, 224, device="npu:0")
with torch.no_grad():
    out = model(x)
print("out_shape", tuple(out.shape))
```

输出结果如下：

```shell #test-result id="inference"
out_shape (1, 1000)
```

### 示例 2：多尺度特征提取

timm 的 `features_only=True` 可将任意模型转为多尺度特征提取器，输出各层 feature map。详见[Feature Extraction 文档](https://huggingface.co/docs/timm/en/feature_extraction)。

加载预训练 ResNet-18，输出 stride 2/4/8/16/32 共 5 层 feature map，打印每层形状和通道数，运行下面的 Python 脚本：

```python #test id="features"
import os

import safetensors.torch
import timm
import torch
import torch_npu

try:
    from modelscope.hub.snapshot_download import snapshot_download

    model_dir = snapshot_download("timm/resnet18.a1_in1k")
    model = timm.create_model("resnet18", features_only=True)
    model.load_state_dict(
        safetensors.torch.load_file(os.path.join(model_dir, "model.safetensors")),
        strict=False,
    )
except Exception:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    model = timm.create_model("resnet18", features_only=True, pretrained=True)

model = model.to("npu:0").eval()
x = torch.randn(1, 3, 224, 224, device="npu:0")
with torch.no_grad():
    outs = model(x)
print("num_features", len(outs))
print("channels", model.feature_info.channels())
for i, o in enumerate(outs):
    print(i, tuple(o.shape))
```

输出结果如下：

```shell #test-result id="features"
num_features 5
channels [64, 64, 128, 256, 512]
0 (1, 64, 112, 112)
1 (1, 64, 56, 56)
2 (1, 128, 28, 28)
3 (1, 256, 14, 14)
4 (1, 512, 7, 7)
```

## 更多用法

训练脚本、多卡分布式、模型导出等更多用法见 [timm 官方文档](https://huggingface.co/docs/timm/) 和 [GitHub 仓库](https://github.com/huggingface/pytorch-image-models)。
