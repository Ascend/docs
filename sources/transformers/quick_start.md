# Transformers Ascend 快速开始

在单卡昇腾 NPU 上验证 Transformers 的文本生成 Pipeline。本文使用公开模型，
不需要 Hugging Face token。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或其他兼容的 Ascend NPU，至少有一张可用设备，
并已完成物理机或容器中的设备与驱动配置。

### 基础软件

运行本文档前，需要准备：

- Linux aarch64 和 Python 3.12；
- CANN 9.1.0 toolkit、驱动和 `npu-smi`；
- 与 CANN 匹配的 `torch==2.9.0` 和 `torch_npu==2.9.0.post2`；
- 可安装 Python 包的网络或本地缓存。

### 本文档示例使用的版本

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| torch | 2.9.0 |
| torch_npu | 2.9.0.post2 |
| transformers | 最新 release |
| accelerate | 当前稳定版本 |
| 模型 | `Qwen/Qwen2.5-1.5B` |
| NPU | Ascend 910B4 × 1 |

## 环境准备

本文示例在下面的 Ascend 镜像中验证通过：

```text
swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12
```

镜像通常已经包含兼容的 `torch` / `torch_npu`。如果本地镜像没有提供，
请先按照 CANN 与 PyTorch-NPU 的兼容矩阵安装对应版本。

首次运行示例会自动从 Hugging Face 下载公开的 `Qwen/Qwen2.5-1.5B`；也可以
通过环境变量 `QUICK_START_MODEL` 指定本地已下载的模型路径，实现离线加载。

## 检查前置是否满足

检查 Python 版本：

```shell #test id="check-py"
python --version
```

```shell #test-result id="check-py" fuzzy="xxx"
Python 3.12.xxx
```

检查 CANN、Torch、Torch-NPU 和 NPU 设备：

```shell #test id="check-torch"
python -c "import torch, torch_npu; print('torch=', torch.__version__); print('torch_npu=', torch_npu.__version__); print('is_available:', torch.npu.is_available()); print('count:', torch.npu.device_count())"
```

```shell #test-result id="check-torch" fuzzy="xxx"
torch= 2.9.0xxx
torch_npu= 2.9.0.post2
is_available: True
count: 1
```

确认 `npu-smi` 可以看到设备：

```shell #test id="check-npu-smi"
npu-smi info >/dev/null
echo "npu-smi: ready"
```

```shell #test-result id="check-npu-smi"
npu-smi: ready
```

如果 `npu-smi` 不存在或 `import torch_npu` 失败，请先修复驱动、CANN、Torch
与 Torch-NPU 的版本匹配问题。

## 安装 Transformers

```shell #test id="install-transformers"
python -m pip install -q -U transformers accelerate
python -c "import accelerate, transformers; print('transformers', transformers.__version__); print('accelerate', accelerate.__version__)"
```

```shell #test-result id="install-transformers" fuzzy="xxx"
transformers xxx
accelerate xxx
```

## 运行文本生成

`Accelerator().device` 会自动选择当前可用的 NPU。先看交互式示例：

```pycon
>>> import os
>>> from accelerate import Accelerator
>>> from transformers import pipeline
>>>
>>> device = Accelerator().device
>>> pipe = pipeline(
...     "text-generation",
...     model=os.environ.get("QUICK_START_MODEL", "Qwen/Qwen2.5-1.5B"),
...     device=device,
... )
>>> result = pipe("The secret to baking a good cake is ", max_new_tokens=16)
>>> print(result[0]["generated_text"])
The secret to baking a good cake is ...
```

也可以用下面的脚本一键运行：

```shell #test id="transformers-pipeline"
python - <<'PY'
import os
from accelerate import Accelerator
from transformers import pipeline
device = Accelerator().device
pipe = pipeline(
	"text-generation",
	model=os.environ.get("QUICK_START_MODEL", "Qwen/Qwen2.5-1.5B"),
	device=device,
)
result = pipe("The secret to baking a good cake is ", max_new_tokens=16)
print(result[0]["generated_text"])
PY
```

```shell #test-result id="transformers-pipeline"
The secret to baking a good cake is ...
```
