# Liger-Kernel

Liger Kernel 提供面向大语言模型训练的 Triton 融合算子。它可以替换模型中的部分计算模块，以提高训练吞吐、减少显存占用。本文展示在昇腾npu使用Liger Kernel。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品，并按需完成物理机或容器内的设备挂载。

### 基础软件

在运行本文档示例之前，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）

本文档示例使用 Python 3.12、CANN 9.1.0。

本文档配套镜像：

```text
swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann:9.1.0-910b-ubuntu22.04-py3.12
```

## 加载 CANN 环境

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 安装 Ascend NPU 所需依赖

安装torch、torch_npu、triton-ascend：

```shell #test-setup
pip install torch==2.9.0 torch_npu==2.9.0.post2 --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi
pip install "triton-ascend==3.2.2" --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple
```

查看安装的依赖版本，执行下面的python代码：

```python #test id="install-ascend-deps"
from importlib.metadata import version

import torch
import torch_npu
import triton.backends.ascend

print('torch version:', torch.__version__)
print('torch_npu version:', torch_npu.__version__)
print('triton-ascend version:', version('triton-ascend'))
```

输出结果如下：

```shell #test-result id="install-ascend-deps"
torch version: 2.9.0+cpu
torch_npu version: 2.9.0.post2
triton-ascend version: 3.2.2
```

## 安装 Liger Kernel

安装最新发布的 Liger Kernel：

```shell #test-setup
pip install --upgrade liger-kernel
```

查看安装的 Liger-Kernel 版本，执行下面python代码：

```python #test id="install-liger"
from importlib.metadata import version

import liger_kernel

print('liger-kernel', version('liger-kernel'))
```

输出结果如下：

```shell #test-result id="install-liger" fuzzy='xxx'
liger-kernel xxx
```

```{admonition} Note
:class: note
`xxx` 表示实际安装的 Liger Kernel 版本号。
```

## 用法示例

### 方式一：自动 Patch

安装模型加载依赖：

```shell #test-setup
pip install "transformers==5.24.1" "modelscope==1.37.0"
```

`AutoLigerKernelForCausalLM` 会在加载受支持的模型时，自动将部分模型算子替换为 Liger 的优化实现。用python执行以下代码：

```python #test id="auto-patch"
import torch
import torch_npu
from modelscope import snapshot_download
from transformers import AutoTokenizer
from liger_kernel.transformers import AutoLigerKernelForCausalLM

model_path = snapshot_download('Qwen/Qwen3-0.6B')
# 加载模型时自动应用默认 Liger 算子
model = AutoLigerKernelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to('npu').train()
batch = AutoTokenizer.from_pretrained(model_path)('Liger Kernel 加速模型训练。', return_tensors='pt').to('npu')
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
for _ in range(3):
    optimizer.zero_grad()
    loss = model(**batch, labels=batch['input_ids']).loss
    assert torch.isfinite(loss).item()
    loss.backward()
    optimizer.step()

print('device:', next(model.parameters()).device.type)
print('loss:', f'{loss.item():.4f}')
```
输出结果如下：
```shell #test-result id="auto-patch" fuzzy='...' fuzzy='xxx'
...
device: npu
loss: xxx
```

```{admonition} Note
:class: note
`xxx` 表示最后一步计算得到的训练损失。
```

### 方式二：指定模型 Patch

指定模型的 Patch API 会在模型创建前替换选定的算子，控制哪些计算使用 Liger 实现、哪些保持原样。用 Python 执行以下代码：

```python #test id="model-patch"
import torch
import torch_npu
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from liger_kernel.transformers import apply_liger_kernel_to_qwen3

# 加载前选择要替换的算子；两种交叉熵实现不能同时启用
apply_liger_kernel_to_qwen3(rope=True, rms_norm=True, swiglu=True,
                            cross_entropy=True, fused_linear_cross_entropy=False)
model_path = snapshot_download('Qwen/Qwen3-0.6B')
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to('npu').train()
batch = AutoTokenizer.from_pretrained(model_path)('Liger Kernel 加速模型训练。', return_tensors='pt').to('npu')
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
for _ in range(3):
    optimizer.zero_grad()
    loss = model(**batch, labels=batch['input_ids']).loss
    assert torch.isfinite(loss).item()
    loss.backward()
    optimizer.step()

print('device:', next(model.parameters()).device.type)
print('loss:', f'{loss.item():.4f}')
```
输出结果如下：
```shell #test-result id="model-patch" fuzzy='...' fuzzy='xxx'
...
device: npu
loss: xxx
```

```{admonition} Note
:class: note
`xxx` 表示最后一步计算得到的训练损失。
```

### 方式三：自行组合算子

直接使用 Liger 提供的归一化和损失模块，可以在自定义模型中组合算子并控制计算流程。用 Python 执行以下代码：

```python #test id="kernel-example"
import torch
import torch_npu
from liger_kernel.transformers import LigerRMSNorm, LigerFusedLinearCrossEntropyLoss

hidden_size = 128
vocab_size = 256
# 直接创建 Liger 模块，自行组合计算流程
norm = LigerRMSNorm(hidden_size).npu()
loss_fn = LigerFusedLinearCrossEntropyLoss()
x = torch.randn(4, 16, hidden_size, dtype=torch.bfloat16, device='npu', requires_grad=True)
target = torch.randint(vocab_size, (4, 16), device='npu')
weight = torch.randn(vocab_size, hidden_size, dtype=torch.bfloat16, device='npu', requires_grad=True)
optimizer = torch.optim.SGD([weight, *norm.parameters()], lr=0.01)

for _ in range(3):
    optimizer.zero_grad()
    x.grad = None
    # 将 RMSNorm 输出交给融合交叉熵，展平批次和序列维
    hidden = norm(x)
    loss = loss_fn(weight, hidden.reshape(-1, hidden_size), target.reshape(-1))
    assert torch.isfinite(loss).item()
    loss.backward()
    optimizer.step()

print('device:', x.device.type)
print('loss:', f'{loss.item():.4f}')
```
输出结果如下：
```shell #test-result id="kernel-example" fuzzy='xxx'
device: npu
loss: xxx
```

```{admonition} Note
:class: note
`xxx` 表示最后一步计算得到的训练损失。
```

## 训练框架集成

Liger Kernel 已集成到多个训练框架，在 NPU 上可通过框架配置启用：

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)：训练参数中设置 `--use_liger_kernel true`。
- [VeOmni](https://github.com/ByteDance-Seed/VeOmni)：模型 Patch 中启用 Liger Kernel。
- [verl](https://github.com/volcengine/verl)：配置 `model.use_liger=true`。

## 外部链接

- [Liger Kernel 官方仓库](https://github.com/linkedin/Liger-Kernel)
- [Liger Kernel 官方文档](https://linkedin.github.io/Liger-Kernel/)
