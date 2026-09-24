# flash-linear-attention

在单张昇腾 NPU 上安装 flash-linear-attention，并运行 `GatedDeltaNet` 的
前向与反向计算。完成后，可以确认 Triton-Ascend 能识别 NPU，且模型输出
和输入梯度均正常。

## 前置条件

### 硬件

- Atlas 800T / 900 A2 训练系列；
- 至少一张可用的 Ascend 910B NPU；
- 物理机或容器已正确配置驱动和设备。

### 基础软件

在运行本文档之前，机器上需要已经安装并可用：

- Linux aarch64 操作系统；
- 可用的 Python 环境；
- 可用的 CANN toolkit 和驱动；
- `npu-smi` 能正常显示 NPU 设备。

CANN 安装可参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)。Torch、Torch-NPU、torchvision 和 Triton-Ascend 由目标 release 的 `[npu]` extra 安装。

### 本文档示例使用的版本

本文以最新正式 release 为安装目标。以下是基于 `v0.5.2`
[NPU 依赖声明](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/pyproject.toml)
验证过的环境组合，使用的镜像为
`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-910b-ubuntu22.04-py3.11`。

| 组件 | 版本 |
| --- | --- |
| 操作系统 | Ubuntu 22.04，Linux aarch64 |
| Python | 3.11 |
| CANN | 9.0.0 |
| flash-linear-attention | 最新 release |
| torch | 2.7.1+cpu（`v0.5.2` 运行时回显） |
| torch_npu | 2.7.1.post4 |
| torchvision | 0.22.1 |
| triton-ascend | 3.2.1（分发包版本） |
| NPU | Ascend 910B × 1 |

```{admonition} Note
:class: note
不同 release 的 NPU 依赖可能不同。安装时以所选 release 的 `[npu]` 依赖为准，
不要混用 `main` 分支的版本要求；如需升级 CANN，请先核对相应版本的兼容性。
```

### 检查前置是否满足

```shell #test id="check-cann"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
test -n "$ASCEND_HOME_PATH"
command -v npu-smi >/dev/null
printf 'CANN ready\n'
```

输出结果如下：

```text #test-result id="check-cann"
CANN ready
```

## 安装 flash-linear-attention

从 [Releases](https://github.com/fla-org/flash-linear-attention/releases) 选择
最新正式版本，将 `<UPSTREAM_REF>` 替换为对应的标签。安装命令会读取该版本
的 `[npu]` 依赖，安装匹配的 Torch、Torch-NPU、torchvision 与 Triton-Ascend。

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

```shell #test id="install-fla" load="upstream_ref>>UPSTREAM_REF"
test ! -e flash-linear-attention
git clone https://github.com/fla-org/flash-linear-attention.git
cd flash-linear-attention
git checkout <UPSTREAM_REF>
python -m pip install -q -U pip setuptools wheel
python -m pip install -q pybind11 cmake attrs sympy pyyaml scipy decorator einops
python -m pip install -q ".[npu]" \
  --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple
python -c "import fla; print('fla', fla.__version__)"
```

输出结果如下：

```text #test-result id="install-fla" fuzzy="xxx"
fla xxx
```

## 验证 Ascend NPU backend

flash-linear-attention 通过 Triton runtime 识别 `npu` backend，并将 `fla.utils.IS_NPU` 设置为 `True`。

```python #test id="check-npu"
import torch
import torch_npu
import triton

from fla.utils import IS_NPU, device_platform

assert torch.npu.is_available()
assert IS_NPU
assert device_platform == "npu"

print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("triton", triton.__version__)
print("device_platform", device_platform)
print("npu_available", torch.npu.is_available())
```

输出结果如下：

```text #test-result id="check-npu" fuzzy="xxx"
torch xxx+cpu
torch_npu xxx
triton xxx
device_platform npu
npu_available True
```

## 使用样例

### GatedDeltaNet 前向与反向

创建一个较小的 `GatedDeltaNet` 模型，在 NPU 上完成前向计算和反向传播。
示例会检查输出形状，并确认输出和输入梯度均为有限值。

```python #test id="gdn-forward-backward"
import torch
import torch_npu

from fla.layers import GatedDeltaNet
from fla.utils import IS_NPU, device_platform

assert IS_NPU
assert device_platform == "npu"
torch.manual_seed(42)

layer = GatedDeltaNet(
    hidden_size=512,
    head_dim=64,
    num_heads=6,
    expand_v=2,
    mode="chunk",
).to(device="npu", dtype=torch.bfloat16).train()

x = torch.randn(
    1,
    128,
    512,
    device="npu",
    dtype=torch.bfloat16,
    requires_grad=True,
)
y = layer(x)[0]
loss = y.float().square().mean()
loss.backward()
torch.npu.synchronize()

assert y.shape == (1, 128, 512)
assert torch.isfinite(y).all()
assert x.grad is not None
assert torch.isfinite(x.grad).all()

print("device", y.device.type)
print("output_shape", tuple(y.shape))
print("forward_finite", torch.isfinite(y).all().item())
print("backward_finite", torch.isfinite(x.grad).all().item())
```

输出结果如下：

```text #test-result id="gdn-forward-backward"
device npu
output_shape (1, 128, 512)
forward_finite True
backward_finite True
```

## 说明

- `device npu` 表示输出位于昇腾 NPU；
- `forward_finite True` 和 `backward_finite True` 表示本次前向与反向计算没有出现非有限值；
- 本示例使用单卡，不涉及多卡分布式训练。
