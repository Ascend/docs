# bitsandbytes

bitsandbytes 是 8-bit / 4-bit 量化库。本文在单卡昇腾上安装它，并用默认后端完成一次 NF4 `Linear4bit` 前向。

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列，Ascend **910B**。本文示例为单卡。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit 与驱动已安装，并能 `source set_env.sh`。版本按 [昇腾软件配套清单](https://www.hiascend.com/developer/download/compatibility) 选择 |
| Python | 落在官方配套表给出的范围内，并满足 bitsandbytes 下限；当前正式版要求 `>=3.10` |
| PyTorch | 安装官方当前推荐的 `torch` 与 `torch_npu`，见 [CANN 与 PyTorch 配套表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md) 和 [PyTorch 安装包](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download) |
| bitsandbytes | 从 PyPI 安装最新发布版 |

阅读本文前，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备好 CANN 与驱动。

### 本文验证环境

本文在配套镜像 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12` 上验证，镜像内为 CANN 9.1.0 与 Python 3.12。这不是唯一支持组合。

## 1. 加载 CANN 环境

加载 CANN 环境变量，并把 `npu-smi` 所在目录加入 `PATH`。

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
```

## 2. 检查环境是否就绪

### 2.1 确认 NPU 在线

确认驱动能看到设备。

```shell
npu-smi info
```

输出类似：

```text
+------------------------------------------------------------------------------------------------+
| npu-smi 25.5.2                   Version: 25.5.2                                               |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 5     910B4               | OK            | 89.9        39                0    / 0             |
| 0                         | 0000:41:00.0  | 0           0    / 0          2922 / 32768         |
+===========================+===============+====================================================+
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| No running processes found in NPU 5                                                            |
+===========================+===============+====================================================+
```

如果 `npu-smi` 找不到，回到 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 检查驱动与设备挂载。

### 2.2 确认工具可用

确认 CANN 已加载，并且 Python 可用。

```shell #test id="check-tools"
test -n "$ASCEND_HOME_PATH"
python --version
```

<!--
```shell #test-result id="check-tools"
...
Python ...
```
-->

## 3. 安装 PyTorch NPU 栈

安装 CPU 版 PyTorch、配套的 torch_npu、numpy 和 pyyaml。

```shell #test id="install-torch"
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  torch_npu numpy pyyaml
python -c "import numpy, yaml, torch, torch_npu; print('torch', torch.__version__); print('torch_npu', torch_npu.__version__); print('npu_available', torch.npu.is_available())"
```

完整输出较长，其中应包含：

```shell #test-result id="install-torch"
...
torch ...+cpu
...
npu_available True
```


## 4. 安装 bitsandbytes

安装最新版本。

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

```shell #test id="install-bnb" load="upstream_ref>>UPSTREAM_REF"
python -m pip install --no-build-isolation "bitsandbytes==<UPSTREAM_REF>"
python -c "import bitsandbytes as bnb; import bitsandbytes.cextension as ce; print('bitsandbytes', bnb.__version__); print('BNB_BACKEND', ce.BNB_BACKEND); print('lib', type(ce.lib).__name__)"
```

输出：

```shell #test-result id="install-bnb"
...
bitsandbytes ...
BNB_BACKEND CPU
lib BNBNativeLibrary
```

## 5. 在 NPU 上做一次 NF4 Linear4bit 前向

在 CPU 上构造 `Linear4bit(64, 32)`，搬到 `npu:0` 后做一次 float16 前向。

```python #test id="nf4-forward"
import torch
import torch_npu
import bitsandbytes as bnb

layer = bnb.nn.Linear4bit(
    64, 32, bias=False, compute_dtype=torch.float16, quant_type="nf4",
    compress_statistics=False,
)
layer = layer.to("npu:0")
x = torch.randn(4, 64, dtype=torch.float16, device="npu:0")
out = layer(x)
weight = layer.weight
print("weight.device", weight.device)
print("weight.dtype", weight.dtype)
print("weight.bnb_quantized", weight.bnb_quantized)
print("out.device", out.device)
print("out.shape", tuple(out.shape))
print("out.dtype", out.dtype)
print("NF4 forward on Ascend NPU: OK")
```

输出结果如下：

```shell #test-result id="nf4-forward"
weight.device npu:0
weight.dtype torch.uint8
weight.bnb_quantized True
out.device npu:0
out.shape (4, 32)
out.dtype torch.float16
NF4 forward on Ascend NPU: OK
```

## 6. 更多文档

8-bit 优化器、模型量化和其余模块的用法与社区相同，见 [bitsandbytes 文档](https://huggingface.co/docs/bitsandbytes)。
