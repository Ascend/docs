# bitsandbytes

在单卡昇腾上安装 bitsandbytes，用默认后端完成一次 NF4 `Linear4bit` 前向。

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**）。本文示例为单卡。

### 软件


| 类别           | 要求                                              |
| ------------ | ----------------------------------------------- |
| CANN         | toolkit + 驱动固件已安装，并可 `source set_env.sh`        |
| Python       | 3.12                                            |
| PyTorch      | `torch==2.9.0` 与 `torch_npu==2.9.0.post2`，见下文安装 |
| bitsandbytes | 从 PyPI 安装发布版，见下文                                |


配套镜像：`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`。

阅读本文前，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备好 CANN 与驱动。

## 1. 加载 CANN 环境

新开终端后 CANN 变量不会自动生效。常见容器里 `npu-smi` 在 `/usr/local/sbin`，需要把该目录加入 `PATH`。

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
```



## 2. 检查环境是否就绪



### 2.1 确认 NPU 在线

下面确认驱动能看到设备。表格中的功耗、HBM 占用每次不同，不必与样例逐字一致。

```shell
npu-smi info
```

如果 `npu-smi` 找不到，回到 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 检查驱动与设备挂载。

### 2.2 确认工具可用

下面确认 CANN 已加载，并且 `npu-smi` 与 `python` 都在 `PATH` 里。

```shell #test id="check-tools"
test -n "$ASCEND_HOME_PATH"
command -v npu-smi
python --version
```

输出结果如下：

```shell #test-result id="check-tools"
...
Python 3.12...
```



## 3. 安装 PyTorch NPU 栈

昇腾上的 `torch_npu` 要从华为 PyPI 额外索引安装，并钉死与 CANN 匹配的版本。`numpy` 和 `pyyaml` 也要一起装：`torch_npu` 的 wheel 没有声明这两项依赖，缺了会在显式 `import torch_npu` 之前失败。

```shell #test id="install-torch"
python -m pip install --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi \
  torch==2.9.0 torch_npu==2.9.0.post2 numpy pyyaml
python -c "import numpy, yaml, torch, torch_npu; print('torch', torch.__version__); print('torch_npu', torch_npu.__version__); print('npu_available', torch.npu.is_available())"
```

输出结果如下：

```shell #test-result id="install-torch"
...
torch 2.9.0...
torch_npu 2.9.0.post2
npu_available True
```

`npu_available` 必须是 `True`。`False` 时不要继续，先查 CANN、驱动和可见设备。

## 4. 安装 bitsandbytes

将 `<UPSTREAM_REF>` 换成目标 PyPI 版本号（撰写时最新正式版是 `0.50.2`）。

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

```shell #test id="install-bnb" load="upstream_ref>>UPSTREAM_REF"
python -m pip install bitsandbytes==<UPSTREAM_REF>
python -c "import bitsandbytes as bnb; import bitsandbytes.cextension as ce; print('bitsandbytes', bnb.__version__); print('BNB_BACKEND', ce.BNB_BACKEND); print('lib', type(ce.lib).__name__)"
```

输出结果如下：

```shell #test-result id="install-bnb"
...
bitsandbytes ...
BNB_BACKEND CPU
lib BNBNativeLibrary
```



## 5. 在 NPU 上做一次 NF4 Linear4bit 前向

下面在 CPU 上构造 `Linear4bit(64, 32)`，搬到 `npu:0` 量化，再用 float16 输入做一次前向。进程须退出码为 0，且 `out.device` 必须是 `npu:0`；若打印 `cpu`，那是静默回退，视为失败。

保存为 `nf4_forward.py`：

```python
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

<!--
```shell #test-setup
cat > nf4_forward.py <<'PY'
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
PY
```
-->

运行：

```shell #test id="nf4-forward"
python nf4_forward.py
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

