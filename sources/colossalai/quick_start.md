# ColossalAI

ColossalAI 是面向大模型的训练系统，用 Booster 把模型和优化器接到并行插件上。本文在单卡昇腾 NPU 上安装它，并用 `TorchDDPPlugin` 对 [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) 做一步训练。

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**），单卡。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit 与驱动已安装，并能 `source set_env.sh`。版本按 [昇腾软件配套清单](https://www.hiascend.com/developer/download/compatibility) 选择 |
| Python | 落在官方配套表给出的范围内，并满足 ColossalAI 该版本的 Python 下限；当前正式版要求 `>=3.6` |
| PyTorch | 安装官方当前推荐的 `torch` 与 `torch_npu`，见 [CANN 与 PyTorch 配套表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md) 和 [PyTorch 安装包](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download) |
| ColossalAI | 当前正式版。将 `<ref>` 换成 PyPI 版本号，安装步骤见第 4 节 |
| 模型 | [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) |

### 本文验证环境

本文验证时的环境是 CANN 9.1.0 与 Python 3.12，配套镜像为 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`。本次按官方 CPU 轮子源解析到 `torch 2.12.0+cpu` 与 `torch_npu 2.12.0`。这不是唯一支持组合。

---

## 1. 加载 CANN 环境

加载 CANN，并把 `npu-smi` 所在目录加入 `PATH`。

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:/usr/local/bin:$PATH
```

---

## 2. 检查环境

确认 NPU 在线；找不到 `npu-smi` 时参阅 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)。

```shell
npu-smi info
```

输出类似：

```text
+------------------------------------------------------------------------------------------------+
| npu-smi 25.5.1                   Version: 25.5.1                                               |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 6     910B4               | OK            | 85.6        39                0    / 0             |
| 0                         | 0000:82:00.0  | 0           0    / 0          2888 / 32768         |
+===========================+===============+====================================================+
| 7     910B4               | OK            | 88.6        39                0    / 0             |
| 0                         | 0000:42:00.0  | 0           0    / 0          2890 / 32768         |
+===========================+===============+====================================================+
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| No running processes found in NPU 6                                                            |
+===========================+===============+====================================================+
| No running processes found in NPU 7                                                            |
+===========================+===============+====================================================+
```

---

## 3. 安装 PyTorch NPU 栈

安装当前配套的 `torch_npu`，并带上 `numpy`、`pyyaml`。

```shell #test id="install-torch"
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  torch_npu numpy pyyaml
python -c "import numpy, yaml, torch, torch_npu; print('torch', torch.__version__); print('torch_npu', torch_npu.__version__); print('npu_available', torch.npu.is_available())"
```

<!--
```shell #test-result id="install-torch"
...
torch ...+cpu
torch_npu ...
npu_available True
```
-->

---

## 4. 安装 ColossalAI

先安 <ref> 指定的版本，再安装 Booster 导入时需要的 `transformers==4.51.3`、`peft`、`galore_torch`、`bitsandbytes`、`einops`。将 `<ref>` 换成 PyPI 版本号。

```{note}
当前正式版的安装声明把 torch 限制在 2.5.1 及以下。直接安装会换掉上一节的 NPU 栈。
```

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF#v}"
```
-->

```shell #test id="install-colossalai" load="upstream_ref>>ref"
python -m pip install "colossalai==<ref>" --no-deps
python -m pip install transformers==4.51.3 peft galore_torch bitsandbytes einops
python -c "import torch, torch_npu, colossalai; from colossalai.accelerator import get_accelerator; print('torch', torch.__version__); print('colossalai', colossalai.__version__); acc = get_accelerator(); print('accel_name', acc.name); print('accel_device', acc.get_current_device()); print('npu_available', torch.npu.is_available())"
```

<!--
```shell #test-result id="install-colossalai" load="upstream_ref>>ref"
...
torch ...+cpu
colossalai <ref>
accel_name npu
accel_device npu:0
npu_available True
```
-->

---

## 5. 在 NPU 上做一步训练

单卡、单进程，用 `launch` 与 `Booster(plugin=TorchDDPPlugin())` 做一次前向、反向和 `optimizer.step()`，并打印这一步的 `loss`。

用 python 执行下面的代码。

```python #test id="train"
import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.booster.plugin import TorchDDPPlugin

SEED = 42
set_seed(SEED)
torch.npu.manual_seed(SEED)
torch.npu.manual_seed_all(SEED)

model_id = "Qwen/Qwen2.5-0.5B"
colossalai.launch(rank=0, world_size=1, host="127.0.0.1", port=29599)
acc = get_accelerator()
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16,
)
optimizer = AdamW(model.parameters(), lr=1e-5)
booster = Booster(plugin=TorchDDPPlugin())
model, optimizer, _, _, _ = booster.boost(model, optimizer)
model.train()
set_seed(SEED)
torch.npu.manual_seed(SEED)
torch.npu.manual_seed_all(SEED)
print("accel_name", acc.name)
print("accel_device", acc.get_current_device())
print("boosted_param_device", next(model.parameters()).device)
enc = tokenizer("ColossalAI on Ascend NPU", return_tensors="pt")
enc = {k: v.to(acc.get_current_device()) for k, v in enc.items()}
loss = model(**enc, labels=enc["input_ids"]).loss
print("loss", f"{float(loss.item()):.6f}")
booster.backward(loss, optimizer)
optimizer.step()
optimizer.zero_grad()
```

完整输出较长，其中应包含：

```text #test-result id="train"
...accel_name npu
accel_device npu:0
boosted_param_device npu:0
loss 5.531600
```

---

## 外部链接

一步训练之外的用法，包括张量并行、流水线并行和多机启动，与社区文档相同。

- GitHub：[hpcaitech/ColossalAI](https://github.com/hpcaitech/ColossalAI)
- 社区手册：[Colossal-AI Docs](https://colossalai.readthedocs.io/en/latest/)
