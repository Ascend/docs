# ColossalAI

本文在单卡昇腾 NPU 上安装 [ColossalAI](https://github.com/hpcaitech/ColossalAI)，并用 `Booster` + `TorchDDPPlugin` 对 [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) 做一步训练。

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**），单卡。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit 与驱动已安装，并能 `source set_env.sh` |
| Python | 3.12 |
| PyTorch | `torch==2.9.0` 与 `torch_npu==2.9.0.post2`，见第 3 节 |
| ColossalAI | `colossalai==0.5.0 --no-deps`，见第 4 节 |
| 模型 | [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) |

**配套机器**：Atlas 900 A2 PODc（Ascend 910B4）。**配套镜像**：`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`。

---

## 1. 加载 CANN 环境

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
export PYTHONNOUSERSITE=1
```

---

## 2. 检查环境

确认 NPU 在线；找不到 `npu-smi` 时参阅 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)。

```shell
npu-smi info
```

检查 Python 版本：

```shell #test id="check-py"
python --version
```

输出结果如下：

```shell #test-result id="check-py" fuzzy="xxx"
Python 3.12.xxx
```

---

## 3. 安装 PyTorch NPU 栈

从华为昇腾索引安装与 CANN 9.1.0 匹配的 `torch==2.9.0` / `torch_npu==2.9.0.post2`，并一并装上 `numpy`、`pyyaml`。`npu_available` 须为 `True`。

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

---

## 4. 安装 ColossalAI

不要写 `pip install colossalai`：上游把 torch 钉在 `<=2.5.1`，会换掉第 3 节的 NPU 栈。先装 `colossalai==0.5.0 --no-deps`，再装 Booster 导入所需的 `transformers==4.51.3`、`peft`、`galore_torch`、`bitsandbytes`、`einops`。

```shell #test id="install-colossalai"
python -m pip install colossalai==0.5.0 --no-deps
python -m pip install transformers==4.51.3 peft galore_torch bitsandbytes einops
python -c "import torch, torch_npu, colossalai; from colossalai.accelerator import get_accelerator; print('torch', torch.__version__); print('colossalai', colossalai.__version__); acc = get_accelerator(); print('accel_name', acc.name); print('accel_device', acc.get_current_device()); print('npu_available', torch.npu.is_available())"
```

输出结果如下：

```shell #test-result id="install-colossalai"
...
torch 2.9.0...
colossalai 0.5.0
accel_name npu
accel_device npu:0
npu_available True
```

`accel_name` 为 `npu`、`accel_device` 为 `npu:0` 只说明设备探测成功。pip 提示 `torch<=2.5.1` 冲突是预期现象。

---

## 5. 在 NPU 上做一步训练

下面用 `launch` 与 `Booster(plugin=TorchDDPPlugin())` 包住 Qwen2.5-0.5B，做一次前向、反向和 `optimizer.step()`。`world_size=1` 为单进程；首次验证用 `bfloat16` 降低显存。随机种子固定为 `42`，并打印这一步的 `loss`。

工作目录为 `/root/colossalai-qs`：

```shell
mkdir -p /root/colossalai-qs
```

保存为 `/root/colossalai-qs/train_one_step.py`：

```python
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

<!--
```shell #test-setup
mkdir -p /root/colossalai-qs
cat > /root/colossalai-qs/train_one_step.py <<'PY'
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
PY
```
-->

```shell #test id="train"
PYTHONHASHSEED=42 python /root/colossalai-qs/train_one_step.py
```

输出结果如下：

```shell #test-result id="train"
...
accel_name npu
accel_device npu:0
boosted_param_device npu:0
loss 5.531600
```

导入时可能出现 `tensornvme`、`apex` 警告，昇腾上可忽略。`boosted_param_device` 为 `npu:0` 表示参数已放到 NPU。`PYTHONHASHSEED=42` 与脚本里的 `SEED = 42` 一起冻结随机性；`loss 5.531600` 是这一步训练的结果。

---

## 故障排查

| 现象 | 可能原因 | 建议 |
| --- | --- | --- |
| `import torch` 报缺 `numpy` 或 `yaml` | `torch_npu` 未声明这两项依赖 | 与 torch 栈一起安装 `numpy` `pyyaml` |
| `torch.npu.is_available()` 为 `False` | 未 `source set_env.sh`，或设备未挂进容器 | 重做第 1–2 节 |
| `accel_name` 为 `cpu` | `torch_npu` 没装上，或 NPU 不可用 | 重做第 3 节 |
| `accel_name` 为 `cuda` | 当前进程里 CUDA 版 torch 可用 | 卸掉 CUDA 版 torch |
| pip 把 `torch` 降到 `2.5.1` 或更低 | 安装 ColossalAI 时没加 `--no-deps` | 卸掉后按第 3–4 节重装 |
| `ModuleNotFoundError: galore_torch`、`peft` 或 `einops` | 只装了 `--no-deps` 的 ColossalAI | 按第 4 节把导入期依赖装上 |
| `boosted_param_device` 为 `cpu` 但退出码 0 | 模型没放到 NPU | 检查可见设备和 `torch.npu.is_available()` |
| `Address already in use` | 端口 `29599` 被占用 | 换一个端口，或结束占用该端口的进程 |

## 外部链接

- GitHub：[hpcaitech/ColossalAI](https://github.com/hpcaitech/ColossalAI)
- 文档中心：[Colossal-AI Docs](https://colossalai.readthedocs.io/en/latest/)

