# Unsloth

Unsloth 是用于大语言模型微调的训练库，通过融合内核缩短 LoRA 训练时间。本文在单卡昇腾 NPU 上完成两段 bfloat16 LoRA 监督微调：先关闭 torch.compile，训练 Qwen3.5-0.8B；再训练 Qwen3-0.6B，走 Unsloth 的 Triton 融合内核。

## 前置条件

### 硬件

Atlas 900 A2 训练系列产品，单卡。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit 与驱动已安装，并能 `source set_env.sh`。版本按 [昇腾软件配套清单](https://www.hiascend.com/developer/download/compatibility) 选择 |
| Python | 落在官方配套表给出的范围内，并满足 Unsloth 下限。当前正式版要求 Python 3.9 至 3.14 |
| PyTorch | `torch==2.10.0` 与 `torch_npu==2.10.0.post7`。配套表见 [CANN 与 PyTorch 配套表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md) |
| Unsloth | 克隆上游最新 release 标签 `<ref>`，安装检出目录里的 `huggingfacenotorch` 扩展 |
| triton-ascend | 从昇腾 PyPI 安装，供 Qwen3 的 Triton 融合内核编译 |
| 模型 | [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)、[Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) |

### 本文验证环境

- 机器类型：Atlas 900 A2，Ascend 910B4，32 GB × 1，Ubuntu 22.04
- 配套镜像：swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| torch | 2.10.0+cpu |
| torch_npu | 2.10.0.post7 |
| unsloth | 2026.11.4 |
| torchvision | 0.25.0+cpu |
| triton-ascend | 3.6.0.post2 |

---

## 1. 加载 CANN 环境

在终端加载 CANN 环境：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:/usr/local/bin:$PATH
```

---

## 2. 检查环境

确认 NPU 在线。`npu-smi` 找不到时参阅 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)。

```shell
npu-smi info
```

输出结果类似如下：

```text
+------------------------------------------------------------------------------------------------+
| npu-smi 25.5.1                   Version: 25.5.1                                               |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B4               | OK            | 89.7        39                0    / 0             |
| 0                         | 0000:C1:00.0  | 0           0    / 0          2888 / 32768         |
+===========================+===============+====================================================+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| No running processes found in NPU 0                                                            |
+===========================+===============+====================================================+
```

---

## 3. 安装 PyTorch NPU 栈

安装 triton-ascend 3.6 能编译内核的 torch 与 torch_npu。更新的 torch 2.14 要求 C++20，而这份 triton-ascend 按 C++17 编译，内核会失败：

```shell #test id="install-torch"
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  torch==2.10.0 torch_npu==2.10.0.post7 numpy pyyaml
python -c "import numpy, yaml, torch, torch_npu; print('torch', torch.__version__); print('torch_npu', torch_npu.__version__); print('npu_available', torch.npu.is_available())"
```

完整输出较长，其中应包含：

```shell #test-result id="install-torch"
...
torch 2.10.0+cpu
torch_npu 2.10.0.post7
npu_available True
```


---

## 4. 安装 Unsloth

克隆上游最新 release 标签 `<ref>`，安装检出目录里的 Hugging Face 训练依赖和 bitsandbytes。Qwen3.5 要加载视觉处理器，再从 PyTorch CPU 轮子源安装与 torch 2.10.0 配套的 torchvision。

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

```shell #test id="install-unsloth" load="upstream_ref>>ref"
mkdir -p /root/unsloth-qs
git clone --depth 1 --branch <ref> https://github.com/unslothai/unsloth.git /root/unsloth-qs/unsloth
python -m pip install --no-build-isolation "/root/unsloth-qs/unsloth[huggingfacenotorch]" bitsandbytes
python -m pip install --force-reinstall --no-deps \
  --index-url https://download.pytorch.org/whl/cpu \
  torchvision==0.25.0
python -c "import unsloth, torchvision; print('unsloth', unsloth.__version__); print('torchvision', torchvision.__version__)"
```


完整输出较长，其中应包含：

```shell #test-result id="install-unsloth" fuzzy="..." fuzzy="xxx"
...
unsloth xxx
torchvision 0.25.0+cpu
```

```{admonition} Note
:class: note
`<ref>` 替换为 Unsloth 当前的最新 release 标签。
```

---

## 5. 安装 triton-ascend

卸掉 Unsloth 带进来的社区 Triton，再安装 triton-ascend，并补上导入后端需要的 pybind11：

```shell #test id="install-triton-ascend"
python -m pip uninstall -y triton
python -m pip install --force-reinstall --no-deps \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  triton-ascend
python -m pip install pybind11
python -c "import triton.backends.ascend; print('triton_backend ascend')"
```

<!--
完整输出较长，其中应包含：

```shell #test-result id="install-triton-ascend"
...
triton_backend ascend
```
-->

---

## 6. 关闭 torch.compile，微调 Qwen3.5-0.8B

本示例的工作目录为 `/root/unsloth-qs`。下面脚本关闭 torch.compile，以 bfloat16 加载 [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)，做 3 步 LoRA。`seed` 为 42。

```shell
mkdir -p /root/unsloth-qs
```

训练脚本如下，保存为 `/root/unsloth-qs/train_qwen35.py`：

```python
import os

os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
import torch_npu
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3.5-0.8B",
    max_seq_length=128,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=42,
)
print("model_type", model.config.model_type)
print("device", next(model.parameters()).device)
print("compile_disable", os.environ["UNSLOTH_COMPILE_DISABLE"])

dataset = Dataset.from_list([
    {"text": "Instruction: Translate to English.\nInput: hello\nOutput: Hello"},
    {"text": "Instruction: Name a color.\nInput: sky\nOutput: Blue"},
    {"text": "Instruction: Add the numbers.\nInput: 2 and 3\nOutput: 5"},
    {"text": "Instruction: Say yes or no.\nInput: ready\nOutput: Yes"},
])
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="/root/unsloth-qs/outputs-qwen35",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=3,
        learning_rate=2e-4,
        logging_steps=1,
        bf16=True,
        optim="adamw_torch",
        seed=42,
        report_to="none",
        dataset_text_field="text",
        max_length=128,
        packing=False,
        dataset_num_proc=1,
    ),
)
stats = trainer.train()
print("train_loss", stats.training_loss)
```

<!--
```shell #test-setup
mkdir -p /root/unsloth-qs
cat > /root/unsloth-qs/train_qwen35.py << 'PY'
import os

os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
import torch_npu
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3.5-0.8B",
    max_seq_length=128,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=42,
)
print("model_type", model.config.model_type)
print("device", next(model.parameters()).device)
print("compile_disable", os.environ["UNSLOTH_COMPILE_DISABLE"])

dataset = Dataset.from_list([
    {"text": "Instruction: Translate to English.\nInput: hello\nOutput: Hello"},
    {"text": "Instruction: Name a color.\nInput: sky\nOutput: Blue"},
    {"text": "Instruction: Add the numbers.\nInput: 2 and 3\nOutput: 5"},
    {"text": "Instruction: Say yes or no.\nInput: ready\nOutput: Yes"},
])
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="/root/unsloth-qs/outputs-qwen35",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=3,
        learning_rate=2e-4,
        logging_steps=1,
        bf16=True,
        optim="adamw_torch",
        seed=42,
        report_to="none",
        dataset_text_field="text",
        max_length=128,
        packing=False,
        dataset_num_proc=1,
    ),
)
stats = trainer.train()
print("train_loss", stats.training_loss)
PY
```
-->

运行训练：

```shell #test id="train-qwen35"
python /root/unsloth-qs/train_qwen35.py 2>&1
```

完整输出较长，其中应包含：

```text #test-result id="train-qwen35"
...Fast Qwen3_5 patching...
...Ascend910B...
...Ascend NPU...
...device npu:0...
...compile_disable 1...
...{'loss': '1.833'...
...{'loss': '3.048'...
...{'loss': '2.395'...
...'train_loss': '2.426'...
...train_loss 2.4255578915278115
```

---

## 7. 用 Triton 融合内核微调 Qwen3-0.6B

下面脚本以 bfloat16 加载 [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)，做 3 步 LoRA，并关闭 torch.compile。模型层走 Unsloth 的 Triton 融合内核。`seed` 为 42。

训练脚本如下，保存为 `/root/unsloth-qs/train_qwen3.py`：

```python
import os

os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
import torch_npu
import triton.backends.ascend
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-0.6B",
    max_seq_length=128,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=42,
)
print("model_type", model.config.model_type)
print("device", next(model.parameters()).device)
print("triton_backend ascend")
print("compile_disable", os.environ["UNSLOTH_COMPILE_DISABLE"])

dataset = Dataset.from_list([
    {"text": "Instruction: Translate to English.\nInput: hello\nOutput: Hello"},
    {"text": "Instruction: Name a color.\nInput: sky\nOutput: Blue"},
    {"text": "Instruction: Add the numbers.\nInput: 2 and 3\nOutput: 5"},
    {"text": "Instruction: Say yes or no.\nInput: ready\nOutput: Yes"},
])
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="/root/unsloth-qs/outputs-qwen3",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=3,
        learning_rate=2e-4,
        logging_steps=1,
        bf16=True,
        optim="adamw_torch",
        seed=42,
        report_to="none",
        dataset_text_field="text",
        max_length=128,
        packing=False,
        dataset_num_proc=1,
    ),
)
stats = trainer.train()
print("train_loss", stats.training_loss)
```

<!--
```shell #test-setup
mkdir -p /root/unsloth-qs
cat > /root/unsloth-qs/train_qwen3.py << 'PY'
import os

os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
import torch_npu
import triton.backends.ascend
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-0.6B",
    max_seq_length=128,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=42,
)
print("model_type", model.config.model_type)
print("device", next(model.parameters()).device)
print("triton_backend ascend")
print("compile_disable", os.environ["UNSLOTH_COMPILE_DISABLE"])

dataset = Dataset.from_list([
    {"text": "Instruction: Translate to English.\nInput: hello\nOutput: Hello"},
    {"text": "Instruction: Name a color.\nInput: sky\nOutput: Blue"},
    {"text": "Instruction: Add the numbers.\nInput: 2 and 3\nOutput: 5"},
    {"text": "Instruction: Say yes or no.\nInput: ready\nOutput: Yes"},
])
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="/root/unsloth-qs/outputs-qwen3",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=3,
        learning_rate=2e-4,
        logging_steps=1,
        bf16=True,
        optim="adamw_torch",
        seed=42,
        report_to="none",
        dataset_text_field="text",
        max_length=128,
        packing=False,
        dataset_num_proc=1,
    ),
)
stats = trainer.train()
print("train_loss", stats.training_loss)
PY
```
-->

运行训练：

```shell #test id="train-qwen3"
python /root/unsloth-qs/train_qwen3.py 2>&1
```

完整输出较长，其中应包含：

```text #test-result id="train-qwen3"
...Fast Qwen3 patching...
...Ascend910B...
...Ascend NPU...
...device npu:0...
...triton_backend ascend...
...compile_disable 1...
...{'loss': '2.351'...
...{'loss': '4.066'...
...{'loss': '2.821'...
...'train_loss': '3.079'...
...train_loss 3.0792861779530845
```

---

## 相关链接

LoRA 目标层、数据格式和其余模型的用法与社区文档相同。

- 项目仓库：[unsloth](https://github.com/unslothai/unsloth)
- 社区文档：[Unsloth Docs](https://docs.unsloth.ai/)
