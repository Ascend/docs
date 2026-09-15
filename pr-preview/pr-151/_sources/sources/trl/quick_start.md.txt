# TRL

TRL 用同一套 `Trainer` / `Config` API 覆盖 SFT / DPO / GRPO / PPO 等后训练方法。本示例在单卡昇腾 NPU 上，用同一个 Qwen2.5-0.5B-Instruct 模型先跑通最小 SFT LoRA，再换成 `DPOTrainer` 跑通偏好优化 DPO LoRA，并验证两种方法的产物。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，并按需完成物理机或容器内的设备挂载。

### 基础软件

在运行本文档示例之前，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 根据 CANN 版本安装匹配的 `torch_npu`（参考 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)）

### 检查设备

确认能看到 NPU 设备：

```shell
npu-smi info
```

确认命令能够列出 NPU 设备，且设备的 `Health` 状态为 `OK`。

```{admonition} Note
:class: note
如果 `npu-smi` 不存在，请回到 [Ascend 官方快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 补装驱动
```

## 安装 TRL 和示例依赖

使用 pip 安装 TRL、PEFT 和 ModelScope，安装完成后验证这些软件包可以正常导入：

```shell #test id="install-trl"
python -m pip install "trl[peft]" "transformers>=4.56.2,<5.0" "modelscope==1.37.0"
python -c "import trl, peft, modelscope; print('TRL environment ready')"
```

输出结果如下：

```shell #test-result id="install-trl"
...
TRL environment ready
```

## 使用样例：最小 SFT LoRA 后训练

用 ModelScope 数据集 `HuggingFaceH4/ultrafeedback_binarized` 的 SFT 子集对 Qwen2.5-0.5B-Instruct 做 5 步 LoRA SFT。模型由脚本内的 `snapshot_download` 首次运行时自动下载到默认缓存（约 1 GB），数据集经 ModelScope 自动下载；`SFTTrainer` 通过 `peft_config` 注入 LoRA 适配器，底座权重冻结、只训练新注入的低秩矩阵；训练完成后把适配器保存到 `output/trl-sft-lora`。

```shell #test id="sft-lora"
python << 'PY'
import os
import shutil
import torch
import torch_npu
from datasets import load_dataset
from modelscope import snapshot_download
from peft import LoraConfig, TaskType
from trl import SFTConfig, SFTTrainer

print("TRL_SFT_BEGIN")

ds_path = snapshot_download(
    'HuggingFaceH4/ultrafeedback_binarized', repo_type='dataset',
)
data_dir = './ultrafeedback_sft'
if os.path.isdir(data_dir):
    shutil.rmtree(data_dir)
os.makedirs(data_dir, exist_ok=True)
for name in os.listdir(os.path.join(ds_path, 'data')):
    if name.startswith('train_sft-'):
        shutil.copy2(os.path.join(ds_path, 'data', name), data_dir)
train_dataset = load_dataset(
    'parquet', data_files=os.path.join(data_dir, 'train_sft-*.parquet'),
    split='train',
).select_columns(['messages'])

model = snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=LoraConfig(r=8, lora_alpha=32, task_type=TaskType.CAUSAL_LM),
    args=SFTConfig(
        output_dir="output/trl-sft-lora",
        max_steps=5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        max_length=512,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        model_init_kwargs={"dtype": torch.bfloat16},
    ),
)
print("model device:", next(trainer.model.parameters()).device)
trainer.train()
trainer.save_model("output/trl-sft-lora")
print("TRL_SFT_DONE")
PY
```

输出结果类似如下（训练日志走 stderr，stdout 只保留首尾标记）：

```shell #test-result id="sft-lora"
...
TRL_SFT_DONE
```

## 切换方法：偏好优化 DPO LoRA

同一个模型与 LoRA 配置，把 `SFTTrainer` / `SFTConfig` 换成 `DPOTrainer` / `DPOConfig` 就是偏好优化：ModelScope 数据集 `HuggingFaceH4/ultrafeedback_binarized` 的 `prompt` / `chosen` / `rejected` 三段对话让模型更倾向 `chosen` 而非 `rejected` 的回答。这里跑 3 步 DPO LoRA，产物保存到 `output/trl-dpo-lora`。

```shell #test id="dpo-lora"
python << 'PY'
import os
import shutil
import torch
import torch_npu
from datasets import load_dataset
from modelscope import snapshot_download
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

print("TRL_DPO_BEGIN")

ds_path = snapshot_download(
    'HuggingFaceH4/ultrafeedback_binarized', repo_type='dataset',
)
data_dir = './ultrafeedback_prefs'
if os.path.isdir(data_dir):
    shutil.rmtree(data_dir)
os.makedirs(data_dir, exist_ok=True)
for name in os.listdir(os.path.join(ds_path, 'data')):
    if name.startswith('train_prefs-'):
        shutil.copy2(os.path.join(ds_path, 'data', name), data_dir)
train_dataset = load_dataset(
    'parquet', data_files=os.path.join(data_dir, 'train_prefs-*.parquet'),
    split='train',
)

# prompt 列是纯字符串，chosen / rejected 是 messages 列表；把 prompt 转成
# 单条 user 消息即可让 DPOTrainer 按 conversational 格式处理
def to_conversational(example):
    example['prompt'] = [{'role': 'user', 'content': example['prompt']}]
    return example

train_dataset = train_dataset.map(to_conversational)

model_path = snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path)

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    peft_config=LoraConfig(r=8, lora_alpha=32, task_type=TaskType.CAUSAL_LM),
    args=DPOConfig(
        output_dir="output/trl-dpo-lora",
        max_steps=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        max_length=512,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    ),
)
print("model device:", next(trainer.model.parameters()).device)
trainer.train()
trainer.save_model("output/trl-dpo-lora")
print("TRL_DPO_DONE")
PY
```

输出结果类似如下（训练日志走 stderr，stdout 只保留首尾标记）：

```shell #test-result id="dpo-lora"
...
TRL_DPO_DONE
```

## 结果验证

检查两个输出目录中的 LoRA 适配器产物：`adapter_config.json`（LoRA 配置）与 `adapter_model.safetensors`（适配器权重）。

```shell #test id="verify-output"
ls output/trl-sft-lora/adapter_config.json output/trl-sft-lora/adapter_model.safetensors
```

输出结果如下：

```shell #test-result id="verify-output"
output/trl-sft-lora/adapter_config.json
output/trl-sft-lora/adapter_model.safetensors
```

检查 DPO 输出目录的适配器产物：

```shell #test id="verify-dpo"
ls output/trl-dpo-lora/adapter_config.json output/trl-dpo-lora/adapter_model.safetensors
```

输出结果如下：

```shell #test-result id="verify-dpo"
output/trl-dpo-lora/adapter_config.json
output/trl-dpo-lora/adapter_model.safetensors
```

更多方法（GRPO / PPO / Reward / KTO 等）入口形态一致，切换对应的 `Trainer` / `Config` 即可；GRPO 依赖 vLLM 生成，不在本示例运行。更多用法见 [TRL examples](https://github.com/huggingface/trl/tree/main/examples)。
