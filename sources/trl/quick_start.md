# TRL

TRL 用统一的 `Trainer` / `Config` API 支持多种模型后训练方法。本示例在单卡昇腾 NPU 上，用 Qwen2.5-0.5B-Instruct 分别运行 SFT 和 DPO LoRA。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，并按需完成物理机或容器内的设备挂载。

### 基础软件

在运行本文档示例之前，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 根据 CANN 版本安装匹配的 `torch_npu`（参考 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)）

本文档示例在 Python 3.12、CANN 9.1.0、`torch_npu` 2.9.0.post2 环境下验证通过。

## 加载 CANN 环境

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 安装 TRL

安装 TRL 并查看安装版本：

```shell #test id="install-trl"
python -m pip install trl
python -c "import trl; print('trl', trl.__version__)"
```

输出结果如下，其中 `xxx` 为实际安装的 TRL 版本：

```shell #test-result id="install-trl" fuzzy='...' fuzzy='xxx'
...
trl xxx
```

## 示例一：SFT LoRA 后训练

用 Qwen2.5-0.5B-Instruct 和 ModelScope 的 `HuggingFaceH4/ultrafeedback_binarized` SFT 子集进行 5 步 LoRA 微调，模型与数据集会自动下载，适配器保存到 `output/trl-sft-lora`。

安装示例依赖：

```shell #test-setup
python -m pip install peft "transformers>=4.56.2,<5.0" datasets "modelscope==1.37.0"
```
运行 SFT 训练脚本：
```python #test id="sft-lora"
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
print("LoRA adapter saved to: output/trl-sft-lora")
print("TRL_SFT_DONE")
```

输出结果类似如下：

```shell #test-result id="sft-lora"
...
LoRA adapter saved to: output/trl-sft-lora
TRL_SFT_DONE
```

## 示例二：偏好优化 DPO LoRA

再用相同模型和数据集运行 3 步 DPO LoRA，适配器保存到 `output/trl-dpo-lora`。

```python #test id="dpo-lora"
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
print("LoRA adapter saved to: output/trl-dpo-lora")
print("TRL_DPO_DONE")
```

输出结果类似如下：

```shell #test-result id="dpo-lora"
...
LoRA adapter saved to: output/trl-dpo-lora
TRL_DPO_DONE
```

更多方法（GRPO / PPO / Reward / KTO 等）见 [TRL examples](https://github.com/huggingface/trl/tree/main/examples)。
