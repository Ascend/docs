# Transformers Ascend 快速开始

本文介绍通过 Transformers 在昇腾 NPU 上进行模型推理的两种方式：
`AutoModelForCausalLM` 与 `pipeline`，并给出完整的对话流程示例。
示例使用公开模型 `Qwen/Qwen2.5-1.5B-Instruct`，不需要 Hugging Face token。

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
| 模型 | `Qwen/Qwen2.5-1.5B-Instruct` |
| NPU | Ascend 910B4 × 1 |

## 环境准备

本文示例在下面的 Ascend 镜像中验证通过：

```text
swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12
```

镜像通常已经包含兼容的 `torch` / `torch_npu`。如果本地镜像没有提供，
请先按照 CANN 与 PyTorch-NPU 的兼容矩阵安装对应版本。

模型 `Qwen/Qwen2.5-1.5B-Instruct` 权重约 3 GB，下载步骤见下文「下载模型」一节；
也可以通过环境变量 `QUICK_START_MODEL` 指定本地已下载的模型路径，实现离线加载。

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

## 下载模型

模型权重约 3 GB，首次下载需要一些时间，建议单独执行，便于排查网络问题。
如果访问 Hugging Face 较慢，可先切换到国内镜像（该设置只影响当前终端）：

```shell
export HF_ENDPOINT=https://hf-mirror.com
```

下载到本地 Hugging Face 缓存目录：

```shell #test id="download-model"
python -c "from huggingface_hub import snapshot_download; print('downloaded to:', snapshot_download('Qwen/Qwen2.5-1.5B-Instruct'))"
```

输出结果如下（缓存路径因环境而异）：

```shell #test-result id="download-model"
downloaded to: ...
```

## 模型推理

针对模型推理，Transformers 提供了 `AutoModelForCausalLM` 与 `pipeline`
两种方式。每个示例都附有可直接复制运行的脚本版本。

### 使用 AutoModelForCausalLM

```python
import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
device = "npu:0" if torch.npu.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
).to(device)
```

加载完成后，确认模型已位于 NPU 上：

```shell #test id="automodel-load"
python - <<'PY'
import os
import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = os.environ.get("QUICK_START_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
device = "npu:0" if torch.npu.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
).to(device)

print(model.dtype)
print(model.device)
PY
```

输出结果如下：

```shell #test-result id="automodel-load"
torch.bfloat16
npu:0
```

### 使用 pipeline

```python
import torch
import torch_npu
import transformers

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
device = "npu:0" if torch.npu.is_available() else "cpu"

pipe = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device=device,
)
```

生成一段文本试试（为便于复现，这里固定使用贪心解码）：

```shell #test id="pipeline-generate"
python - <<'PY'
import os
import torch
import torch_npu
import transformers

model_id = os.environ.get("QUICK_START_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
device = "npu:0" if torch.npu.is_available() else "cpu"

pipe = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device=device,
)
result = pipe("1+1等于几？只回答一个数字。", max_new_tokens=8, do_sample=False)
text = result[0]["generated_text"]
print("generated:", text)
assert "2" in text
PY
```

输出结果如下：

```shell #test-result id="pipeline-generate"
generated: ...
```

### 全流程对话

下面把加载、对话模板与生成串成完整流程：

```python
import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
device = "npu:0" if torch.npu.is_available() else "cpu"  # 指定使用的设备为 NPU 0

# 加载预训练的分词器
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 加载预训练的语言模型，并指定数据类型为 bfloat16，自动选择设备映射
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
).to(device)

# 定义消息列表，包含系统消息和用户消息
messages = [
    {"role": "system", "content": "You are a housekeeper chatbot who always responds in polite expression!"},
    {"role": "user", "content": "Who are you? what should you do?"},
]

# 使用分词器将消息列表应用到聊天模板中，并转换为张量
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

# 生成响应
outputs = model.generate(
    input_ids,
    max_new_tokens=256,  # 设置生成的最大 token 数
    do_sample=True,
    temperature=0.6,     # 设置采样温度，影响生成的多样性
    top_p=0.9,
)

# 获取生成的响应，排除输入的部分
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))
```

输出示例（采样生成，每次内容会有所不同）：

```shell #test id="chat-flow"
python - <<'PY'
import os
import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = os.environ.get("QUICK_START_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
device = "npu:0" if torch.npu.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
).to(device)

messages = [
    {"role": "system", "content": "You are a housekeeper chatbot who always responds in polite expression!"},
    {"role": "user", "content": "Who are you? what should you do?"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print("response:", tokenizer.decode(response, skip_special_tokens=True))
PY
```

```shell #test-result id="chat-flow"
response: ...
```
