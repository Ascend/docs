## 快速开始

本文介绍通过 Transformers 在昇腾 NPU 上进行模型推理的两种方式：
`AutoModelForCausalLM` 与 `pipeline`，并给出完整的对话流程示例；同时提供
pipeline 在音频与自然语言处理任务上的使用示例，以及微调预训练模型的完整流程。
快速开始主链路的示例使用公开模型 `Qwen/Qwen2.5-1.5B-Instruct`，不需要 Hugging Face token。

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
| torch | 2.9.0+cpu |
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

模型 `Qwen/Qwen2.5-1.5B-Instruct` 权重约 3 GB，下载步骤见下文「下载模型」一节。

## 检查前置是否满足

检查 Python 版本：

```shell #test id="check-py"
python --version
```

输出结果如下：

```shell #test-result id="check-py" fuzzy="xxx"
Python 3.12.xxx
```

检查 CANN、Torch、Torch-NPU 和 NPU 设备：
（以下代码使用Python执行）
```python #test id="check-torch"
import torch
import torch_npu

print('torch=', torch.__version__)
print('torch_npu=', torch_npu.__version__)
print('is_available:', torch.npu.is_available())
print('count:', torch.npu.device_count())
```

输出结果如下（torch 为 CPU 构建版本，`+cpu` 表示 NPU 支持由 torch_npu 提供）：

```shell #test-result id="check-torch" fuzzy="xxx"
torch= 2.9.0+cpu
torch_npu= 2.9.0.post2
is_available: True
count: 1
```

确认能看到 NPU 设备：

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
```

```{admonition} Note
:class: note
如果 `npu-smi` 不存在，请回到 [Ascend 官方快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 补装驱动。
如果 `import torch_npu` 失败，请先修复驱动、CANN、Torch 与 Torch-NPU 的版本匹配问题。
```

## 安装 Transformers

### 创建虚拟环境

首先需要安装并激活 Python 环境：

```shell
conda create -n your_env_name python=3.12
conda activate your_env_name
```

下面安装与验证命令用 Python 执行。安装 transformers、accelerate 并验证：

```shell #test id="install-transformers"
python -m pip install -q -U transformers accelerate
python -c "import accelerate, transformers; print('transformers', transformers.__version__); print('accelerate', accelerate.__version__)"
```

输出结果如下：

```shell #test-result id="install-transformers" fuzzy="xxx"
transformers xxx
accelerate xxx
```

```{admonition} Note
:class: note
`xxx` 表示实际安装的版本号。
```

## 下载模型

模型权重约 3 GB，首次下载需要一些时间，建议单独执行，便于排查网络问题。
如果访问 Hugging Face 较慢，可先切换到国内镜像（该设置只影响当前终端）：

```shell
export HF_ENDPOINT=https://hf-mirror.com
```

下载到本地 Hugging Face 缓存目录：

（以下代码使用Python执行）
```python #test id="download-model"
from huggingface_hub import snapshot_download
print('downloaded to:', snapshot_download('Qwen/Qwen2.5-1.5B-Instruct'))
```

输出结果如下（缓存路径因环境而异）：

```shell #test-result id="download-model"
downloaded to: ...
```

## 模型推理

针对模型推理，Transformers 提供了 `AutoModelForCausalLM` 与 `pipeline`
两种方式。每个示例都附有可直接复制运行的脚本版本。

### 使用 AutoModelForCausalLM

（以下代码使用Python执行）

```python #test id="automodel-load"
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

print(model.dtype)
print(model.device)
```

输出结果如下，加载完成后，确认模型已位于 NPU 上：

```shell #test-result id="automodel-load"
torch.bfloat16
npu:0
```

### 使用 pipeline

创建 pipeline 并生成一段文本：

（以下代码使用Python执行）

```python #test id="pipeline-generate"
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
result = pipe("1+1等于几？只回答一个数字。", max_new_tokens=8, do_sample=False)
text = result[0]["generated_text"]
prompt = "1+1等于几？只回答一个数字。"
answer = text[len(prompt):]
print("generated:", text)
print("answer:", answer)
```

输出结果如下：

```shell #test-result id="pipeline-generate" fuzzy='...'
generated: ...
answer: ...2...
```

### 全流程对话

把加载、对话模板与生成串成完整流程：

（以下代码使用Python执行）

```python #test id="chat-flow"
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

messages = [
    {"role": "system", "content": "You are a housekeeper chatbot who always responds in polite expression!"},
    {"role": "user", "content": "Who are you? what should you do?"},
]

text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

outputs = model.generate(
    **model_inputs,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][model_inputs.input_ids.shape[-1]:]
print("response:", tokenizer.decode(response, skip_special_tokens=True))
```

输出结果如下：

```shell #test-result id="chat-flow"
response: ...
```

输出示例（采样生成，每次内容会有所不同，以下为一次真实运行的参考对比）：

```text
response: Hello! I'm an AI designed to assist with various household tasks. Here's how we can work together
```

### pipeline 抽象类

pipeline 抽象类是所有其他 pipeline 的封装，可以像其他任何 pipeline 一样实例化。

pipeline 参数由 task、tokenizer、model、optional 组成：

- task 将确定返回哪一个 pipeline，比如 text-classification 将会返回 TextClassificationPipeline，automatic-speech-recognition 将会返回 AutomaticSpeechRecognitionPipeline。
- tokenizer 分词器是用来将输入进行编码，str 或者 PreTrainedTokenizer，如果未提供将使用 model 参数，如果 model 也未提供或者非 str，将使用 config 参数，如果 config 参数也未提供或者非 str，将提供 task 的默认 tokenizer。
- model 是模型，str 或者 PreTrainedModel，一般为有 .bin 模型文件的目录。
- optional 其他参数包括，config、feature_extractor、device、device_map 等。

### pipeline 场景示例

pipeline 适用于音频、自然语言处理等任务，下面介绍它在各场景的使用方式。

音频类示例需要环境中已安装 `ffmpeg`（pipeline 通过 `ffmpeg` 解码音频输入）：

```shell #test-setup
apt-get update
apt-get install -y ffmpeg
```

#### 音频

**音频识别**：用于提取某些音频中包含的文本，如下创建 pipeline，并输入音频文件：

（以下代码使用Python执行）

```python #test id="pipeline-asr"
from transformers import pipeline

transcriber = pipeline(task="automatic-speech-recognition")
print(transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"))
```

输出结果如下：

```shell #test-result id="pipeline-asr"
{'text': 'I HAVE A DREAM BUT ONE DAY THIS NATION WILL RISE UP LIVE UP THE TRUE MEANING OF ITS TREES'}
```

#### 自然语言处理

**文本分类**：根据标签对文本进行分类：

（以下代码使用Python执行）

```python #test id="pipeline-zero-shot"
from transformers import pipeline

classifier = pipeline(model="facebook/bart-large-mnli")
print(classifier(
    "I have a problem with my iphone that needs to be resolved asap!!",
    candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"],
))
```

输出结果如下：

```shell #test-result id="pipeline-zero-shot"
{'sequence': 'I have a problem with my iphone that needs to be resolved asap!!', 'labels': ...
```

**文本生成**：根据文本生成对话响应：

（以下代码使用Python执行）

```python #test id="pipeline-chat"
from transformers import pipeline

generator = pipeline(model="HuggingFaceH4/zephyr-7b-beta")
# Zephyr-beta is a conversational model, so let's pass it a chat instead of a single string
output = generator([{"role": "user", "content": "What is the capital of France? Answer in one word."}], do_sample=False, max_new_tokens=2)
print("chat:", output)
```

输出结果如下：

```shell #test-result id="pipeline-chat"
chat: [{'generated_text': [{'role': 'user', 'content': 'What is the capital of France? Answer in one word.'}, {'role': 'assistant', 'content': ...
```

## 微调预训练模型

大模型微调本质是利用特定领域的数据集对已预训练的大模型进行进一步训练的过程。它旨在优化模型在特定任务上的性能，使模型能够更好地适应和完成特定领域的任务。
本文在使用 transformers 库选定相关数据集和预训练模型的基础上，通过超参数调优完成对模型的微调。

### 前置准备

**安装必要库**：

```shell
pip install transformers datasets evaluate accelerate scikit-learn
```

**加载数据集**：模型训练需要使用数据集，这里使用 [Yelp Reviews dataset](https://huggingface.co/datasets/Yelp/yelp_review_full)：

（以下代码使用Python执行）

```python #test id="finetune-load-dataset"
from datasets import load_dataset

# load_dataset 会自动下载数据集并将其保存到本地路径中
dataset = load_dataset("Yelp/yelp_review_full")
# 输出数据集的第 100 条数据
print(dataset["train"][100])
```

输出结果如下：

```shell #test-result id="finetune-load-dataset"
{'label': 0, 'text': 'My expectations for McDonalds are t rarely high. But for one to still fail so spectacularly...
```


### 预训练全流程

（以下代码使用Python执行）

```python #test id="finetune-train"
import torch
import torch_npu
import numpy as np
import sklearn
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
device = "npu:0" if torch.npu.is_available() else "cpu"

# 加载模型：使用 AutoModelForCausalLM 将自动加载模型：
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
).to(device)

# load_dataset 会自动下载数据集并将其保存到本地路径中
dataset = load_dataset("Yelp/yelp_review_full")

# 预处理数据集：预处理数据集需要使用 AutoTokenizer，它用来自动获取与模型匹配的分词器，分词器根据规则将文本拆分为标记，并转换为张量作为模型输入
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 分词函数：语言建模需要 labels=input_ids
def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    tokenized["labels"] = tokenized["input_ids"]
    return tokenized

# 训练全部的数据集会耗费更长的时间，通常将其划分为较小的训练集和验证集，以提高训练速度
small_train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = dataset["test"].shuffle(seed=42).select(range(100))

# 使用 dataset.map 方法对数据集进行预处理，只对选中的样本做分词，避免全量数据集的无效预处理；
small_train_dataset = small_train_dataset.map(tokenize_function, batched=True, remove_columns=["text", "label"])
small_eval_dataset = small_eval_dataset.map(tokenize_function, batched=True, remove_columns=["text", "label"])

# 加载评估指标
metric = evaluate.load("accuracy")

# 模型评估：模型评估用于衡量模型在给定数据集上的表现，包括准确率，完全匹配速率，平均并交集点等
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # 评估库仅接受一维格式，压平 token 级的 [样本, 序列长度] 数组
    return metric.compute(predictions=predictions.flatten(), references=labels.flatten())

# 超参数调优：超参数调优用于激活不同训练选项的标志，它定义了关于模型的更高层次的概念，例如模型复杂程度或学习能力，下面使用 TrainingArguments 类来加载
training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")

# Trainer：使用已加载的模型、训练参数、训练和测试数据集以及评估函数创建一个 Trainer 对象，并调用 trainer.train() 来微调模型
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
print("global_step:", trainer.state.global_step)
```

输出结果如下（训练集 1000 条、验证集 100 条、batch size 8、默认 3 个 epoch，共 375 步；耗时与指标随环境不同）：

```shell #test-result id="finetune-train" fuzzy='...'
...global_step: 375...
```
