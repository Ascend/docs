# 快速开始

在单卡昇腾上对 Qwen2.5-3B-Instruct 应用 LoRA、保存并重新加载 PEFT 适配器。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，并按需完成物理机或容器内的设备挂载。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 与上面 CANN 匹配的 `torch` + `torch_npu`，且 `torch` 能正常 `import` 并 `torch.npu.is_available() == True`（参考 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)，按 torch ↔ torch_npu ↔ CANN 三方兼容矩阵选择版本）

### 本文档示例使用的版本

**配套机器**：

- **机器类型**：Atlas 900 A2 PODc（Ascend 910B4，64 GB × 1）
- **操作系统**：Ubuntu 22.04

**配套镜像**：

swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12

**软件版本**：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| torch | 2.9.0+cpu |
| torch_npu | 2.9.0.post2 |
| transformers | `<5.0` |
| peft | 最新 release 的源码/二进制 |
| huggingface_hub | 最新稳定版（随 `transformers` 一起拉） |
| 模型 | [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) |

### 前置安装
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
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| No running processes found in NPU 5                                                            |
+===========================+===============+====================================================+
```

> 如果 `npu-smi` 不存在，请回到 [Ascend 官方快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 补装驱动

检查 Python 版本：

```shell #test id="check-py"
python --version
```
输出结果如下：
```shell #test-result id="check-py" fuzzy='xxx'
Python 3.12.xxx
```

检查 NPU 设备运行时可用：

```shell #test id="check-npu-runtime"
python -c "import torch, torch_npu; print(f'torch={torch.__version__}'); print(f'torch_npu={torch_npu.__version__}'); print('is_available:', torch.npu.is_available()); print('count:', torch.npu.device_count())"
```

输出结果如下：

```shell #test-result id="check-npu-runtime"
torch=2.9.0+cpu
torch_npu=2.9.0.post2
is_available: True
count: 1
```

> 如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。

安装 transformers / huggingface_hub：

```shell #test-setup
pip install 'transformers<5.0' huggingface_hub
```

打印安装版本：
```shell #test id="install-deps"
python -c "import transformers, huggingface_hub; print(f'transformers={transformers.__version__} huggingface_hub={huggingface_hub.__version__}')"
```

输出结果如下：

```shell #test-result id="install-deps" fuzzy='xxx'
transformers=xxx huggingface_hub=xxx
```

## 安装 PEFT

### 使用 uv 进行安装

```shell #test id="peft-install-binary"
uv pip install --index-url https://mirrors.aliyun.com/pypi/simple peft
python -c "import peft; print('peft', peft.__version__)"
```

输出结果类似如下：

```shell #test-result id="peft-install-binary" fuzzy='xxx'
peft xxx
```
- xxx 表示最新的版本号
<!--
```shell #test-setup
uv pip uninstall peft -y
```
-->

### 从源码安装
<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

克隆上游仓库并 checkout 到工作流注入的最新 release tag，安装并且验证

```shell #test id="peft-install-source" load="upstream_ref>>ref"
git clone --depth 1 --branch <ref> https://github.com/huggingface/peft.git
cd peft
uv pip install -e .
python -c "import peft; print('peft', peft.__version__)"
```
\<ref> 为安装的最新的 release 分支

输出结果类似如下：

```shell #test-result id="peft-install-source" fuzzy='xxx'
peft xxx
```
- xxx 表示最新的版本号

## 使用 PEFT 方法（例如 LoRA）准备训练模型

将基础模型和 PEFT 配置包装起来 `get_peft_model`，并保存适配器。对于 Qwen2.5-3B-Instruct 这种 3B 模型，仅训练约 0.12% 的参数！

### 下载基础模型

默认使用 **HuggingFace Hub** 进行模型下载。

```shell #test-setup store="model_path"
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-3B-Instruct'))" | tail -n 1
```

### 应用 LoRA 适配器

把基础模型加载到 NPU 上（`bfloat16` 省显存），构造 `LoraConfig` 描述要插入的 LoRA 矩阵（rank=16 / alpha=32 / 自回归 LM 任务），再用 `get_peft_model` 包成 PEFT 模型——底座权重默认冻结，只有新注入的 LoRA 矩阵参与训练。

```shell #test id="apply-lora" load="model_path>>model_path"
python << 'PY'
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model

model = AutoModelForCausalLM.from_pretrained(
    "<model_path>", torch_dtype=torch.bfloat16,
).to("npu:0")

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    task_type=TaskType.CAUSAL_LM,
)
peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()
peft_model.save_pretrained("output/peft-adapter")
PY
ls output/peft-adapter/adapter_config.json output/peft-adapter/adapter_model.safetensors
```

> `<model_path>` 为上面“下载基础模型” 章节对应命令的输出

输出结果如下：

```shell #test-result id="apply-lora" fuzzy='xxx'
trainable params: xxx || all params: xxx || trainable%: xxx
output/peft-adapter/adapter_config.json
output/peft-adapter/adapter_model.safetensors
```

## 加载用于推理的 PEFT 模型

推理的入口。PEFT 把「底座」与「适配器」解耦得很干净——同一份底座可以快速切换不同任务的适配器，无需拷贝整个模型。本节演示：先加载底座（与训练同源），再把上一步保存的 LoRA 适配器「贴」上去，最后用 `generate()` 端到端跑一次生成验证链路通。

### 加载 PEFT 模型

推理的第一步：加载 `tokenizer` + 底座（`AutoModelForCausalLM`），然后用 `PeftModel.from_pretrained(base, "output/peft-adapter")` 把适配器「贴」上去——这一步在底座上原地构造 PEFT 包装，权重来自上一步保存的目录。

```shell #test id="load-adapter" load="model_path>>model_path"
python << 'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "<model_path>", torch_dtype=torch.bfloat16,
).to("npu:0")
tokenizer = AutoTokenizer.from_pretrained("<model_path>")
peft_model = PeftModel.from_pretrained(base, "output/peft-adapter")
peft_model.print_trainable_parameters()
PY
```

输出结果如下：

```shell #test-result id="load-adapter"
trainable params: ... || all params: ... || trainable%: ...
```

> 这里的 `<model_path>` 和上面 `apply-lora` 块里的一样，由「下载基础模型」一节的 `#test-setup store="model_path"` 捕获并注入；不需要在本块手动替换。

### 跑一次生成验证

端到端跑一次生成：tokenizer 把 prompt 编码成 ids，搬到 NPU 上，`model.generate(max_new_tokens=20, do_sample=False)` 续写 20 个 token，解码回文本。PEFT 模型继承 `PreTrainedModel` 接口，`generate` 调用方式与底座完全一致。

```shell #test id="infer" load="model_path>>model_path"
python << 'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "<model_path>", torch_dtype=torch.bfloat16,
).to("npu:0")
tokenizer = AutoTokenizer.from_pretrained("<model_path>")
peft_model = PeftModel.from_pretrained(base, "output/peft-adapter")

inputs = tokenizer("Preheat the oven to 350 degrees and place the cookie dough", return_tensors="pt").to("npu:0")
outputs = peft_model.generate(**inputs, max_new_tokens=20, do_sample=False)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
PY
```

输出结果如下：

```shell #test-result id="infer"
Preheat the oven to 350 degrees and place the cookie dough...
```

小贴士：

- 如果要切换其他 PEFT 方法（如 AdaLoRA、IA3、VeRA 等），只需要把 `LoraConfig` 换成对应方法的 config 即可，调用方式保持不变。
- `target_modules` 接受字符串列表、模块类或正则；不指定时，PEFT 会自动对所有 `nn.Linear` 子模块注入 LoRA。
- `task_type` 用来辅助 PEFT 保存与任务相关的层；自回归 LM 任务填 `TaskType.CAUSAL_LM`，分类任务填 `TaskType.SEQ_CLS`。
- 推理段 prompt 选的是英文烘焙场景：base 模型未针对该任务训练，生成内容是 base 的自然续写，验证 `PeftModel.from_pretrained` 链路可用即可。