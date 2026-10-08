# llm-compressor

在单卡昇腾上安装 llm-compressor，对公开小模型做一次 W4A16 RTN 量化，再保存、重载并完成一次前向。

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**）。本文示例为单卡。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit 与驱动已安装，并能 `source set_env.sh`。版本按 [昇腾软件配套清单](https://www.hiascend.com/developer/download/compatibility) 选择 |
| Python | 落在官方配套表给出的范围内，并满足 llm-compressor 下限；当前正式版要求 `>=3.10` |
| PyTorch | 安装官方当前推荐的 `torch` 与 `torch_npu`，见 [CANN 与 PyTorch 配套表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md) 和 [PyTorch 安装包](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download) |
| 模型 | [nm-testing/tinysmokeqwen3](https://huggingface.co/nm-testing/tinysmokeqwen3)（约 10 MB） |

阅读本文前，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备好 CANN 与驱动。

## 1. 加载 CANN 环境

常见容器里 `npu-smi` 在 `/usr/local/sbin`，需要把该目录加入 `PATH`。

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
```

## 2. 检查环境是否就绪

### 2.1 确认 NPU 在线

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

输出路径和版本号随环境而变化，格式如下：

```text #test-result id="check-tools"
.../npu-smi
Python ...
```

## 3. 安装 PyTorch NPU 栈

```shell #test id="install-torch"
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  torch_npu numpy pyyaml
python -c "import numpy, yaml, torch, torch_npu; print('torch', torch.__version__); print('torch_npu', torch_npu.__version__); print('npu_available', torch.npu.is_available())"
```

完整安装日志较长，其中应包含：

```text #test-result id="install-torch"
...
torch ...+cpu
torch_npu ...
npu_available True
```

## 4. 安装 llm-compressor

NPU 栈就绪后，用 `--no-build-isolation` 安装当前正式版，让构建对着已装的 torch。

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

```shell #test id="install-llmcompressor" load="upstream_ref>>UPSTREAM_REF"
python -m pip install --no-build-isolation "llmcompressor==<UPSTREAM_REF>"
python -c "import llmcompressor; print('llmcompressor', llmcompressor.__version__)"
```


完整安装日志较长，其中应包含：

```text #test-result id="install-llmcompressor" load="upstream_ref>>UPSTREAM_REF"
...
llmcompressor <UPSTREAM_REF>
```

```{admonition} Note
:class: note
将 `<UPSTREAM_REF>` 换成最新的 release 版本号。
```

## 5. 在 NPU 上做一次 W4A16 RTN 量化

下面从 Hugging Face 下载公开小模型 `nm-testing/tinysmokeqwen3`，用 RTN 将全部 `Linear` 层的权重量化到 4 位，并跳过 `lm_head`。`oneshot` 把压缩后的模型写到本机 `~/llm-compressor-work/compressed`，随后从该目录重载并做一次前向。

保存为 `oneshot_forward.py`：

```python
from pathlib import Path

import torch
import torch_npu
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

model_id = "nm-testing/tinysmokeqwen3"
device = "npu:0"
compressed_dir = Path.home() / "llm-compressor-work" / "compressed"

model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)
recipe = QuantizationModifier(
    ignore=["lm_head"],
    config_groups={
        "group_0": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(num_bits=4, strategy="group", group_size=32),
        )
    },
)
torch.accelerator.max_memory_allocated = lambda device=None: 0
torch.accelerator.get_memory_info = lambda device=None: (0, 1)
oneshot(
    model=model,
    recipe=recipe,
    output_dir=str(compressed_dir),
)

reloaded = AutoModelForCausalLM.from_pretrained(compressed_dir).to(device)
inputs = tokenizer("hello", return_tensors="pt").to(device)
logits = reloaded(**inputs).logits
qc = reloaded.config.quantization_config
inner = getattr(qc, "quantization_config", qc)
group0 = inner.config_groups["group_0"]
print("weight_num_bits", group0.weights.num_bits)
print("q_proj_quantized", hasattr(reloaded.model.layers[2].self_attn.q_proj, "quantization_scheme"))
print("lm_head_quantized", hasattr(reloaded.lm_head, "quantization_scheme"))
print("logits.device", logits.device)
```

<!--
```shell #test-setup
cat > oneshot_forward.py <<'PY'
from pathlib import Path

import torch
import torch_npu
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

model_id = "nm-testing/tinysmokeqwen3"
device = "npu:0"
compressed_dir = Path.home() / "llm-compressor-work" / "compressed"

model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)
recipe = QuantizationModifier(
    ignore=["lm_head"],
    config_groups={
        "group_0": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(num_bits=4, strategy="group", group_size=32),
        )
    },
)
torch.accelerator.max_memory_allocated = lambda device=None: 0
torch.accelerator.get_memory_info = lambda device=None: (0, 1)
oneshot(
    model=model,
    recipe=recipe,
    output_dir=str(compressed_dir),
)

reloaded = AutoModelForCausalLM.from_pretrained(compressed_dir).to(device)
inputs = tokenizer("hello", return_tensors="pt").to(device)
logits = reloaded(**inputs).logits
qc = reloaded.config.quantization_config
inner = getattr(qc, "quantization_config", qc)
group0 = inner.config_groups["group_0"]
print("weight_num_bits", group0.weights.num_bits)
print("q_proj_quantized", hasattr(reloaded.model.layers[2].self_attn.q_proj, "quantization_scheme"))
print("lm_head_quantized", hasattr(reloaded.lm_head, "quantization_scheme"))
print("logits.device", logits.device)
PY
```
-->

运行：

```shell #test id="oneshot-forward"
python oneshot_forward.py
```

完整输出较长，其中应包含：

```shell #test-result id="oneshot-forward"
...
weight_num_bits 4
q_proj_quantized True
lm_head_quantized False
logits.device npu:0
```

## 6. 官方文档

- 上游仓库：[vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor)
- 文档中心：[LLM Compressor Docs](https://docs.vllm.ai/projects/llm-compressor/en/latest/)
- oneshot 与 RTN：[oneshot](https://docs.vllm.ai/projects/llm-compressor/en/latest/guides/entrypoints/oneshot)
