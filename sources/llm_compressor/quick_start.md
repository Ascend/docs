# llm-compressor

在单卡昇腾上安装 llm-compressor，对公开小模型做一次 W4A16 GPTQ，再保存、重载并完成一次前向。

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**）。本文示例为单卡。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit + 驱动固件已安装，并可 `source set_env.sh` |
| Python | 3.12 |
| PyTorch | `torch==2.10.0` 与 `torch_npu==2.10.0.post4`，见下文安装 |
| llm-compressor | 从 PyPI 安装发布版，见下文 |
| 模型 | [nm-testing/tinysmokeqwen3](https://huggingface.co/nm-testing/tinysmokeqwen3)（约 10 MB） |

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

昇腾上的 `torch_npu` 要从华为 PyPI 额外索引安装，并钉死与 CANN 9.1.0 匹配的版本。`numpy` 和 `pyyaml` 也要一起装：缺了会在 `import torch_npu` 之前失败。

```shell #test id="install-torch"
python -m pip install --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/variant \
  --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi \
  torch==2.10.0 torch_npu==2.10.0.post4 numpy pyyaml
python -c "import numpy, yaml, torch, torch_npu; print('torch', torch.__version__); print('torch_npu', torch_npu.__version__); print('npu_available', torch.npu.is_available())"
```

输出结果如下：

```shell #test-result id="install-torch"
...
torch 2.10.0...
torch_npu 2.10.0.post4
npu_available True
```

`npu_available` 必须是 `True`。`torch` 版本串可能带 `+cpu` 后缀，以 `npu_available True` 为准。

## 4. 安装 llm-compressor

将 `<UPSTREAM_REF>` 换成目标 PyPI 版本号（撰写时最新正式版是 `0.13.0`）。

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

```shell #test id="install-llmcompressor" load="upstream_ref>>UPSTREAM_REF"
python -m pip install --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/variant \
  --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi \
  llmcompressor==<UPSTREAM_REF> torch==2.10.0 torch_npu==2.10.0.post4
python -c "import llmcompressor; print('llmcompressor', llmcompressor.__version__)"
```

输出结果如下：

```shell #test-result id="install-llmcompressor"
...
llmcompressor ...
```

## 5. 在 NPU 上做一次单层 W4A16 GPTQ

下面从 Hugging Face 下载公开小模型 `nm-testing/tinysmokeqwen3`（不必事先准备权重），用 8 条本地校准文本只量化第 3 层的 `q_proj`。`oneshot` 把压缩后的模型写到本机 `~/llm-compressor-work/compressed`，随后从该目录重载并做一次前向。

保存为 `oneshot_forward.py`：

```python
from pathlib import Path

import torch
import torch_npu
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

model_id = "nm-testing/tinysmokeqwen3"
device = "npu:0"
compressed_dir = Path.home() / "llm-compressor-work" / "compressed"

model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)
ds = Dataset.from_dict(
    {
        "text": [
            "The quick brown fox jumps over the lazy dog.",
            "Quantization maps weights to fewer bits.",
            "Ascend NPU runs this oneshot calibration.",
            "A short sentence is enough for a smoke test.",
        ]
        * 2
    }
)
recipe = GPTQModifier(
    ignore=["lm_head"],
    config_groups={
        "group_0": QuantizationScheme(
            targets=["re:.*model.layers.2.self_attn.q_proj$"],
            weights=QuantizationArgs(num_bits=4, strategy="group", group_size=32),
        )
    },
)
torch.accelerator.max_memory_allocated = lambda device=None: 0
torch.accelerator.get_memory_info = lambda device=None: (0, 1)
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    num_calibration_samples=8,
    max_seq_length=64,
    output_dir=str(compressed_dir),
)

reloaded = AutoModelForCausalLM.from_pretrained(compressed_dir).to(device)
inputs = tokenizer("hello", return_tensors="pt").to(device)
logits = reloaded(**inputs).logits
qc = reloaded.config.quantization_config
inner = getattr(qc, "quantization_config", qc)
group0 = inner.config_groups["group_0"]
print("LLM_COMPRESSOR_WORKLOAD_DEVICE=npu:0")
print("weight_num_bits", group0.weights.num_bits)
print("targeted", hasattr(reloaded.model.layers[2].self_attn.q_proj, "quantization_scheme"))
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
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

model_id = "nm-testing/tinysmokeqwen3"
device = "npu:0"
# oneshot 写入量化结果的目录，不是事先准备好的权重路径
compressed_dir = Path.home() / "llm-compressor-work" / "compressed"

model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)
ds = Dataset.from_dict(
    {
        "text": [
            "The quick brown fox jumps over the lazy dog.",
            "Quantization maps weights to fewer bits.",
            "Ascend NPU runs this oneshot calibration.",
            "A short sentence is enough for a smoke test.",
        ]
        * 2
    }
)
recipe = GPTQModifier(
    ignore=["lm_head"],
    config_groups={
        "group_0": QuantizationScheme(
            targets=["re:.*model.layers.2.self_attn.q_proj$"],
            weights=QuantizationArgs(num_bits=4, strategy="group", group_size=32),
        )
    },
)
torch.accelerator.max_memory_allocated = lambda device=None: 0
torch.accelerator.get_memory_info = lambda device=None: (0, 1)
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    num_calibration_samples=8,
    max_seq_length=64,
    output_dir=str(compressed_dir),
)

reloaded = AutoModelForCausalLM.from_pretrained(compressed_dir).to(device)
inputs = tokenizer("hello", return_tensors="pt").to(device)
logits = reloaded(**inputs).logits
qc = reloaded.config.quantization_config
inner = getattr(qc, "quantization_config", qc)
group0 = inner.config_groups["group_0"]
print("LLM_COMPRESSOR_WORKLOAD_DEVICE=npu:0")
print("weight_num_bits", group0.weights.num_bits)
print("targeted", hasattr(reloaded.model.layers[2].self_attn.q_proj, "quantization_scheme"))
print("lm_head_quantized", hasattr(reloaded.lm_head, "quantization_scheme"))
print("logits.device", logits.device)
PY
```
-->

运行：

```shell #test id="oneshot-forward"
python oneshot_forward.py
```

输出结果如下：

```shell #test-result id="oneshot-forward"
...
LLM_COMPRESSOR_WORKLOAD_DEVICE=npu:0
weight_num_bits 4
targeted True
lm_head_quantized False
logits.device npu:0
```

进程须退出码为 0，且 `logits.device` 必须是 `npu:0`；若打印 `cpu`，那是静默回退，视为失败。
