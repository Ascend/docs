# InternLM

在单张昇腾 NPU 上使用 [InternLM](https://github.com/InternLM/InternLM)：
下载 InternLM3-8B-Instruct 模型，以 FP16 加载到 `npu:0`，并生成一段文本。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者其他兼容的 Ascend NPU，至少有一张可用
设备，并已完成驱动和设备配置。

### 基础软件

运行本文档之前，需要准备：

- Linux aarch64 和 Python 3.12；
- CANN toolkit、驱动和 `npu-smi`；
- 与 CANN 匹配的 PyTorch 和 PyTorch-NPU；
- 至少约 20 GB 的模型缓存空间，并能访问 GitHub 和 ModelScope。

### 本文档示例使用的版本

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| torch | 2.9.0+cpu |
| torch_npu | 2.9.0.post2 |
| transformers | 4.48.0 |
| modelscope | 1.37.0 |
| InternLM | `main` |
| 模型 | `Shanghai_AI_Laboratory/internlm3-8b-instruct` |
| 精度 | FP16 |
| NPU | Ascend 910B4 × 1 |

本文使用 InternLM `main` 分支及上表所示的依赖组合。安装其他版本时，
请先核对模型与 Transformers、PyTorch-NPU 的兼容性。

### 检查前置条件

检查 Python：

```shell #test id="check-python"
python --version
```

输出结果如下：

```text #test-result id="check-python" fuzzy="xxx"
Python 3.12.xxx
```

检查 PyTorch-NPU 和当前可见设备：

```python #test id="check-torch"
import torch
import torch_npu

assert torch.npu.is_available()
assert torch.npu.device_count() == 1
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("npu_available:", torch.npu.is_available())
print("npu_count:", torch.npu.device_count())
```

输出结果如下：

```text #test-result id="check-torch"
torch: 2.9.0+cpu
torch_npu: 2.9.0.post2
npu_available: True
npu_count: 1
```

## 获取 InternLM 上游源码

获取 InternLM 官方源码。将 `<ref>` 替换为 `main`；仓库中的
`ecosystem/README_npu.md` 提供更多 NPU 使用说明。

<!--
```shell #test-setup store="upstream_ref"
printf '%s\n' "$UPSTREAM_REF"
```
-->

```shell #test id="checkout-upstream" load="upstream_ref>>ref"
git clone --depth 1 --branch <ref> https://github.com/InternLM/InternLM.git internlm-src
grep -Fq 'InternLM3-8B-Instruct' internlm-src/ecosystem/README_npu.md
grep -Fq ').npu()' internlm-src/ecosystem/README_npu.md
printf 'InternLM checkout: %s\n' "$(git -C internlm-src rev-parse --short HEAD)"
echo "upstream NPU guide: OK"
```

输出结果如下：

```text #test-result id="checkout-upstream" fuzzy="xxx"
InternLM checkout: xxx
upstream NPU guide: OK
```

## 安装推理依赖

安装模型推理和下载所需的 Python 包。本文使用 Transformers 4.48.0 和
ModelScope 1.37.0；已在前置环境中安装的 `torch` 与 `torch_npu` 无需重复安装。

```shell #test-setup
uv pip install \
  "transformers==4.48.0" \
  "modelscope==1.37.0" \
  sentencepiece \
  safetensors
```

```python #test id="install-deps"
import modelscope
import transformers

print("transformers:", transformers.__version__)
print("modelscope:", modelscope.__version__)
```

输出结果如下：

```text #test-result id="install-deps"
...transformers: 4.48.0
modelscope: 1.37.0
```

## 下载 InternLM3-8B-Instruct

从 ModelScope 下载 InternLM3-8B-Instruct。首次运行需要下载模型；后续运行
可复用本机缓存。下面在当前目录创建 `internlm-model` 链接，供推理步骤使用。

```python #test id="download-model"
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from modelscope import snapshot_download

with redirect_stdout(StringIO()):
    model_dir = Path(snapshot_download("Shanghai_AI_Laboratory/internlm3-8b-instruct")).resolve()
assert (model_dir / "config.json").is_file()

link = Path.cwd() / "internlm-model"
if link.is_symlink():
    link.unlink()
elif link.exists():
    raise RuntimeError(f"model link path already exists: {link}")
link.symlink_to(model_dir, target_is_directory=True)
assert (link / "config.json").is_file()
print(f"model: {link / 'config.json'}")
```

输出结果如下：

```text #test-result id="download-model" fuzzy="xxx"
model: xxx/internlm-model/config.json
```

## 单卡 NPU 推理

加载 tokenizer 和模型，将模型与输入移到 NPU，然后根据提问生成回答。
`max_new_tokens=64` 限制回答长度，`do_sample=False` 关闭随机采样。

```python #test id="npu-inference"
import torch
import torch_npu
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = str(Path.cwd() / "internlm-model")
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    trust_remote_code=True,
    torch_dtype=torch.float16,
).npu()
model.eval()

messages = [
    {
        "role": "system",
        "content": "You are InternLM, a helpful, honest, and harmless AI assistant.",
    },
    {"role": "user", "content": "Please name one scenic spot in Shanghai."},
]
tokenized_chat = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).npu()

with torch.inference_mode():
    generated_ids = model.generate(
        tokenized_chat,
        max_new_tokens=64,
        do_sample=False,
    )
torch.npu.synchronize()

new_tokens = generated_ids[:, tokenized_chat.shape[-1]:]
response = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
model_device = next(model.parameters()).device

assert model_device.type == "npu"
assert tokenized_chat.device.type == "npu"
assert new_tokens.shape[-1] > 0
assert response

print("model device:", model_device)
print("generated tokens:", new_tokens.shape[-1])
print("response:", response.replace("\n", " "))
print("NPU inference PASSED")
```

输出结果如下：

```text #test-result id="npu-inference" fuzzy="xxx"
model device: npu:0
generated tokens: xxx
response: xxx
NPU inference PASSED
```

看到 `NPU inference PASSED` 表示模型已在单张 NPU 上完成文本生成。
多卡推理、量化和服务部署不在本文范围内。
