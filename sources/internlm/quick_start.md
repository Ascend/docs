# 快速开始

在单张昇腾 NPU 上运行 [InternLM](https://github.com/InternLM/InternLM)
上游 `ecosystem/README_npu.md` 中的 Transformers 推理链路：下载
InternLM3-8B-Instruct，以 FP16 加载到 `npu:0`，并完成一次确定性的文本生成。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者其他兼容的 Ascend NPU，至少有一张可用
设备，并已完成驱动和设备配置。CI 使用 `linux-aarch64-a2-1` Runner，由 Runner
自动提供一张配置完好的 NPU，不需要额外传入设备挂载参数。

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
| InternLM | `main`，由工作流解析并监控其 commit SHA |
| 模型 | `Shanghai_AI_Laboratory/internlm3-8b-instruct` |
| 精度 | FP16 |
| NPU | Ascend 910B4 × 1 |

上游最近的 GitHub release 早于 InternLM3 的 NPU 指南，因此本文不使用旧 release，
而是由工作流注入并持续监控 `main`。依赖版本则固定为本 Quick Start 的已知兼容栈，
避免包的自动升级改变验证结果。

### 检查前置条件

检查 Python：

```shell #test id="check-python"
python --version
```

```shell #test-result id="check-python" fuzzy="xxx"
Python 3.12.xxx
```

检查 PyTorch-NPU 和当前可见设备：

```shell #test id="check-torch"
python - <<'PY'
import torch
import torch_npu

assert torch.npu.is_available()
assert torch.npu.device_count() == 1
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("npu_available:", torch.npu.is_available())
print("npu_count:", torch.npu.device_count())
PY
```

```shell #test-result id="check-torch" fuzzy="xxx"
torch: 2.9.0xxx
torch_npu: 2.9.0.post2
npu_available: True
npu_count: 1
```

## 获取 InternLM 上游源码

工作流把受监控的上游分支写入 `UPSTREAM_REF`。下面 checkout 同一个 ref，并确认
其中确实包含 InternLM3-8B-Instruct 的官方 NPU Transformers 示例。

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

```shell #test-result id="checkout-upstream" fuzzy="xxx"
InternLM checkout: xxx
upstream NPU guide: OK
```

## 安装推理依赖

InternLM3 的模型卡要求 Transformers 4.48 或更新版本。这里采用上游 NPU 指南
对应的 4.48.0，并固定 ModelScope 版本；`torch` 与 `torch_npu` 属于前置环境，
不在这一步重复替换。

```shell #test id="install-deps"
uv pip install \
  "transformers==4.48.0" \
  "modelscope==1.37.0" \
  sentencepiece \
  safetensors
python - <<'PY'
import modelscope
import transformers

print("transformers:", transformers.__version__)
print("modelscope:", modelscope.__version__)
PY
```

```shell #test-result id="install-deps"
...transformers: 4.48.0
modelscope: 1.37.0
```

## 下载 InternLM3-8B-Instruct

通过上游 Model Zoo 提供的 ModelScope 仓库下载模型。CI 将 ModelScope 默认缓存目录
挂载到持久化磁盘；首次运行需要下载模型，后续运行复用已校验的缓存。

```shell #test id="download-model"
set -euo pipefail
mkdir -p /root/internlm-quick-start
model_dir="$(python - <<'PY' | sed -n 's/^MODEL_DIR=//p' | tail -n 1
from modelscope import snapshot_download

model_dir = snapshot_download("Shanghai_AI_Laboratory/internlm3-8b-instruct")
print(f"MODEL_DIR={model_dir}")
PY
)"
test -n "$model_dir"
test -f "$model_dir/config.json"
ln -sfn "$model_dir" /root/internlm-quick-start/model
echo "model: /root/internlm-quick-start/model/config.json"
```

```shell #test-result id="download-model"
model: /root/internlm-quick-start/model/config.json
```

## Quick Start：单卡 NPU 推理

以下流程保留上游示例的 `AutoTokenizer`、`AutoModelForCausalLM`、chat template、
FP16 和 `.npu()`。为了让持续集成的耗时和输出稳定，将生成长度缩短为 64 token，
并关闭随机采样；测试只验证设备、生成 token 数和非空响应，不绑定具体自然语言答案。

```shell #test id="npu-inference"
python - <<'PY'
import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "/root/internlm-quick-start/model"
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
PY
```

```shell #test-result id="npu-inference" fuzzy="xxx"
model device: npu:0
generated tokens: xxx
response: xxx
NPU inference PASSED
```

本 Quick Start 只看护 InternLM 上游已经存在的单卡 Transformers NPU 推理入口，
不覆盖多卡 HCCL、量化、服务部署以及外部训练框架。
