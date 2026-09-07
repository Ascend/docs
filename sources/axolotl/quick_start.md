# axolotl

本文在单卡昇腾 NPU 上安装 [axolotl](https://github.com/axolotl-ai-cloud/axolotl)，并对 [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) 完成 3 步 LoRA 监督微调。

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**），单卡。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit 与驱动已安装，并能 `source set_env.sh` |
| Python | 3.12 |
| PyTorch | `torch==2.11.0` 与 `torch_npu==2.11.0` |
| axolotl | 从 PyPI 安装当前正式版（`--no-deps`） |
| 模型 | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |

**配套机器**：Atlas 900 A2 PODc（Ascend 910B4）。**配套镜像**：`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`。

---

## 1. 加载 CANN 环境

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:/usr/local/bin:$PATH
export PYTHONNOUSERSITE=1
```

`PYTHONNOUSERSITE=1` 令 Python 忽略用户目录中的包，避免旧版 CANN 相关包干扰依赖解析。

---

## 2. 检查环境

确认 NPU 在线；找不到 `npu-smi` 时参阅 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)。

```shell
npu-smi info
```

检查 Python 版本：

```shell #test id="check-py"
python --version
```

输出结果如下：

```shell #test-result id="check-py" fuzzy="xxx"
Python 3.12.xxx
```

---

## 3. 安装 PyTorch NPU 栈

从华为 PyPI 额外索引安装与 CANN 9.1.0 匹配的 `torch==2.11.0` / `torch_npu==2.11.0`，并一并装上 `numpy`、`pyyaml`。版本行出现 `2.11.0+cpu` 属正常，`npu_available` 须为 `True`。

```shell #test id="install-torch"
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.11.0
python -m pip install --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi \
  torch_npu==2.11.0 numpy pyyaml
python -c "import numpy, yaml, torch, torch_npu; print('torch', torch.__version__); print('torch_npu', torch_npu.__version__); print('npu_available', torch.npu.is_available())"
```

输出结果如下：

```shell #test-result id="install-torch"
...torch 2.11.0...
torch_npu 2.11.0
npu_available True
```

---

## 4. 安装 axolotl

NPU 栈就绪后，用 `--no-build-isolation --no-deps` 安装当前正式版 axolotl，并补齐 LoRA 训练所需的其余包。不要用 `pip install axolotl[deepspeed]`；`bitsandbytes` 只为满足 import，不用 4-bit / 8-bit。

<!--
```shell #test-setup store="axolotl_ver"
echo "${UPSTREAM_REF#v}"
```
-->

```shell #test id="install-axolotl" load="axolotl_ver>>ver"
python -m pip install --no-build-isolation --no-deps "axolotl==<ver>"
python -m pip install \
  'packaging==26.0' 'huggingface_hub==1.17.0' 'peft==0.19.1' \
  'tokenizers==0.22.2' 'transformers==5.14.1' 'accelerate==1.13.0' \
  'datasets==4.8.4' 'trl==1.8.0' \
  sentencepiece einops colorama fire addict \
  'typer==0.25.1' 'pydantic==2.12.5' 'python-dotenv==1.0.1' \
  requests art 'hf_xet==1.4.3' hf_transfer \
  'axolotl-contribs-lgpl==0.0.7' 'axolotl-contribs-mit==0.0.6' \
  'bitsandbytes==0.49.1' scipy evaluate tensorboard \
  'schedulefree==1.4.1' numba posthog fastcore triton wandb torchao
python -c "import axolotl; print('axolotl', axolotl.__version__)"
```

输出结果如下：

```shell #test-result id="install-axolotl" load="axolotl_ver>>ver"
...axolotl <ver>
```

---

## 5. 准备数据和配置

工作目录为 `/root/axolotl-qs`。先从 Hugging Face Hub 把模型链到该目录，再按下面内容保存训练数据和 LoRA 配置。

```shell #test id="download-model"
mkdir -p /root/axolotl-qs
rm -f /root/axolotl-qs/model
ln -s "$(python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("Qwen/Qwen2.5-0.5B-Instruct"))' | grep '^/' | tail -n 1)" /root/axolotl-qs/model
ls /root/axolotl-qs/model/config.json
```

输出结果如下：

```shell #test-result id="download-model"
/root/axolotl-qs/model/config.json
```

训练数据如下，保存为 `/root/axolotl-qs/tiny_alpaca.jsonl`：

```json
{"instruction": "Translate to English.", "input": "你好", "output": "Hello"}
{"instruction": "Name a color.", "input": "", "output": "Blue"}
{"instruction": "Add the numbers.", "input": "2 and 3", "output": "5"}
{"instruction": "Say yes or no.", "input": "", "output": "Yes"}
```

<!--
```shell #test-setup
cat > /root/axolotl-qs/tiny_alpaca.jsonl <<'EOF'
{"instruction": "Translate to English.", "input": "你好", "output": "Hello"}
{"instruction": "Name a color.", "input": "", "output": "Blue"}
{"instruction": "Add the numbers.", "input": "2 and 3", "output": "5"}
{"instruction": "Say yes or no.", "input": "", "output": "Yes"}
EOF
```
-->

LoRA 配置如下，保存为 `/root/axolotl-qs/lora-npu.yml`：

```yaml
base_model: /root/axolotl-qs/model
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer
load_in_8bit: false
load_in_4bit: false
strict: false
datasets:
  - path: /root/axolotl-qs/tiny_alpaca.jsonl
    type: alpaca
    ds_type: json
dataset_prepared_path: /root/axolotl-qs/prepared
val_set_size: 0
output_dir: /root/axolotl-qs/outputs
sequence_len: 256
sample_packing: false
pad_to_sequence_len: false
adapter: lora
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target_linear: true
gradient_accumulation_steps: 1
micro_batch_size: 1
num_epochs: 1
max_steps: 3
optimizer: adamw_torch
lr_scheduler: cosine
learning_rate: 0.0002
bf16: true
tf32: false
gradient_checkpointing: true
flash_attention: false
attn_implementation: eager
lora_mlp_kernel: false
lora_qkv_kernel: false
lora_o_kernel: false
lora_embedding_kernel: false
logging_steps: 1
warmup_steps: 0
saves_per_epoch: 1
save_total_limit: 1
trust_remote_code: false
```

<!--
```shell #test-setup
cat > /root/axolotl-qs/lora-npu.yml <<'YAML'
base_model: /root/axolotl-qs/model
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer
load_in_8bit: false
load_in_4bit: false
strict: false
datasets:
  - path: /root/axolotl-qs/tiny_alpaca.jsonl
    type: alpaca
    ds_type: json
dataset_prepared_path: /root/axolotl-qs/prepared
val_set_size: 0
output_dir: /root/axolotl-qs/outputs
sequence_len: 256
sample_packing: false
pad_to_sequence_len: false
adapter: lora
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target_linear: true
gradient_accumulation_steps: 1
micro_batch_size: 1
num_epochs: 1
max_steps: 3
optimizer: adamw_torch
lr_scheduler: cosine
learning_rate: 0.0002
bf16: true
tf32: false
gradient_checkpointing: true
flash_attention: false
attn_implementation: eager
lora_mlp_kernel: false
lora_qkv_kernel: false
lora_o_kernel: false
lora_embedding_kernel: false
logging_steps: 1
warmup_steps: 0
saves_per_epoch: 1
save_total_limit: 1
trust_remote_code: false
YAML
```
-->

---

## 6. 在 NPU 上训练

单卡昇腾用 `--launcher python`（默认 accelerate 会找 CUDA）。训练日志里的 `"device": "npu:0"` 表示这一次跑在 NPU 上。

```shell #test id="train"
axolotl train /root/axolotl-qs/lora-npu.yml --launcher python 2>&1
```

输出结果如下：

```shell #test-result id="train"
...
  "device": "npu:0",
...Training completed!...
```
