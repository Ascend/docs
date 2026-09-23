# axolotl

本文在单卡昇腾 NPU 上安装 [axolotl](https://github.com/axolotl-ai-cloud/axolotl)，并对 [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) 完成 3 步 LoRA 监督微调。

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**），单卡。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit 与驱动已安装，并能 `source set_env.sh`。版本按 [昇腾软件配套清单](https://www.hiascend.com/developer/download/compatibility) 选择 |
| Python | 落在官方配套表给出的范围内，并满足 axolotl 下限；当前正式版要求 `>=3.10` |
| PyTorch | 安装官方当前推荐的 `torch` 与 `torch_npu`，见 [CANN 与 PyTorch 配套表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md) 和 [PyTorch 安装包](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download) |
| axolotl | 从 PyPI 安装当前正式版；用 `--no-build-isolation` 对着已装的 torch 装，不要加 `[deepspeed]` |
| 模型 | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |

---

## 1. 加载 CANN 环境

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:/usr/local/bin:$PATH
```

---

## 2. 检查环境

确认 NPU 在线；找不到 `npu-smi` 时参阅 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)。

```shell
npu-smi info
```

---

## 3. 安装 PyTorch NPU 栈

```shell #test id="install-torch"
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  torch_npu numpy pyyaml
python -c "import numpy, yaml, torch, torch_npu; print('torch', torch.__version__); print('torch_npu', torch_npu.__version__); print('npu_available', torch.npu.is_available())"
```

<!--
```shell #test-result id="install-torch"
...
npu_available True
```
-->

---

## 4. 安装 axolotl

NPU 栈就绪后，用 `--no-build-isolation` 安装当前正式版，让构建对着已装的 torch。

<!--
```shell #test-setup store="axolotl_ver"
echo "${UPSTREAM_REF#v}"
```
-->

```shell #test id="install-axolotl" load="axolotl_ver>>ver"
python -m pip install --no-build-isolation "axolotl==<ver>"
python -c "import axolotl; print('axolotl', axolotl.__version__)"
```

<!--
```shell #test-result id="install-axolotl" load="axolotl_ver>>ver"
...axolotl <ver>
```
-->

---

## 5. 准备数据和配置

本示例的工作目录为 `/root/axolotl-qs`。先从 Hugging Face Hub 把 [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) 下载到 `/root/axolotl-qs/model`，再按下面内容保存训练数据和 LoRA 配置。

```shell
mkdir -p /root/axolotl-qs
python -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", local_dir="/root/axolotl-qs/model")'
```

<!--
```shell #test-setup
mkdir -p /root/axolotl-qs
rm -rf /root/axolotl-qs/model
export HF_HOME=/root/.cache/huggingface
ln -s "$(python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("Qwen/Qwen2.5-0.5B-Instruct"))' | grep '^/' | tail -n 1)" /root/axolotl-qs/model
test -f /root/axolotl-qs/model/config.json
```
-->

确认模型文件已就位：

```shell #test id="download-model"
ls /root/axolotl-qs/model/config.json
```

<!--
```shell #test-result id="download-model"
/root/axolotl-qs/model/config.json
```
-->

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
seed: 42
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
seed: 42
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

单卡昇腾用 `--launcher python`，默认 accelerate 会找 CUDA。配置里的 `seed: 42` 用来固定随机初始化。训练日志里的 `"device": "npu:0"` 表示这一次跑在 NPU 上；每步会打出 `loss`，3 步结束后有 `train_loss`。第一步的 `loss` 应是 `12.49`；后几步在同一配置下仍可能有小幅波动。

```shell #test id="train"
axolotl train /root/axolotl-qs/lora-npu.yml --launcher python 2>&1
```

完整输出较长，其中应包含：

```shell #test-result id="train"
...
  "device": "npu:0",
...{'loss': '12.49'...
...{'loss': ...
...{'loss': ...
...{'train_loss': ...
...Training completed!...
```
