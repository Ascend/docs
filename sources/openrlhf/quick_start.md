# OpenRLHF

本文在单卡昇腾 NPU 上安装 [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)，并对 [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) 完成 4 步 LoRA 监督微调。

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**），单卡。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit 与驱动已安装，并能 `source set_env.sh` |
| Python | 3.12 |
| PyTorch | `torch==2.11.0` 与 `torch_npu==2.11.0` |
| OpenRLHF | 从 GitHub 克隆当前正式 Release，见第 6 节 |
| 模型 | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |

**配套机器**：Atlas 900 A2 PODc（Ascend 910B4）。**配套镜像**：`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`。

上游安装说明见 [OpenRLHF Quick Start — Installation](https://github.com/OpenRLHF/OpenRLHF#installation)。本文镜像是 aarch64，不能走那里的 `pip install openrlhf`，改用克隆源码。

---

## 1. 加载 CANN 环境

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:/usr/local/bin:$PATH
export PYTHONNOUSERSITE=1
```

`PYTHONNOUSERSITE=1` 让 Python 忽略用户目录里的包，避免旧版 CANN 相关包干扰依赖解析。

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

从 PyTorch CPU 轮子和华为昇腾索引安装与 CANN 9.1.0 匹配的 `torch==2.11.0` / `torch_npu==2.11.0`，并一并装上 `numpy`、`pyyaml`。版本行出现 `2.11.0+cpu` 属正常，`npu_available` 须为 `True`。

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

## 4. 让 OpenRLHF 走 NPU

OpenRLHF 通过 `torch.cuda` 选设备。下面把 `transfer_to_npu` 写入 `sitecustomize.py`，新的 Python 进程会把 `torch.cuda` 映射到 NPU。

工作目录为 `/root/openrlhf-qs`：

```shell
mkdir -p /root/openrlhf-qs
```

保存为 `/root/openrlhf-qs/enable_npu.py`：

```python
from pathlib import Path
import site

path = Path(site.getsitepackages()[0]) / "sitecustomize.py"
path.write_text("from torch_npu.contrib import transfer_to_npu\n")
print(path)
```

<!--
```shell #test-setup
mkdir -p /root/openrlhf-qs
cat > /root/openrlhf-qs/enable_npu.py <<'PY'
from pathlib import Path
import site

path = Path(site.getsitepackages()[0]) / "sitecustomize.py"
path.write_text("from torch_npu.contrib import transfer_to_npu\n")
print(path)
PY
```
-->

```shell #test id="enable-npu"
python /root/openrlhf-qs/enable_npu.py
python -c "import torch; print(torch.zeros(1, device='cuda').device)"
```

输出结果如下：

```shell #test-result id="enable-npu"
...sitecustomize.py
npu:0
```

---

## 5. 补一份 flash_attn 占位包

OpenRLHF 启动训练时会立刻 `import flash_attn`，与是否启用 Flash Attention 无关。昇腾上没有 CUDA 版这个包，下面写入一份只满足 import 的占位包。第 8 节用 `--ds.attn_implementation eager`，不要加 `--ds.packing_samples`。

保存为 `/root/openrlhf-qs/write_flash_attn_stub.py`：

```python
from pathlib import Path

root = Path("/root/openrlhf-qs/flash_attn")
(root / "utils").mkdir(parents=True, exist_ok=True)
(root / "__init__.py").write_text("")
(root / "utils" / "__init__.py").write_text("")
(root / "bert_padding.py").write_text(
    "from einops import rearrange\n"
    "\n"
    "def index_first_axis(*args, **kwargs):\n"
    "    raise RuntimeError('packing needs CUDA flash_attn')\n"
    "\n"
    "def unpad_input(*args, **kwargs):\n"
    "    raise RuntimeError('packing needs CUDA flash_attn')\n"
    "\n"
    "def pad_input(*args, **kwargs):\n"
    "    raise RuntimeError('packing needs CUDA flash_attn')\n"
)
(root / "utils" / "distributed.py").write_text(
    "def all_gather(*args, **kwargs):\n"
    "    raise RuntimeError('ring attention needs CUDA flash_attn')\n"
)
print(root)
```

<!--
```shell #test-setup
cat > /root/openrlhf-qs/write_flash_attn_stub.py <<'PY'
from pathlib import Path

root = Path("/root/openrlhf-qs/flash_attn")
(root / "utils").mkdir(parents=True, exist_ok=True)
(root / "__init__.py").write_text("")
(root / "utils" / "__init__.py").write_text("")
(root / "bert_padding.py").write_text(
    "from einops import rearrange\n"
    "\n"
    "def index_first_axis(*args, **kwargs):\n"
    "    raise RuntimeError('packing needs CUDA flash_attn')\n"
    "\n"
    "def unpad_input(*args, **kwargs):\n"
    "    raise RuntimeError('packing needs CUDA flash_attn')\n"
    "\n"
    "def pad_input(*args, **kwargs):\n"
    "    raise RuntimeError('packing needs CUDA flash_attn')\n"
)
(root / "utils" / "distributed.py").write_text(
    "def all_gather(*args, **kwargs):\n"
    "    raise RuntimeError('ring attention needs CUDA flash_attn')\n"
)
print(root)
PY
```
-->

```shell #test id="flash-attn-stub"
python /root/openrlhf-qs/write_flash_attn_stub.py
```

输出结果如下：

```shell #test-result id="flash-attn-stub"
/root/openrlhf-qs/flash_attn
```

---

## 6. 克隆 OpenRLHF 并安装依赖

把 `<ref>` 换成 [Releases](https://github.com/OpenRLHF/OpenRLHF/releases) 里当前正式 tag，克隆后把源码目录加入 `PYTHONPATH`。

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

<!--
```shell #test-setup load="upstream_ref>>ref"
ci='/root/.cache/huggingface/openrlhf'
cached="$ci/OpenRLHF"
dest='/root/openrlhf-qs/OpenRLHF'
if [ -d "$cached/.git" ]; then
  tag=$(git -C "$cached" describe --tags --exact-match 2>/dev/null || true)
  if [ "$tag" = "<ref>" ]; then
    mkdir -p /root/openrlhf-qs
    rm -rf "$dest"
    cp -a "$cached" "$dest"
  else
    rm -rf "$cached"
  fi
fi
```
-->

```shell #test id="clone-openrlhf" load="upstream_ref>>ref"
mkdir -p /root/openrlhf-qs
if [ ! -d /root/openrlhf-qs/OpenRLHF/.git ]; then
  GIT_TERMINAL_PROMPT=0 GIT_HTTP_VERSION=HTTP/1.1 git clone --depth 1 --branch "<ref>" \
    https://github.com/OpenRLHF/OpenRLHF.git /root/openrlhf-qs/OpenRLHF
fi
export PYTHONPATH=/root/openrlhf-qs/OpenRLHF:/root/openrlhf-qs:${PYTHONPATH:-}
python -c "import openrlhf; print(openrlhf.__file__)"
```

输出结果如下：

```shell #test-result id="clone-openrlhf"
/root/openrlhf-qs/OpenRLHF/openrlhf/__init__.py
```

<!--
```shell #test-setup
ci='/root/.cache/huggingface/openrlhf'
src='/root/openrlhf-qs/OpenRLHF'
cached="$ci/OpenRLHF"
if [ -d "$src/.git" ]; then
  mkdir -p "$ci"
  rm -rf "${cached}.part" "$cached"
  cp -a "$src" "${cached}.part"
  mv "${cached}.part" "$cached"
fi
```
-->

安装 SFT 路径需要的包：

```shell #test id="install-deps"
python -m pip install \
  accelerate aiohttp datasets 'deepspeed==0.19.5' einops 'grpcio>=1.74.0' \
  'huggingface_hub>=1.0.0' jsonlines loralib optimum 'optree>=0.15.0' \
  packaging peft pylatexenc tensorboard torchdata torchmetrics tqdm \
  'transformers==5.15.0' transformers_stream_generator
python -c "import deepspeed, transformers; print('deepspeed', deepspeed.__version__); print('transformers', transformers.__version__)"
```

输出结果如下：

```shell #test-result id="install-deps"
...deepspeed 0.19.5
transformers 5.15.0
```

---

## 7. 准备模型和数据

从 Hugging Face Hub 下载底座模型，并链到工作目录：

```shell #test id="download-model"
mkdir -p /root/openrlhf-qs
rm -f /root/openrlhf-qs/model
ln -s "$(python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("Qwen/Qwen2.5-0.5B-Instruct"))' | grep '^/' | tail -n 1)" /root/openrlhf-qs/model
ls /root/openrlhf-qs/model/config.json
```

输出结果如下：

```shell #test-result id="download-model"
/root/openrlhf-qs/model/config.json
```

训练数据如下，保存为 `/root/openrlhf-qs/tiny_sft.jsonl`：

```json
{"question": "Translate to English: 你好", "response": "Hello"}
{"question": "Name a color.", "response": "Blue"}
{"question": "Add 2 and 3.", "response": "5"}
{"question": "Say yes or no.", "response": "Yes"}
```

<!--
```shell #test-setup
cat > /root/openrlhf-qs/tiny_sft.jsonl <<'EOF'
{"question": "Translate to English: 你好", "response": "Hello"}
{"question": "Name a color.", "response": "Blue"}
{"question": "Add 2 and 3.", "response": "5"}
{"question": "Say yes or no.", "response": "Yes"}
EOF
```
-->

---

## 8. 在 NPU 上训练

下面用 4 条样本做 1 个 epoch 的 LoRA，确认能在 NPU 上跑完。日志里的 `npu::npu_format_cast` 和 `device='npu'` 表示这一次跑在 NPU 上。

```shell #test id="train"
export PYTHONPATH=/root/openrlhf-qs/OpenRLHF:/root/openrlhf-qs:${PYTHONPATH:-}
deepspeed --module openrlhf.cli.train_sft \
  --data.max_len 256 \
  --data.dataset /root/openrlhf-qs/tiny_sft.jsonl \
  --data.input_key question \
  --data.output_key response \
  --train.batch_size 1 \
  --train.micro_batch_size 1 \
  --data.max_samples 4 \
  --model.model_name_or_path /root/openrlhf-qs/model \
  --ckpt.output_dir /root/openrlhf-qs/ckpt \
  --ckpt.save_steps -1 \
  --logger.logging_steps 1 \
  --eval.steps -1 \
  --ds.zero_stage 2 \
  --ds.adam_offload \
  --train.max_epochs 1 \
  --ds.param_dtype bf16 \
  --ds.attn_implementation eager \
  --adam.lr 5e-6 \
  --ds.lora.rank 8 \
  --model.gradient_checkpointing_enable \
  --data.dataloader_num_workers 0 \
  2>&1
```

输出结果如下：

```shell #test-result id="train"
...Setting ASCEND_RT_VISIBLE_DEVICES=0...
...npu::npu_format_cast...
...device='npu'...
...Train step of epoch 0: 100%...
...exits successfully.
```

---

## 常见问题

| 现象 | 处理 |
| --- | --- |
| `pip install openrlhf` 报平台或 wheel 错误 | 回到第 6 节克隆源码，不要装 PyPI 包装 |
| `npu_available False` | 回到第 3 节核对 CANN、驱动与 `torch_npu` 安装 |
| 写入 `sitecustomize.py` 后张量仍在 `cpu` | 回到第 4 节确认 `transfer_to_npu` 已写入且 import 显示 `npu:0` |
| `ModuleNotFoundError: flash_attn` | 回到第 5 节写入占位包，并确认 `PYTHONPATH` 含 `/root/openrlhf-qs` |
| `packing needs CUDA flash_attn` / `ring attention needs CUDA flash_attn` | 不要加 `--ds.packing_samples` 或 `--ds.ring_attn_size` |
| FusedAdam / decorator / scipy 相关报错 | 确认训练命令含 `--ds.adam_offload` |
