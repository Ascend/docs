# torchtune

在单卡昇腾 NPU 上跑通 [torchtune](https://github.com/meta-pytorch/torchtune) 的最小 LoRA 微调链路：

- **安装**：`uv pip install torchtune`（二进制）与源码安装两条路径，拿到 `tune` CLI + 内置 recipes / configs。
- **LoRA 微调**：从 HuggingFace Hub 拉 `Qwen/Qwen2.5-0.5B-Instruct` 作为底座，用 `tune run lora_finetune_single_device` 配 `qwen2_5/0.5B_lora_single_device` 配置跑 3 步 LoRA 微调。
- **产物验证**：检查 LoRA 适配器（`adapter_config.json` + `adapter_model.pt`）落盘与 LoRA 矩阵配置。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，并按需完成物理机或容器内的设备挂载（`/dev/davinci*` 等）。

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
| torch | 2.11.0+cpu |
| torch_npu | 2.11.0 |
| torchtune | 最新 release 的源码/二进制 |
| huggingface_hub | 最新稳定版 |
| torchao | `<0.16` |
| 模型 | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |
| 数据集 | 本文自带的 50 条 alpaca 格式样例（写入 `data.json`，通过 `dataset=torchtune.datasets.alpaca_dataset` 切换到本地 JSON loader） |

## 前置安装

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
| 0     910B4               | OK            | 89.9        39                0    / 0             |
| 0                         | 0000:41:00.0  | 0           0    / 0          2922 / 32768         |
+===========================+===============+====================================================+
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| No running processes found in NPU 0                                                            |
+===========================+===============+====================================================+
```

```{admonition} Note
:class: note
如果 `npu-smi` 不存在，请回到 [Ascend 官方快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 补装驱动。
```

检查 Python 版本：

```shell #test id="check-py"
python --version
```

输出结果如下：

```shell #test-result id="check-py" fuzzy='xxx'
Python 3.12.xxx
```

对齐上游 pin 装 `torch` / `torch_npu`：

```shell #test-setup
uv pip install -f https://mirrors.aliyun.com/pytorch-wheels/cpu torch==2.11.0
uv pip install --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi torch_npu==2.11.0
```

检查 torch / torch_npu 是否装好且 NPU 设备可用(下面的命令用 Python 执行)：

```python #test id="check-torch"
import torch, torch_npu
print('torch=', torch.__version__)
print('torch_npu=', torch_npu.__version__)
print('is_available:', torch.npu.is_available())
print('count:', torch.npu.device_count())
```

输出结果如下：

```shell #test-result id="check-torch"
torch= 2.11.0+cpu
torch_npu= 2.11.0
is_available: True
count: 1
```

```{admonition} Note
:class: note
如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。
```

安装 `huggingface_hub`（用于从 HuggingFace Hub 下载底座模型）+ `torchao`：

<!-- torchao 两个 API 在新版都改了位置/签名，torchtune v0.6.1 都没跟上：
  * 0.18 起 NF4Tensor 挪到 torchao.prototype.dtypes，
     `torchtune.modules.common_utils:19` 的 `from torchao.dtypes.nf4tensor import NF4Tensor` 炸；
  * 0.16 起 `int4_weight_only()` 函数被 class-based 的 `Int4WeightOnlyConfig` 取代，
     `torchtune` 内部还在用老 API。所以 pin 在最后一个两个 import 都 OK 的 0.15 系列。 -->
```shell #test-setup
uv pip install huggingface_hub
uv pip install 'torchao<0.16'
```

打印安装版本(下面的命令用 Python 执行)：

```python #test id="install-deps"
import huggingface_hub, torchao
print('huggingface_hub', huggingface_hub.__version__)
print('torchao', torchao.__version__)
```

输出结果如下：

```shell #test-result id="install-deps" fuzzy='xxx'
huggingface_hub xxx
torchao xxx
```

```{admonition} Note
:class: note
xxx 表示实际安装的版本号。
```

## 安装 torchtune

### 使用 uv 进行安装

通过 PyPI 镜像直接装最新 release 的二进制 wheel：

```shell #test id="torchtune-install-binary"
uv pip install --index-url https://mirrors.aliyun.com/pypi/simple torchtune
python -c "import torchtune; print('torchtune', torchtune.__version__)"
```

输出结果类似如下：

```shell #test-result id="torchtune-install-binary" fuzzy='xxx'
torchtune xxx
```

```{admonition} Note
:class: note
xxx 表示最新的版本号。
```

<!--
```shell #test-setup
uv pip uninstall torchtune -y
```
-->

### 从源码安装

<!-- 工作流注入的 UPSTREAM_REF（最新 release tag）通过这个隐藏的 #test-setup 捕获并注入到下方 install 命令中-->
<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

克隆上游仓库并 checkout 到当前 torchtune 的最新 release tag，安装并且验证：

<!--  非 editable 安装：uv 的 `pip install -e .` (PEP 660) 会把 torchtune 当
  namespace package 加载，导致 `torchtune.__file__` 为 None，进而使
  torchtune/_cli/cp.py:15 的 `Path(torchtune.__file__).parent.parent` 抛
  TypeError，`tune --help` 等所有 CLI 调用都炸。新手 quick-start 不需要 hot-reload。 -->
```shell #test id="torchtune-install-source" load="upstream_ref>>ref"
git clone --depth 1 --branch <ref> https://github.com/meta-pytorch/torchtune.git
cd torchtune
uv pip install .
python -c "import torchtune; print('torchtune', torchtune.__version__)"
```

```{admonition} Note
:class: note
`<ref>` 替换为 torchtune 当前的最新 release 版本。
```

输出结果类似如下：

```shell #test-result id="torchtune-install-source" fuzzy='xxx'
torchtune xxx
```

```{admonition} Note
:class: note
xxx 表示最新的版本号。
```

## CLI 自检

`tune --help` 列出 torchtune 的子命令：

```shell #test id="tune-help"
tune --help
```

输出结果类似如下：

```shell #test-result id="tune-help"
usage: tune [-h] {download,ls,cp,run,validate,cat} ...

Welcome to the torchtune CLI!

options:
  -h, --help            show this help message and exit
...
```

## 使用样例：单卡 LoRA 微调 Qwen2.5-0.5B

对应上游 [First Finetune Tutorial](https://pytorch.org/torchtune/0.6/tutorials/first_finetune_tutorial.html)，在单卡昇腾 NPU 上跑通 3 步 LoRA 微调。

### 下载基础模型

默认使用 **HuggingFace Hub** 进行模型下载：

```shell #test-setup store="model_path"
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-0.5B-Instruct'))" | tail -n 1
```

输出类似：

```text
/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/xxxxxxxx
```

### 准备本地样例数据

本文档用 50 条 alpaca 格式的样例写到 `data.json`，再通过 `dataset=torchtune.datasets.alpaca_dataset` 切到本地 JSON 路径：

```shell #test-setup store="data_path"
cat > data.json <<'JSON'
[
  {"instruction": "Briefly explain why the sky looks blue.", "input": "", "output": "The sky appears blue because shorter-wavelength sunlight is scattered in all directions by the gases in Earth's atmosphere."},
  {"instruction": "Name a prime number below ten.", "input": "", "output": "7"},
  {"instruction": "Translate 'hello' into Spanish.", "input": "", "output": "Hola"},
  {"instruction": "What is the capital of France?", "input": "", "output": "Paris"},
  {"instruction": "Compute 12 times 13.", "input": "", "output": "156"},
  {"instruction": "Who wrote 'Pride and Prejudice'?", "input": "", "output": "Jane Austen"},
  {"instruction": "Give a synonym for 'happy'.", "input": "", "output": "Joyful"},
  {"instruction": "What is the boiling point of water in Celsius?", "input": "", "output": "100"},
  {"instruction": "List three primary colors.", "input": "", "output": "Red, yellow, blue"},
  {"instruction": "Define photosynthesis in one sentence.", "input": "", "output": "Photosynthesis is the process by which plants convert light energy into chemical energy stored as glucose."},
  {"instruction": "What's the chemical symbol for gold?", "input": "", "output": "Au"},
  {"instruction": "Name a planet with rings.", "input": "", "output": "Saturn"},
  {"instruction": "How many continents are there?", "input": "", "output": "Seven"},
  {"instruction": "Translate 'thank you' into Japanese.", "input": "", "output": "ありがとう (arigatou)"},
  {"instruction": "What is the largest ocean on Earth?", "input": "", "output": "The Pacific Ocean"},
  {"instruction": "Define 'algorithm'.", "input": "", "output": "A step-by-step procedure for solving a problem or accomplishing a task."},
  {"instruction": "Name the first president of the United States.", "input": "", "output": "George Washington"},
  {"instruction": "What's 25 percent of 200?", "input": "", "output": "50"},
  {"instruction": "Translate 'goodbye' into German.", "input": "", "output": "Auf Wiedersehen"},
  {"instruction": "List the four fundamental forces of nature.", "input": "", "output": "Gravitational, electromagnetic, strong nuclear, and weak nuclear forces."},
  {"instruction": "What's the square root of 64?", "input": "", "output": "8"},
  {"instruction": "Name the author of '1984'.", "input": "", "output": "George Orwell"},
  {"instruction": "What is the speed of light in vacuum (m/s, approximate)?", "input": "", "output": "About 3 x 10^8 meters per second."},
  {"instruction": "Translate 'cat' into Italian.", "input": "", "output": "Gatto"},
  {"instruction": "Define 'gravity'.", "input": "", "output": "Gravity is the force by which a planet or other body draws objects toward its center."},
  {"instruction": "List three even numbers.", "input": "", "output": "2, 4, 6"},
  {"instruction": "What's H2O commonly known as?", "input": "", "output": "Water"},
  {"instruction": "Name the longest river in the world.", "input": "", "output": "The Nile (commonly cited) or the Amazon (by discharge volume)."},
  {"instruction": "Translate 'yes' into Mandarin Chinese (pinyin).", "input": "", "output": "Shi (是)"},
  {"instruction": "What is the periodic table?", "input": "", "output": "A tabular arrangement of chemical elements organized by atomic number."},
  {"instruction": "Compute 7 squared.", "input": "", "output": "49"},
  {"instruction": "Define 'democracy'.", "input": "", "output": "A system of government in which power is vested in the people, who exercise it directly or through elected representatives."},
  {"instruction": "Name the largest mammal on Earth.", "input": "", "output": "The blue whale"},
  {"instruction": "Translate 'red' into French.", "input": "", "output": "Rouge"},
  {"instruction": "What is the smallest unit of life?", "input": "", "output": "The cell"},
  {"instruction": "Define 'ecosystem'.", "input": "", "output": "A community of living organisms together with the nonliving components of their environment, interacting as a system."},
  {"instruction": "List three noble gases.", "input": "", "output": "Helium, neon, argon"},
  {"instruction": "What's the tallest mountain on Earth?", "input": "", "output": "Mount Everest"},
  {"instruction": "Translate 'house' into Korean (romanized).", "input": "", "output": "Jip (집)"},
  {"instruction": "What year did World War II end?", "input": "", "output": "1945"},
  {"instruction": "Define 'metabolism'.", "input": "", "output": "The chemical processes by which an organism maintains life, including converting food to energy."},
  {"instruction": "Name the gas plants take in for photosynthesis.", "input": "", "output": "Carbon dioxide (CO2)"},
  {"instruction": "Compute 144 divided by 12.", "input": "", "output": "12"},
  {"instruction": "Translate 'book' into Portuguese.", "input": "", "output": "Livro"},
  {"instruction": "What is the hardest natural substance?", "input": "", "output": "Diamond"},
  {"instruction": "Define 'protein'.", "input": "", "output": "A large biomolecule composed of amino acids, essential for the structure and function of cells."},
  {"instruction": "List three common programming languages.", "input": "", "output": "Python, JavaScript, C++"},
  {"instruction": "What's the currency of Japan?", "input": "", "output": "Japanese yen (JPY)"},
  {"instruction": "Name the process by which liquid becomes gas.", "input": "", "output": "Evaporation (or vaporization)"},
  {"instruction": "Translate 'sun' into Russian (transliterated).", "input": "", "output": "Solntse (солнце)"}
]
JSON
echo "${PWD}/data.json"
```

输出结果类似：

```text
/path/to/ascend_docs/data.json
```

### 跑 3 步 LoRA 微调

```shell #test id="torchtune-train" load="model_path>>model_path" load="data_path>>data_path"
ASCEND_RT_VISIBLE_DEVICES=0 tune run lora_finetune_single_device \
  --config qwen2_5/0.5B_lora_single_device \
  device=npu \
  checkpointer.checkpoint_dir="<model_path>" \
  tokenizer.path="<model_path>/vocab.json" \
  tokenizer.merges_file="<model_path>/merges.txt" \
  dataset=torchtune.datasets.alpaca_dataset \
  dataset.source=json \
  dataset.data_files="<data_path>" \
  metric_logger=torchtune.training.metric_logging.StdoutLogger \
  ~metric_logger.log_dir \
  log_peak_memory_stats=False \
  output_dir="${PWD}/output" \
  max_steps_per_epoch=3 \
  epochs=1 \
  log_every_n_steps=1
```

```{admonition} Note
:class: note
`<model_path>` 在运行时指向「下载基础模型」一节 `snapshot_download` 命令打印的模型缓存路径；`<data_path>` 在运行时指向「准备本地样例数据」一节生成的 `data.json` 路径。
```

输出结果如下：

```shell #test-result id="torchtune-train" fuzzy='xxx' fuzzy='...'
Step 1 | loss:xxx lr:xxx tokens_per_second_per_gpu:xxx
Step 2 | loss:xxx lr:xxx tokens_per_second_per_gpu:xxx
Step 3 | loss:xxx lr:xxx tokens_per_second_per_gpu:xxx
...
```

捕获 LoRA checkpoint 目录路径供下一步验证：

```shell #test-setup store="checkpoint"
ls -dt ${PWD}/output/*/ | head -n 1
```

输出类似：

```text
/path/to/ascend_docs/output/epoch_0/
```

### 验证 LoRA 适配器落盘

torchtune 默认落 `adapter_config.json` + `adapter_model.pt`（底座权重不动）：

```shell #test id="torchtune-verify-ckpt" load="checkpoint>>ckpt"
test -f "<ckpt>/adapter_config.json" && test -f "<ckpt>/adapter_model.pt" && echo "adapter files present"
```

输出结果如下：

```shell #test-result id="torchtune-verify-ckpt"
adapter files present
```

```{admonition} Note
:class: note
`<ckpt>` 在运行时替换为「跑 3 步 LoRA 微调」小节落盘的 LoRA checkpoint 目录。
```

捕获 checkpoint 里 LoRA 矩阵的统计信息(下面的命令用 Python 执行)：

```python #test id="torchtune-trainable" load="checkpoint>>ckpt"
import json
cfg = json.load(open('<ckpt>/adapter_config.json'))
print('rank', cfg['r'], 'lora_alpha', cfg['lora_alpha'], 'targets', cfg['target_modules'])
```

输出结果如下：

```shell #test-result id="torchtune-trainable" fuzzy='xxx'
rank xxx lora_alpha xxx targets xxx
```

## 外部链接

- GitHub：[meta-pytorch/torchtune](https://github.com/meta-pytorch/torchtune)
- 文档中心：[torchtune Docs](https://pytorch.org/torchtune/)
