# xtuner

在昇腾 NPU 上跑通 [xtuner](https://github.com/InternLM/xtuner) 的最小微调链路。

本文档以 **Qwen1.5-1.8B-Chat** + Colorist 指令微调数据为例，端到端走通「权重下载 → 配置修改 → 单/多卡训练 → LoRA 合并 → chat 推理」全链路：

- **安装**：PyPI 二进制与源码两条安装路径 + import / CLI 模式自检。
- **LLM 微调**：单卡与双卡 DDP（hccl 后端）LoRA 微调 Qwen1.5-1.8B-Chat（各 5 轮迭代跑通链路）。
- **合并与对话**：`pth_to_hf` 转换 + LoRA 合并 + merged / adapter 两种 `xtuner chat` 对话。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，并按需完成物理机或容器内的设备挂载。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）

### 本文档示例使用的版本

**配套机器**：

- **机器类型**：Atlas 900 A2 PODc（Ascend 910B4）
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
| torchvision | 0.26.0+cpu |
| huggingface_hub | 最新稳定版 |
| xtuner | 最新 release tag |
| 模型 | [Qwen/Qwen1.5-1.8B-Chat](https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat) |
| 数据集 | [burkelibbey/colors](https://huggingface.co/datasets/burkelibbey/colors)（Colorist） |

### 前置安装

确认能看到 NPU 设备：

```shell
npu-smi info
```

`npu-smi info` 完整输出类似：

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
| 1     910B4               | OK            | 89.9        39                0    / 0             |
| 1                         | 0000:42:00.0  | 0           0    / 0          2922 / 32768         |
+===========================+===============+====================================================+
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| No running processes found in NPU 0                                                            |
| No running processes found in NPU 1                                                            |
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

```{admonition} Note
:class: note
xxx 表示 Python 的补丁版本号。
```

装 `torch` / `torch_npu`：

```shell #test-setup
uv pip install -f https://mirrors.aliyun.com/pytorch-wheels/cpu torch==2.11.0
uv pip install --extra-index-url https://mirrors.aliyun.com/pypi/simple torch_npu==2.11.0
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
count: 2
```

```{admonition} Note
:class: note
如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。
```

装 `huggingface_hub`（用于从 HuggingFace Hub 下载模型权重与数据集）：

```shell #test-setup
uv pip install huggingface_hub
```

## 安装 xtuner

### 方式一：PyPI 二进制安装

```shell #test id="xtuner-install-binary"
uv pip install --index-url https://mirrors.aliyun.com/pypi/simple --no-deps xtuner
apt-get update -qq >/dev/null && apt-get install -y -qq libgl1 libglib2.0-0 >/dev/null
uv pip install -f https://mirrors.aliyun.com/pytorch-wheels/cpu 'mmengine==0.10.6' 'transformers==4.48.0' 'peft>=0.14.0' \
    'datasets>=3.2.0,<4.0.0' einops loguru openpyxl 'scikit-image' scipy \
    SentencePiece tiktoken transformers_stream_generator cyclopts \
    'opencv-python-headless<=4.12.0.88' 'torchvision==0.26.0+cpu' timm pyarrow pydantic tensorboard \
    xxhash imageio 'py-libnuma' GitPython
python -c "import xtuner; from xtuner.version import __version__; print('xtuner', __version__)"
```

输出结果类似如下：

```shell #test-result id="xtuner-install-binary" fuzzy='xxx'
xtuner xxx
```

```{admonition} Note
:class: note
xxx 表示最新的版本号
```

### 方式二：源码安装

换成源码版（覆盖方式一的二进制安装）：

<!--
```shell #test-setup
uv pip uninstall xtuner -y
```
-->

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

克隆上游仓库并 checkout 到最新 release tag，装 xtuner 本体 + 运行依赖，最后打印版本号验证：

```shell #test id="xtuner-install-source" load="upstream_ref>>ref"
[ -d xtuner ] || git clone --depth 1 --branch <ref> https://github.com/InternLM/xtuner.git
cd xtuner
uv pip install --no-deps -e .
apt-get update -qq >/dev/null && apt-get install -y -qq libgl1 libglib2.0-0 >/dev/null
uv pip install -f https://mirrors.aliyun.com/pytorch-wheels/cpu 'mmengine==0.10.6' 'transformers==4.48.0' 'peft>=0.14.0' \
    'datasets>=3.2.0,<4.0.0' einops loguru openpyxl 'scikit-image' scipy \
    SentencePiece tiktoken transformers_stream_generator cyclopts \
    'opencv-python-headless<=4.12.0.88' 'torchvision==0.26.0+cpu' timm pyarrow pydantic tensorboard \
    xxhash imageio 'py-libnuma' GitPython
python -c "import xtuner; from xtuner.version import __version__; print('xtuner', __version__)"
```

```{admonition} Note
:class: note
`<ref>` 替换为 xtuner 当前的最新 release tag。
```

输出结果类似如下：

```shell #test-result id="xtuner-install-source" fuzzy='xxx'
xtuner xxx
```

```{admonition} Note
:class: note
xxx 表示最新的版本号
```

### 安装验证

装好后一次性验证：顶层包 + CLI 入口能被解析，`xtuner.entry_point.MODES` 覆盖本文档用到的 `train` / `list-cfg` / `chat` 三个子命令，且 `torchvision` 是与 `torch` 配对的 `+cpu` 构建——PyPI 上的 linux wheel 默认是 CUDA 构建（链 `libcudart`），配 `+cpu` torch 时 C++ 算子注册不上、`import` / 调用直接崩，所以必须 pin 到 `torchvision==0.26.0+cpu` 并在这里实测算子可用(下面的命令用 Python 执行)：

```python #test id="xtuner-verify-install"
import importlib.util as u
specs = {m: u.find_spec(m) for m in ['xtuner', 'xtuner.entry_point']}
for m, s in specs.items():
    print(m, 'ok' if s is not None else 'MISSING')
from xtuner.entry_point import MODES
print('modes_count:', len(MODES))
print('has_train:', 'train' in MODES)
print('has_list_cfg:', 'list-cfg' in MODES)
print('has_chat:', 'chat' in MODES)
import torch, torchvision
from torchvision.ops import nms
print('torch', torch.__version__)
print('torchvision', torchvision.__version__)
boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=torch.float)
scores = torch.tensor([0.9, 0.8])
print('nms kept:', nms(boxes, scores, 0.5).tolist())
```

输出结果如下：

```shell #test-result id="xtuner-verify-install" fuzzy='xxx'
xtuner ok
xtuner.entry_point ok
modes_count: xxx
has_train: True
has_list_cfg: True
has_chat: True
torch 2.11.0+cpu
torchvision 0.26.0+cpu
nms kept: [0]
```

```{admonition} Note
:class: note
xxx 表示 `MODES` 中子命令的个数。
```

## LLM 大模型微调

本文档的训练按 [xtuner legacy 快速上手模板](https://xtuner.readthedocs.io/zh-cn/latest/legacy/get_started/quickstart.html) 的顺序展开。

### 准备模型权重

本文档示例使用 **Qwen1.5-1.8B-Chat**——1.8B 参数 + TikToken BPE 分词，fp16 权重 ≈ 3.5 GB。

从 HuggingFace Hub 下载 Qwen1.5-1.8B-Chat 权重，并打印路径，作为后续章节中 `<weights_dir>` 的引用(下面的命令用 Python 执行)：

```python #test-setup store="xtuner_weights_path"
import os
from huggingface_hub import snapshot_download
path = snapshot_download('Qwen/Qwen1.5-1.8B-Chat', cache_dir='./qwen')
print(os.path.abspath(path))
```

权重落盘校验：

```shell #test id="xtuner-pull-weights"
ws=$(find ./qwen -name config.json -print -quit)
test -n "$ws" && test -f "$ws" && echo "weights_ok"
ls -la "$(dirname "$ws")" | head -1
```

输出结果类似：

```shell #test-result id="xtuner-pull-weights" fuzzy='xxx'
weights_ok
total xxx
```

```{admonition} Note
:class: note
xxx 为 `ls -la` 输出首行的 total 值（磁盘块数，随文件数变化）。
```

权重落到 `./qwen` 下的 HuggingFace Hub cache 目录（`models--Qwen--Qwen1.5-1.8B-Chat/snapshots/<sha>`，`find` 找到的 `config.json` 所在目录即底座模型根目录）。

### 准备微调数据集

Colorist 数据集（[HuggingFace 链接](https://huggingface.co/datasets/burkelibbey/colors)）：根据颜色描述给 16 进制颜色编码的指令微调集，几 MB。本文走 HuggingFace Hub 下载(下面的命令用 Python 执行)：

```python #test-setup
import os, shutil
from huggingface_hub import snapshot_download
path = snapshot_download('burkelibbey/colors', repo_type='dataset', cache_dir='/tmp/xtuner_hf_cache')
target = './colors'
if os.path.isdir(target):
    shutil.rmtree(target)
os.makedirs(target, exist_ok=True)
for entry in os.listdir(path):
    src = os.path.join(path, entry)
    dst = os.path.join(target, entry)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
print('dataset at', target)
```

验证数据集完整落盘——必需文件逐个检查，再列出目录实际内容：

```shell #test id="xtuner-pull-dataset"
# 两个必需文件逐个断言存在，缺了任何一个直接退出报错
for f in colors.jsonl README.md; do
    test -f "colors/$f" || { echo "MISSING: colors/$f"; exit 1; }
done
# 列出目录的实际内容
ls colors/
```

输出结果是 `./colors/` 的实际目录内容：

```shell #test-result id="xtuner-pull-dataset" disable_fuzzy
README.md
colors.jsonl
prepare.py
```

`colors.jsonl` 每行形如 `{"color": "#000000", "description": "..."}`。

#### 把 Colorist 数据集转成 Qwen chat 模板要的 OpenAI 格式

xtuner 的 Qwen 自定义 cfg 用 `openai_map_fn`，要求每行 JSON 形如：

```json
{"messages": [
  {"role": "user", "content": "Tell me about the color #000000"},
  {"role": "assistant", "content": "Pure Black: ..."}
]}
```

但 Colorist 数据集原始格式是 `{color, description}`，需要先转换一下(下面的命令用 Python 执行)：

```python #test-setup
import json, os
os.makedirs('./colors_openai', exist_ok=True)
src = './colors/colors.jsonl'
dst = './colors_openai/train.jsonl'
n = 0
with open(src) as fin, open(dst, 'w') as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        msg = {
            'messages': [
                {'role': 'user', 'content': f"Tell me about the color {row['color']}"},
                {'role': 'assistant', 'content': row['description']},
            ]
        }
        fout.write(json.dumps(msg, ensure_ascii=False) + '\n')
        n += 1
print(f'converted {n} rows -> {dst}')
```

验证转换结果——输出文件存在、行数与原始数据集一致、抽第一条验消息格式正确：

```shell #test id="xtuner-convert-colors"
test -f ./colors_openai/train.jsonl || { echo "MISSING: ./colors_openai/train.jsonl"; exit 1; }
# 行数一致（与 pull-dataset 后的 ./colors/colors.jsonl 对比）
src_n=$(wc -l < ./colors/colors.jsonl)
dst_n=$(wc -l < ./colors_openai/train.jsonl)
test "$src_n" = "$dst_n" || { echo "row count mismatch: src=$src_n dst=$dst_n"; exit 1; }
echo "converted ${dst_n} rows -> ./colors_openai/train.jsonl"
# 抽第一条做字面比对，验格式正确
head -1 ./colors_openai/train.jsonl | python -c "
import sys, json
row = json.loads(sys.stdin.read())
assert 'messages' in row, 'missing messages key'
assert isinstance(row['messages'], list) and len(row['messages']) == 2, 'expected 2 messages'
assert row['messages'][0]['role'] == 'user', 'first message role must be user'
assert row['messages'][1]['role'] == 'assistant', 'second message role must be assistant'
"
```

输出结果：

```shell #test-result id="xtuner-convert-colors" fuzzy='xxx'
converted xxx rows -> ./colors_openai/train.jsonl
```

```{admonition} Note
:class: note
xxx 为转换的样本行数（与 `./colors/colors.jsonl` 的行数一致）。
```

### 准备配置文件

检查 XTuner 自带大量开箱即用的 config(下面的命令用 Python 执行)：

```python #test id="xtuner-list-cfg"
from xtuner.configs import cfgs_name_path
names = sorted(cfgs_name_path.keys())
print('lines:', len(names))
print('head_first:', names[0] if names else '')
print('qwen_1_8b_chat_count:', sum(1 for n in names if 'qwen1_5_1_8b_chat_qlora_custom_sft_e1' in n))
```

输出结果如下：

```shell #test-result id="xtuner-list-cfg" fuzzy='xxx'
lines: xxx
head_first: xxx
qwen_1_8b_chat_count: xxx
```

```{admonition} Note
:class: note
xxx 依次表示 cfg 总数、字典序首个 cfg 名、匹配到的目标 cfg 个数。
```

从 list-cfg 拷一份 Qwen1.5-1.8B-Chat qlora + custom sft 配置到本地（xtuner v0.2.0 的这个 cfg 名字固定为 `qwen1_5_1_8b_chat_qlora_custom_sft_e1`）(下面的命令用 Python 执行)：

```python #test-setup store="xtuner_llm_cfg_path"
import os
import os.path as osp
import shutil
from xtuner.configs import cfgs_name_path
from xtuner.tools.copy_cfg import add_copy_suffix
config_name = 'qwen1_5_1_8b_chat_qlora_custom_sft_e1'
config_path = cfgs_name_path[config_name]
save_dir = '/tmp/xtuner_npu_llm_cfg'
save_path = osp.join(save_dir, add_copy_suffix(osp.basename(config_path)))
os.makedirs(save_dir, exist_ok=True)
shutil.copyfile(config_path, save_path)
print(save_path)
```

输出路径到下一节「修改配置文件」

### 修改配置文件

修改上一节「准备配置文件」产生的配置文件(下面的命令用 Python 执行)：

<!--
**替换 2 处路径**（在 cfg 里改对应行）：

1. `pretrained_model_name_or_path` → `<weights_dir>`（让训练读本地权重，不再联网）
2. `data_files` → `./colors_openai/train.jsonl` 的绝对路径（指向「数据转换」节生成的 jsonl）

**删除 2 处**（NPU 上跑不通）：

3. 删掉 `quantization_config` 块 + 顶部 `BitsAndBytesConfig` 导入：4-bit 量化依赖 bitsandbytes，aarch64 NPU 装不上
4. 删掉 `train_cfg` 里的 `max_epochs`：cfg 强制它与 `max_iters` 二选一，本节用 `max_iters=5` 限迭代数
-->

```python #test-setup load="xtuner_llm_cfg_path>>cfg" load="xtuner_weights_path>>weights_dir" store="xtuner_llm_cfg_path"
import re, os
path = '<cfg>'
weights_dir = '<weights_dir>'
with open(path) as f:
    text = f.read()

# 换权重路径：模板的 HF repo id → 本地权重目录
text, n = re.subn(
    r'pretrained_model_name_or_path = "Qwen/Qwen1\.5-1\.8B-Chat"',
    f"pretrained_model_name_or_path = {weights_dir!r}",
    text,
)
assert n == 1, f'weights path replaced {n} times (expected 1)'

# 换数据路径：占位符 → 转换后的 OpenAI jsonl 绝对路径
data_abs = os.path.abspath('./colors_openai/train.jsonl')
old = 'data_files = ["/path/to/json/file.json"]'
new = f'data_files = [{data_abs!r}]'
assert old in text, f'data placeholder not found: {old!r}'
text = text.replace(old, new)

# 删量化配置：quantization_config 块 + BitsAndBytesConfig 导入（QLoRA → plain LoRA）
text = re.sub(
    r',\s*\n\s*quantization_config=dict\(\n(?:\s+[^\n]*,\n)+?\s*\),\n',
    '\n',
    text,
    count=1,
)
text = re.sub(
    r'(from transformers import [^\n]*?), BitsAndBytesConfig',
    r'\1',
    text,
    count=1,
)

# 删 train_cfg 里的 max_epochs（TrainLoop 强制二选一；用 max_iters=5 限迭代数）
text, n = re.subn(
    r'train_cfg = dict\(type=TrainLoop, max_epochs=max_epochs\)',
    'train_cfg = dict(type=TrainLoop)',
    text,
)
assert n == 1, f'max_epochs removed {n} times (expected 1)'

with open(path, 'w') as f:
    f.write(text)
print(path)
```

```{admonition} Note
:class: note
- `<cfg>`：上一节「准备配置文件」拷 cfg 生成在 `/tmp/xtuner_npu_llm_cfg/` 下、带 `_copy.py` 后缀的那个文件路径
- `<weights_dir>`：「准备模型权重」下载的 Qwen 权重根目录，用 `find ./qwen -name config.json -print -quit | xargs dirname` 拿到
```

<!-- # py_compile 验 cfg 是合法 Python + 4 处修改都生效（grep 关键串）。
## 不用 mmengine.config.Config.fromfile：装了 torchvision 0.26.0+cpu 后它其实能跑，
## 但 py_compile 更轻量、输出确定，不引入 cfg 顶层 import 的执行副作用。 -->
验证修改结果(下面的命令用 Python 执行)：

```python #test id="xtuner-patch-cfg" load="xtuner_llm_cfg_path>>cfg" load="xtuner_weights_path>>weights_dir"
import py_compile
py_compile.compile('<cfg>', doraise=True)
print('cfg_compiles_ok')
with open('<cfg>') as f:
    text = f.read()
weights_dir = '<weights_dir>'
import os
data_abs = os.path.abspath('./colors_openai/train.jsonl')
checks = [
    ('weights', f"pretrained_model_name_or_path = '{weights_dir}'"),
    ('data', f"data_files = ['{data_abs}']"),
]
for name, expected in checks:
    assert expected in text, f'missing edit ({name}): {expected!r}'
assert 'quantization_config' not in text
assert 'BitsAndBytesConfig' not in text
assert 'train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)' not in text
print('cfg_patch_ok')
print(f'weights= {weights_dir}')
print(f'data= {data_abs}')
```

```{admonition} Note
:class: note
`<cfg>` 来自「修改配置文件」节保存的 cfg 路径；`<weights_dir>` 来自「准备模型权重」节保存的权重路径。
```

输出结果类似：

```shell #test-result id="xtuner-patch-cfg" fuzzy='xxx'
cfg_compiles_ok
cfg_patch_ok
weights= xxx
data= xxx
```

```{admonition} Note
:class: note
xxx 分别为权重目录与转换后数据文件的绝对路径；这里不验 `<cfg>` 训出来的实际效果，那要等下面「启动微调」章节真跑。
```

### 启动微调

训练日志（loss、学习率）每次跑都不一样，没法写死预期值。本文档只跑 5 轮迭代验证整条训练链路，不指望训出有意义结果。`EvaluateChatHook` 每轮打印 `Sample output:` 采样段，下面的验证命令检查 `.pth` 落盘 + 采样段格式。

#### 单卡

plain LoRA 而非 QLoRA：aarch64 NPU 上没有可用的 bitsandbytes 装法（PyPI 无 aarch64 wheel，source-build 各处报错）；Qwen1.5-1.8B fp16 ~3.5 GB，plain LoRA 在 32 GB NPU 上峰 RSS ≈ 10.3 GB。

跑最小训练：

```shell #test-setup id="xtuner-train-smoke-setup" load="xtuner_llm_cfg_path>>cfg"
cp <cfg> /tmp/xtuner_npu_smoke_single_cfg.py

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TORCH_NPU_USE_HCCL=1
mkdir -p /tmp/xtuner_sft_llm_out_single
set -o pipefail

# 5 处 --cfg-options override：
#   train_cfg.max_iters=5                    限 5 iter 跑通就够，不训 full epoch
#   default_hooks.checkpoint.interval=1      每 iter 落盘，方便验 .pth
#   custom_hooks.1.every_n_iters=1           EvaluateChatHook 每 iter 打 Sample output
#   custom_hooks.1.evaluation_inputs=...     覆盖 cfg 默认 prompt 改成 color
#   train_dataset.max_length=256             colors 样本短，2048 太浪费
#   optim_wrapper.accumulative_counts=1      5 iter 不需要梯度累积
cd xtuner
python -c "
import sys
sys.argv = ['xtuner.tools.train',
            '/tmp/xtuner_npu_smoke_single_cfg.py',
            '--work-dir', '/tmp/xtuner_sft_llm_out_single',
            '--cfg-options',
            'train_cfg.max_iters=5',
            'default_hooks.checkpoint.interval=1',
            'custom_hooks.1.every_n_iters=1',
            'custom_hooks.1.evaluation_inputs=[Tell me about the color #000000, Tell me about the color #FF5733]',
            'train_dataset.max_length=256',
            'optim_wrapper.accumulative_counts=1']
import xtuner.tools  # noqa: F401  触发 xtuner.__init__.py 完整加载，避免 namespace package 误判
from xtuner.tools import train
train.main()
" 2>&1 | tee /tmp/xtuner_sft_llm_out_single/train.log
```

```{admonition} Note
:class: note
`<cfg>` 来自「修改配置文件」节保存的 cfg 路径。
```

查 .pth 有没有落盘 + 训练日志里的 Sample output 段：

```shell #test id="xtuner-train-smoke"
ls -t /tmp/xtuner_sft_llm_out_single/*.pth 2>/dev/null | head -1
echo "---SAMPLE_OUTPUT---"
# EvaluateChatHook 每个 prompt 打一个 "Sample output:" 段（两个 evaluation_inputs 相邻成对）。
# awk 从第一个 "Sample output:" 行开始打印（跳过 mmengine 环境信息 dump 等 ~480 行前导日志），
# 到第 3 个段头（即下一轮 eval）截断——正好覆盖第一轮 eval 的两个 prompt 段。
awk '/Sample output:/{c++; if(c>2) exit} c>=1 {print}' /tmp/xtuner_sft_llm_out_single/train.log 2>/dev/null | sed -E 's/^[0-9]{2}\/[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2} - mmengine - (INFO|WARNING|ERROR|DEBUG) - //' | head -25
```

输出结果如下：

```shell #test-result id="xtuner-train-smoke" fuzzy='...'
/tmp/xtuner_sft_llm_out_single/iter_5.pth
---SAMPLE_OUTPUT---
Sample output:
<|im_start|>user
Tellmeaboutthecolor#000000<|im_end|>
<|im_start|>assistant
...
Sample output:
<|im_start|>user
Tellmeaboutthecolor#FF5733<|im_end|>
<|im_start|>assistant
...
```

#### 多卡（双卡 DDP）

跑最小训练：

```shell #test-setup id="xtuner-train-smoke-multi-setup" load="xtuner_llm_cfg_path>>cfg"
cp <cfg> /tmp/xtuner_npu_smoke_multi_cfg.py

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TORCH_NPU_USE_HCCL=1
mkdir -p /tmp/xtuner_sft_llm_out_multi
set -o pipefail

cd xtuner
NPROC_PER_NODE=2 python -c "
import sys
sys.argv = ['xtuner.tools.train',
            '/tmp/xtuner_npu_smoke_multi_cfg.py',
            '--work-dir', '/tmp/xtuner_sft_llm_out_multi',
            '--cfg-options',
            'train_cfg.max_iters=5',
            'default_hooks.checkpoint.interval=1',
            'custom_hooks.1.every_n_iters=1',
            'custom_hooks.1.evaluation_inputs=[Tell me about the color #000000, Tell me about the color #FF5733]',
            'train_dataset.max_length=256',
            'optim_wrapper.accumulative_counts=1']
import xtuner.tools  # noqa: F401
from xtuner.tools import train
train.main()
" 2>&1 | tee /tmp/xtuner_sft_llm_out_multi/train.log
```

```{admonition} Note
:class: note
`<cfg>` 来自「修改配置文件」节保存的 cfg 路径。
```

查 .pth + Sample output：

```shell #test id="xtuner-train-smoke-multi"
ls -t /tmp/xtuner_sft_llm_out_multi/*.pth 2>/dev/null | head -1
echo "---SAMPLE_OUTPUT---"
# 跟单卡同样的 awk + sed：从第一个 "Sample output:" 段头开始打印，
# 第 3 个段头截断，覆盖第一轮 eval 的两个 prompt 段。
awk '/Sample output:/{c++; if(c>2) exit} c>=1 {print}' /tmp/xtuner_sft_llm_out_multi/train.log 2>/dev/null | sed -E 's/^[0-9]{2}\/[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2} - mmengine - (INFO|WARNING|ERROR|DEBUG) - //' | head -25
```

输出结果如下：

```shell #test-result id="xtuner-train-smoke-multi" fuzzy='xxx' fuzzy='...'
/tmp/xtuner_sft_llm_out_multi/iter_xxx.pth
---SAMPLE_OUTPUT---
Sample output:
<|im_start|>user
Tellmeaboutthecolor#000000<|im_end|>
<|im_start|>assistant
...
Sample output:
<|im_start|>user
Tellmeaboutthecolor#FF5733<|im_end|>
<|im_start|>assistant
...
```

```{admonition} Note
:class: note
xxx 为实际落盘的迭代号。
```

### 模型转换 + LoRA 合并

训练产物是 LoRA adapter 的 `.pth`（只含 adapter 参数），要跟纯 base 模型对话需要两步：`pth_to_hf` 把 `.pth` 转成 HuggingFace 格式（PEFT adapter），`merge` 把 adapter 合并回 base：

```shell #test-setup load="xtuner_llm_cfg_path>>cfg" load="xtuner_weights_path>>weights_dir"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
src_pth=$(ls -t /tmp/xtuner_sft_llm_out_single/*.pth 2>/dev/null | head -1)
[ -n "$src_pth" ] || { echo "no .pth: 先跑上面的单卡训练"; exit 1; }
hf_dir="${src_pth%.pth}_hf"
merged_dir=/tmp/xtuner_sft_llm_out_single/merged
rm -rf "$hf_dir" "$merged_dir"
mkdir -p "$hf_dir" "$merged_dir"

# pth → hf（PEFT 格式：adapter_config.json + adapter_model.bin）
python -m xtuner.tools.model_converters.pth_to_hf \
    <cfg> \
    "$src_pth" \
    "$hf_dir"

# merge（PEFT adapter 合并回 base → sharded pytorch_model-*.bin，~3.5 GB）
python -m xtuner.tools.model_converters.merge \
    <weights_dir> \
    "$hf_dir" \
    "$merged_dir" \
    --max-shard-size 2GB
```

```{admonition} Note
:class: note
`<cfg>` 来自「启动微调」节保存的 cfg 路径；`<weights_dir>` 来自「准备模型权重」节保存的权重路径。
```

验证合并产物落盘：

```shell #test id="xtuner-merge-verify"
ls -t /tmp/xtuner_sft_llm_out_single/merged/*.bin 2>/dev/null | head -3
```

输出结果如下：

```shell #test-result id="xtuner-merge-verify" fuzzy='xxx'
/tmp/xtuner_sft_llm_out_single/merged/pytorch_modelxxx.bin
```

```{admonition} Note
:class: note
xxx 通配 shard 文件名（通常为 `-00001-of-00002` 形式）。
```

### 与模型对话

合并完权重后，用 `xtuner chat` 跟模型对话（本文档用等价的 `python -m xtuner.tools.chat` 入口）：`--prompt-template qwen_chat` 套 Qwen 对话模板，`--system-template colorist` 加上颜色助手 system prompt。

先跟合并后的模型对话（用上一步合并出的 1.8B merged/ 目录）：

```shell #test id="xtuner-chat-merged"
echo -e "Tell me about the color #66ccff\n\nEXIT\n" | \
python -m xtuner.tools.chat /tmp/xtuner_sft_llm_out_single/merged \
    --prompt-template qwen_chat \
    --system-template colorist \
    --no-streamer \
    --max-new-tokens 64 2>&1 | grep -E "^Load LLM from|Log: Exit!"
```

输出结果如下：

```shell #test-result id="xtuner-chat-merged"
Load LLM from /tmp/xtuner_sft_llm_out_single/merged
...Log: Exit!
```

不合并、只跟 LLM + LoRA adapter 直接对话（adapter 版）：

```shell #test id="xtuner-chat-adapter" load="xtuner_weights_path>>weights_dir"
hf_dir=$(ls -td /tmp/xtuner_sft_llm_out_single/iter_*_hf 2>/dev/null | head -1)
[ -n "$hf_dir" ] || { echo "no iter_*_hf: 先跑上面的模型转换"; exit 1; }
echo -e "Tell me about the color #66ccff\n\nEXIT\n" | \
python -m xtuner.tools.chat <weights_dir> \
    --adapter "$hf_dir" \
    --prompt-template qwen_chat \
    --system-template colorist \
    --no-streamer \
    --max-new-tokens 64 2>&1 | grep -E "^Load LLM from|^Load adapter from|Log: Exit!"
```

输出结果如下：

```shell #test-result id="xtuner-chat-adapter" load="xtuner_weights_path>>weights_dir" fuzzy='xxx' fuzzy='...'
Load LLM from <weights_dir>
Load adapter from /tmp/xtuner_sft_llm_out_single/iter_xxx_hf
...Log: Exit!
```

```{admonition} Note
:class: note
`<weights_dir>` 来自「准备模型权重」节保存的权重路径。
```

## 外部链接

- GitHub：[InternLM/xtuner](https://github.com/InternLM/xtuner)
- 文档中心：[XTuner Docs](https://xtuner.readthedocs.io/zh-cn/latest/)
