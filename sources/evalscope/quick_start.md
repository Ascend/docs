# evalscope

[EvalScope](https://github.com/modelscope/evalscope) 是 ModelScope 团队推出的轻量级大模型评测框架，统一封装了 LLM / VLM / RAG / AIGC 等多类评测后端，并提供可视化看板。本文用其**原生（Native）后端**（基于 transformers + accelerate），在单卡昇腾 NPU 上安装 EvalScope，对 [Qwen/Qwen2.5-0.5B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct) 在 `gsm8k`（数学，0-shot）与 `arc`（AI2 推理，含 ARC-Easy / ARC-Challenge）两个官方数据集上各跑 `--limit 5` 的小规模评测，并打印评测报告。

阅读本文前，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备好 CANN 与驱动。

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**）。本文示例为**单卡**。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit + 驱动固件已安装并可 `source set_env.sh` |
| Python | 3.12 |
| PyTorch | `torch==2.9.0` 与 `torch_npu==2.9.0.post2`，见下文安装 |
| EvalScope | 从 PyPI 安装（撰写时 `1.11.0`），见下文 |
| 模型 / 数据 | ModelScope 上的 Qwen2.5-0.5B-Instruct 与评测数据集 |

本文验证环境：Atlas 900 A2（Ascend 910B4，64 GB × 1）、Python 3.12、镜像 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`。

## 1. 加载 CANN 环境

新开终端后 CANN 变量不会自动生效。`npu-smi` 常见位于 `/usr/local/sbin`，需要把该目录加入 `PATH`。

```shell #test-setup
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
export PYTHONNOUSERSITE=1
test -n "$ASCEND_HOME_PATH"
command -v npu-smi
```

`ASCEND_HOME_PATH` 应非空，`npu-smi` 应能找到。

## 2. 检查环境是否就绪

检查 Python 版本：

```shell #test id="check-py"
python --version
```

输出结果如下：

```shell #test-result id="check-py" fuzzy='xxx'
Python 3.12.xxx
```

检查 torch / torch_npu 是否装好且 NPU 设备可用（下面的命令用 Python 执行）：

```python #test id="check-torch"
import torch, torch_npu
print('torch=', torch.__version__)
print('torch_npu=', torch_npu.__version__)
print('is_available:', torch.npu.is_available())
print('count:', torch.npu.device_count())
```

输出结果如下：

```shell #test-result id="check-torch" fuzzy='xxx'
torch=xxx
torch_npu=xxx
is_available: True
count: 1
```

```{admonition} Note
:class: note
如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。
```

## 3. 安装 EvalScope

EvalScope 的 Native 后端基于 transformers，因此评测机需已具备上一节的 `torch` / `torch_npu` 栈。直接用 `pip` 安装 EvalScope 本体（安装时会顺带拉取 transformers / modelscope 等导入期依赖）：

安装时注意两点：

- **显式安装 `accelerate`**：EvalScope 的本地模型封装在加载时一定会把 `device_map=get_device()`（NPU 上即 `'npu'`）传给 `from_pretrained`，而 transformers ≥5 只要收到 `device_map` 就要求 `accelerate`，但 `evalscope` 的依赖清单里并没有声明它——漏装会在加载模型时报 `requires accelerate`。
- **不要给 `numpy` 加上限（尤其别钉 `<2`）**：EvalScope 经 `modelscope[datasets]` 拉入的新版 `datasets` / `pyarrow` 在导入时**要求 NumPy ≥ 2**。若把 `numpy` 钉在 1.x，`import datasets` 会直接抛 `ImportError: pyarrow requires NumPy 2.0 or newer`（该检查发生在运行期、不在包元数据里，pip 解析时不会报冲突）。这里显式钉 `numpy>=2,<3`，与上游一致地留在 NumPy 2。

```shell #test-setup
python -m pip install "evalscope==1.11.0" "numpy>=2.0,<3" accelerate
```

装完打印版本并确认 NPU 栈可用（下面的命令用 Python 执行）：

```python #test id="check-evalscope"
import evalscope, transformers, accelerate, numpy, torch, torch_npu
print('evalscope', evalscope.__version__)
print('transformers', transformers.__version__)
print('accelerate', accelerate.__version__)
print('numpy', numpy.__version__)
print('torch', torch.__version__)
print('torch_npu', torch_npu.__version__)
print('npu_available', torch.npu.is_available())
```

输出结果如下：

```shell #test-result id="check-evalscope" fuzzy='xxx'
evalscope 1.11.0
transformers xxx
accelerate xxx
numpy 2.xxx
torch 2.9.0xxx
torch_npu 2.9.0.post2
npu_available True
```

`npu_available` 必须是 `True`。`False` 时不要继续，先查 CANN、驱动和可见设备。若 `torch` 不再以 `2.9.0` 开头，说明后面的包把 NPU 栈换掉了，回到上一节重装 `torch` / `torch_npu` 后再装 EvalScope。

## 4. 准备模型与数据

下面用 Python 从 ModelScope 拉取权重（第一次约 1 GB，之后走本地缓存，默认位于 `~/.cache/modelscope`）：

```python #test-setup store="model_path"
from modelscope import snapshot_download
print(snapshot_download('Qwen/Qwen2.5-0.5B-Instruct'))
```

确认权重目录完整：

```shell #test id="check-model" load="model_path>>model_path"
test -f "<model_path>/config.json" && echo has_config True
```

输出结果如下：

```shell #test-result id="check-model"
has_config True
```

`gsm8k` 与 `arc` 数据集在评测时由 EvalScope 自动下载到本地缓存（`--limit 5` 只取少量样本，下载很快）。

## 5. 在 NPU 上评测

### 5.1 先确认模型能够加载到 NPU

先用 transformers 以 `device_map="npu"` 加载模型并打印第一个参数的设备，确认模型确实落在 NPU 上（下面的命令用 Python 执行）。EvalScope 的 Native 后端同样把 `device_map` 直接传给 transformers 的 `from_pretrained`（NPU 上取值为 `'npu'`），因此这一步与 EvalScope 的加载路径一致。

```python #test id="npu-device-check" load="model_path>>model_path"
import torch, torch_npu
from transformers import AutoModelForCausalLM

m = AutoModelForCausalLM.from_pretrained('<model_path>', torch_dtype='auto', device_map='npu')
dev = str(next(m.parameters()).device)
print('evalscope_npu_device', dev)
assert dev.startswith('npu'), dev
```

输出结果如下：

```shell #test-result id="npu-device-check" fuzzy='...'
...evalscope_npu_device npu:0
```

`evalscope_npu_device` 必须是 `npu:0`。若打印 `cpu`/`cuda`，说明 `torch_npu` 没装上或可见设备为空，回到第 2–3 节排查。

### 5.2 运行 EvalScope 评测

`evalscope eval` 在指定数据集上用默认配置评测模型。`--model` 指向本地权重，`--model-args` 传入 `device_map="npu"`（正是 EvalScope 在 NPU 上的默认取值，显式写出更直观）把模型放到 NPU，`--work-dir` 指定输出目录，`--no-timestamp` 让报告直接落在该目录（默认会在 `--work-dir` 下再追加时间戳子目录），`--limit 5` 只跑 5 条样本用于快速验证流程（不代表真实榜单值）。如需限定设备，可用环境变量 `ASCEND_RT_VISIBLE_DEVICES`（例如 `0` 表示仅用第 0 张卡）。

```shell #test id="run-eval" load="model_path>>model_path"
evalscope eval \
  --model <model_path> \
  --datasets gsm8k arc \
  --limit 5 \
  --no-timestamp \
  --model-args '{"device_map": "npu"}' \
  --work-dir outputs/evalscope_npu_qs 2>&1
```

输出结果如下（评测日志较长，此处仅展示关键片段）：

```shell #test-result id="run-eval" fuzzy='...'
...Qwen2.5-0.5B-Instruct...
```

## 6. 查看评测结果

EvalScope 会把每个数据集的评测报告写到 `--work-dir` 下的 `reports/<模型名>/` 目录（JSON）。下面的命令用 Python 执行，把报告逐份打印出来：

```python #test id="check-result"
import glob
import json

print('evalscope 评测报告：')
for path in sorted(glob.glob('outputs/evalscope_npu_qs/reports/**/*.json', recursive=True)):
    print('report:', path)
    print(json.dumps(json.load(open(path, encoding='utf-8')), ensure_ascii=False, indent=2))
```

输出结果如下（`...` 表示省略的部分）：

```shell #test-result id="check-result" fuzzy='...'
...
report: ...arc.json
...
report: ...gsm8k.json
...
```

## 相关链接

更多数据集、批量评测与可视化看板用法见上游文档。

- 上游仓库：[modelscope/evalscope](https://github.com/modelscope/evalscope)
- 官方文档：[EvalScope 文档](https://evalscope.readthedocs.io/zh-cn/latest/)
- 基本用法：[Get Started](https://evalscope.readthedocs.io/zh-cn/latest/get_started/basic_usage.html)
- 本文所用模型：[Qwen2.5-0.5B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct)
- GSM8K 官方数据：[openai/grade-school-math](https://github.com/openai/grade-school-math)
- ARC 数据：[allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc)
