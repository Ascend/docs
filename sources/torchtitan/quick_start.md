# torchtitan

[TorchTitan](https://github.com/pytorch/torchtitan) 是 PyTorch 原生的 LLM 预训练平台，通过 `torch_npu` 把 FSDP / TP / PP / CP 等并行策略的计算调度到昇腾 NPU。

本文在昇腾 NPU 上跑通 TorchTitan 的安装与训练链路：

- **NPU 兼容 override**：用 torchtitan 官方 override 机制替换 attention 后端（上游的 flex / varlen 后端在 NPU 上跑不起来），训练前先断言替换组件与上游组件数值等价。
- **单卡训练**：本文自带的 recipe（llama3 `debugmodel`，~6 M 参数）真跑 2 步，验证配置解析、dataloader 构造、forward + backward 整条链路。
- **多卡训练**：双卡 FSDP shard + bfloat16 训练，验证 HCCL 集合通信与 DTensor 参数分布。

多卡训练需要至少 2 张 NPU。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，并按需完成物理机或容器内的设备挂载（`/dev/davinci*` 等）。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境（3.11 及以上，TorchTitan 要求 `requires-python >= 3.11`）
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）

```{admonition} Note
:class: note
CANN 最低版本为 8.0.rc1，安装 CANN 时请同时安装 Kernel 算子包。
```

### 本文档示例使用的版本

**配套机器**：

- **机器类型**：Atlas 900 A2 PODc（Ascend 910B4，32 GB × 2 — CANN / 驱动预留约 2.5 GB，每张卡可用约 30 GB）/ Atlas 900 A3（Ascend 910C）
- **操作系统**：Ubuntu 22.04

**配套镜像**：

- A2（Ascend 910B）：`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.2.0-910b-ubuntu22.04-py3.12`
- A3（Ascend 910C）：`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.2.0-a3-ubuntu22.04-py3.12`

**软件版本**：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.2.0 |
| torch | 2.15.0+cpu |
| torch_npu | 2.15.0.dev20261116 |
| triton-ascend | 3.6.0+dev20261112 |
| torchtitan | 最新 stable release（二进制 wheel 与源码安装用同一个 tag） |
| 训练配置 | 本文自带的 recipe `npu_recipe.py::llama3_npu_debugmodel`（单卡 fp32）/ `llama3_npu_debugmodel_bf16`（多卡 bf16）：模型是 llama3 `debugmodel`（dim=256 / 6 层 / 16 head / vocab 2048，~6 M 参数），数据是上游自带的 `tests/assets/c4_test`，tokenizer 是上游自带的 `tests/assets/tokenizer`，不下载任何权重 |

## 前置安装

确认能看到 NPU 设备：

```shell
npu-smi info
```

输出类似：

```text
+------------------------------------------------------------------------------------------------+
| npu-smi 25.5.1                   Version: 25.5.1                                               |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 5     910B4               | OK            | 89.9        39                0    / 0             |
| 0                         | 0000:41:00.0  | 0           0    / 0          2922 / 32768         |
+===========================+===============+====================================================+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+---------------------------+---------------+----------------------------------------------------+
| No running processes found in NPU 5                                                            |
+---------------------------+---------------+----------------------------------------------------+
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

装 `torch` + `torch_npu`：

```shell #test-setup
uv pip install -f https://mirrors.aliyun.com/pytorch-wheels/cpu torch==2.15.0
uv pip install --no-deps --find-links https://repo.huaweicloud.com/ascend/repos/pypi/nightly/torch-npu/ torch_npu==2.15.0.dev20261116
uv pip install numpy
```

```{admonition} Note
:class: note
`torch_npu` 装的是 Ascend nightly 源的版本（与 torch 2.15 配套）。
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
torch= 2.15.0+cpu
torch_npu= 2.15.0.dev20261116
is_available: True
count: 2
```

```{admonition} Note
:class: note
如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。
```

### 安装 triton-ascend

`triton-ascend` 是 Ascend 的 Triton fork，本文装与 `torch_npu` 配套的 nightly 版。

```shell #test-setup
set -euo pipefail
uv pip install --no-deps --find-links https://repo.huaweicloud.com/ascend/repos/pypi/nightly/triton-ascend/ triton-ascend==3.6.0+dev20261112
uv pip install pybind11
```

打印安装版本(下面的命令用 Python 执行)：

```python #test id="triton-install"
from importlib.metadata import version
print('triton-ascend', version('triton-ascend'))
```

输出结果如下：

```shell #test-result id="triton-install"
triton-ascend 3.6.0+dev20261112
```

## 安装 torchtitan

### 通过 uv 安装

`uv pip install torchtitan` 直接装最新 stable release：

```shell #test-setup
printf 'attn-gym\n' > /tmp/torchtitan-excludes.txt
uv pip install --excludes /tmp/torchtitan-excludes.txt --extra-index-url https://mirrors.aliyun.com/pypi/simple/ torchtitan
```

验证装上的版本(下面的命令用 Python 执行)：

```python #test id="torchtitan-install-binary"
import torchtitan
print('torchtitan', torchtitan.__version__)
```

输出结果类似如下：

```shell #test-result id="torchtitan-install-binary" fuzzy='xxx'
torchtitan xxx
```

```{admonition} Note
:class: note
xxx 表示最新的版本号。
```

### 从源码安装

```shell #test-setup store="upstream_ref"
set -o pipefail
if [ -n "${UPSTREAM_REF:-}" ]; then
  echo "${UPSTREAM_REF}"
else
  curl -fsSL --retry 3 --retry-connrefused --retry-delay 2 \
    https://api.github.com/repos/pytorch/torchtitan/releases/latest \
    | python -c "import json,sys; print(json.load(sys.stdin)['tag_name'])"
fi
```

克隆上游仓库、`uv pip install -e .` 安装：

```shell #test-setup load="upstream_ref>>ref"
set -euo pipefail
git clone --depth 1 --branch <ref> https://github.com/pytorch/torchtitan.git
cd torchtitan
uv pip install --excludes /tmp/torchtitan-excludes.txt -e .
```

验证装上的版本(下面的命令用 Python 执行，需在克隆出来的源码目录里跑)：

```shell #test id="torchtitan-install-source"
cd torchtitan && python -c "import torchtitan; print('torchtitan', torchtitan.__version__)"
```

输出结果类似如下：

```shell #test-result id="torchtitan-install-source" fuzzy='xxx'
torchtitan xxx
```

```{admonition} Note
:class: note
xxx 表示最新的版本号。
`<ref>`、`UPSTREAM_REF` 换成上游最新 release 的 tag。
```

## 跑训练

### 安装 NPU override 模块与 recipe

override 模块和 recipe 都是本仓（Ascend/docs）里的普通 Python 文件，放在 torchtitan 仓库**之外**，
通过 `PYTHONPATH` 挂进去：

| 文件 | 作用 |
| --- | --- |
| [`tests/torchtitan/npu_compat.py`](https://github.com/Ascend/docs/blob/main/tests/torchtitan/npu_compat.py) | override 模块本身：注册一个 `npu_attention` 工厂，替换 `GQAttention.Config` 节点 |
| [`tests/torchtitan/npu_recipe.py`](https://github.com/Ascend/docs/blob/main/tests/torchtitan/npu_recipe.py) | quick-start 的 recipe：返回 `Trainer.Config`（模型 / 数据 / 步数 / loss）——上游的配置改成 recipe 之后，`--training.steps` 这类通用开关没有了，训练得由 recipe 自己给配置 |
| [`tests/torchtitan/check_npu_overrides.py`](https://github.com/Ascend/docs/blob/main/tests/torchtitan/check_npu_overrides.py) | 等价性自检脚本（CPU）：断言替换进去的组件与上游组件数值等价 |

override 模块由 torchtitan 官方的 override 机制激活（`--override npu_compat.npu_attention`）。

```shell #test-setup
mkdir -p "$HOME/tt-npu-overrides"
BASE="https://api.github.com/repos/Ascend/docs/contents/tests/torchtitan"
REF="${DOC_REF:-main}"
AUTH=()
[ -n "${GH_TOKEN:-}" ] && AUTH+=(-H "Authorization: Bearer $GH_TOKEN")
for f in npu_compat.py npu_recipe.py check_npu_overrides.py; do
  curl -fsSL --retry 3 --retry-connrefused --retry-delay 2 \
    -H "Accept: application/vnd.github.raw" "${AUTH[@]}" \
    "$BASE/$f?ref=$REF" -o "$HOME/tt-npu-overrides/$f"
done
```

### 验证 override 与 recipe

先确认装好的 torch 提供了上游需要的三个 API：

```shell #test id="torch-compat"
cd torchtitan && PYTHONPATH="$HOME/tt-npu-overrides" python -c "import torch, torch.distributed.pipelining as p; from torch.distributed.fsdp import MixedPrecisionPolicy; print('analyze_pipeline_activation_liveness:', hasattr(p, 'analyze_pipeline_activation_liveness')); print('PipelineStageInfo:', hasattr(p, 'PipelineStageInfo')); print('param_dtype_override_fn:', MixedPrecisionPolicy(param_dtype=torch.bfloat16, param_dtype_override_fn=None).param_dtype)"
```

输出结果如下：

```shell #test-result id="torch-compat"
analyze_pipeline_activation_liveness: True
PipelineStageInfo: True
param_dtype_override_fn: torch.bfloat16
```

验证 override 工厂完成注册：

```shell #test id="override-register"
cd torchtitan && PYTHONPATH="$HOME/tt-npu-overrides" python -c "import npu_compat; from torchtitan.config.override import _REGISTRY; print(sorted(_REGISTRY))"
```

输出结果如下：

```shell #test-result id="override-register"
['npu_compat.npu_attention']
```

再断言替换进去的组件与上游组件**数值等价**（CPU 即可跑，不需要 NPU）。

```shell #test id="override-equivalence"
PYTHONPATH="$HOME/tt-npu-overrides" python "$HOME/tt-npu-overrides/check_npu_overrides.py"
```

输出结果如下：

```shell #test-result id="override-equivalence"
npu override equivalence: mask OK, metadata OK, attention selection OK
```

验证 recipe 能被 `--module` 选中，并返回一份可训练的 `Trainer.Config`：

```shell #test id="recipe-import"
cd torchtitan && PYTHONPATH="$HOME/tt-npu-overrides" python -c "import npu_recipe; c = npu_recipe.llama3_npu_debugmodel(); print('recipe ok, steps =', c.training.steps)"
```

输出结果如下：

```shell #test-result id="recipe-import"
recipe ok, steps = 2
```

### 单卡训练

在 1 张 NPU 上用 `torchrun` 真跑 2 步 `debugmodel`，验证完整的训练链路。
`--override` 激活上一步的 NPU 兼容替换，日志会打印 `Applied 6 override(s)`（6 层各替换一次）。
多卡训练见下一节「多卡训练」：

```shell #test id="torchtitan-train-debug"
cd torchtitan && PYTHONPATH="$HOME/tt-npu-overrides" \
ASCEND_RT_VISIBLE_DEVICES=0 \
TORCHINDUCTOR_COMPILE_THREADS=1 \
torchrun --nproc_per_node=1 \
    --rdzv_backend c10d \
    --rdzv_endpoint="localhost:0" \
    -m torchtitan.train \
    --module npu_recipe \
    --config llama3_npu_debugmodel \
    --override npu_compat.npu_attention \
    --output-dir /tmp/torchtitan-quickstart
```

输出结果类似如下：

```shell #test-result id="torchtitan-train-debug" fuzzy='xxx' fuzzy='...'
[titan] ... - __main__ - INFO - torchtitan version: ...
[titan] ... - torchtitan.config.override - INFO - [Override] npu_compat.npu_attention: model.layers.0.attention GQAttention.Config -> NpuGQAttention.Config
[titan] ... - torchtitan.config.override - INFO - Applied 6 override(s)
[titan] ... - torchtitan.observability.metrics - INFO - step: ...
[titan] ... - torchtitan.trainer - INFO - Training completed
[titan] ... - __main__ - INFO - Process group destroyed
...
```

### 多卡训练

在 2 张 NPU 上用 `torchrun` 起真分布式训练，同样真跑 2 步：FSDP 把模型参数和优化器状态分片到两张卡，
并叠 bf16 参数（`llama3_npu_debugmodel_bf16`）验证混合精度。`HCCL_NPU_SOCKET_PORT_RANGE` 用于错开两个 rank 的
HCCL 端口（否则会报 `EI0020` 端口冲突）：

```shell #test id="torchtitan-train-2card"
cd torchtitan && PYTHONPATH="$HOME/tt-npu-overrides" \
ASCEND_RT_VISIBLE_DEVICES=0,1 \
HCCL_NPU_SOCKET_PORT_RANGE=60000-60200 \
TORCHINDUCTOR_COMPILE_THREADS=1 \
torchrun --nproc_per_node=2 \
    --rdzv_backend c10d \
    --rdzv_endpoint="localhost:0" \
    --local-ranks-filter 0 \
    --tee 3 \
    -m torchtitan.train \
    --module npu_recipe \
    --config llama3_npu_debugmodel_bf16 \
    --override npu_compat.npu_attention \
    --output-dir /tmp/torchtitan-quickstart-2card
```

输出结果类似如下：

```shell #test-result id="torchtitan-train-2card" fuzzy='xxx' fuzzy='...'
[default0]:[titan] ... - __main__ - INFO - torchtitan version: ...
[default0]:[titan] ... - torchtitan.config.override - INFO - Applied 6 override(s)
[default0]:[titan] ... - torchtitan.observability.metrics - INFO - step: ...
[default0]:[titan] ... - torchtitan.trainer - INFO - Training completed
[default0]:[titan] ... - __main__ - INFO - Process group destroyed
...
```

## 外部链接

- GitHub：[pytorch/torchtitan](https://github.com/pytorch/torchtitan)
- 论文：[PyTorch Native LLM Training](https://arxiv.org/abs/2410.06511)
