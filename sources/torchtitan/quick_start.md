# torchtitan

[TorchTitan](https://github.com/pytorch/torchtitan) 是 PyTorch 原生的 LLM 预训练平台，通过 `torch_npu` 把 FSDP / TP / PP / CP 等并行策略的计算调度到昇腾 NPU。

本文在昇腾 NPU 上跑通 TorchTitan 的安装与训练链路：

- **安装**：用 uv 装 PyPI 上的二进制 wheel，或从上游最新 release tag 源码安装。
- **NPU 兼容 override**：用 torchtitan 官方 override 机制替换 attention 栈（RoPE + attention），训练前先断言替换组件与上游组件数值等价。
- **单卡训练**：`llama3_debugmodel` 配置真跑 2 步，验证配置解析、dataloader 构造、forward + backward 整条链路。
- **多卡训练**：双卡 FSDP shard 训练，验证 HCCL 集合通信与 DTensor 参数分布。

多卡训练需要至少 2 张 NPU。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，并按需完成物理机或容器内的设备挂载（`/dev/davinci*` 等）。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境（3.11 及以上，TorchTitan v0.3.0 要求 `requires-python >= 3.11`）
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）

```{admonition} Note
:class: note
CANN 最低版本为 8.0.rc1，安装 CANN 时请同时安装 Kernel 算子包。
```

### 本文档示例使用的版本

**配套机器**：

- **机器类型**：Atlas 900 A2 PODc（Ascend 910B4，32 GB × 2 — CANN / 驱动预留约 2.5 GB，每张卡可用约 30 GB）
- **操作系统**：Ubuntu 22.04

**配套镜像**：

swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.2.0-910b-ubuntu22.04-py3.12

**软件版本**：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.2.0 |
| torch | 2.14.0+cpu |
| torch_npu | 2.14.0 |
| triton-ascend | 3.6.0 |
| modelscope | 1.40.2 |
| torchtitan | 最新 release |
| 训练配置 | 单卡 / 多卡均使用上游默认配置 `torchtitan/models/llama3/config_registry.py::llama3_debugmodel`（默认 `ChunkedLossWrapper` loss；基于 debugmodel：dim=256 / 6 层 / 16 head / vocab 2048，~6 M 参数；多卡为 FSDP shard=2 + bf16） |

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
+========================---+---------------+----------------------------------------------------+
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
uv pip install -f https://mirrors.aliyun.com/pytorch-wheels/cpu torch==2.14.0
uv pip install torch_npu==2.14.0
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
torch= 2.14.0+cpu
torch_npu= 2.14.0
is_available: True
count: 2
```

```{admonition} Note
:class: note
如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。
```

### 安装 modelscope

```shell #test-setup
uv pip install modelscope==1.40.2
```

打印安装版本(下面的命令用 Python 执行)：

```python #test id="modelscope-install"
import modelscope
print('modelscope', modelscope.__version__)
```

输出结果如下：

```shell #test-result id="modelscope-install"
modelscope 1.40.2
```

### 安装 triton-ascend

```shell #test-setup
uv pip install --no-deps --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi/test triton-ascend==3.6.0
uv pip install pybind11
```

打印安装版本(下面的命令用 Python 执行)：

```python #test id="triton-install"
from importlib.metadata import version
print('triton-ascend', version('triton-ascend'))
```

输出结果如下：

```shell #test-result id="triton-install"
triton-ascend 3.6.0
```

## 安装 torchtitan

### 通过 uv 安装

torchtitan 上 PyPI 有 `py3-none-any` 的纯 Python wheel（450 KB），不依赖 native build——`uv pip install torchtitan` 直接装最新 stable release，不用管版本号：

```shell #test-setup
uv pip install --extra-index-url https://mirrors.aliyun.com/pypi/simple/ torchtitan
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

<!-- 二进制 wheel 装完后先卸掉，后面的源码安装（可编辑模式）才能干净地接管 torchtitan 包-->
<!--
```shell #test-setup
uv pip uninstall torchtitan -y
```
-->

### 从源码安装

<!-- 工作流注入的 UPSTREAM_REF（最新 release tag）通过这个隐藏的 #test-setup 捕获并注入到下方 install 命令中-->
<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

克隆上游仓库、`uv pip install -e .` 安装：

```shell #test-setup load="upstream_ref>>ref"
git clone --depth 1 --branch <ref> https://github.com/pytorch/torchtitan.git
cd torchtitan
uv pip install -e .
```

```{admonition} Note
:class: note
`<ref>` 替换为 torchtitan 当前的最新 release 版本（如 `v0.3.0`）。
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
```

## 跑训练

### 下载 tokenizer

```shell #test-setup id="modelscope-download-tokenizer" store="ms_tokenizer_path"
python -c "from modelscope import snapshot_download; print(snapshot_download('LLM-Research/Llama-3.2-1B', allow_patterns=['*.json', '*.model', 'tokenizer*']))" | tail -n 1
```

验证 tokenizer 关键文件都落盘：

```shell #test id="modelscope-verify-tokenizer" load="ms_tokenizer_path>>ms_tokenizer_path"
ls <ms_tokenizer_path>
```

```shell #test-result id="modelscope-verify-tokenizer" fuzzy='...'
config.json
configuration.json
generation_config.json
original
special_tokens_map.json
tokenizer.json
tokenizer_config.json
```

文件名是 Llama 3 tokenizer 必备文件，确认 snapshot_download 命中正确。

```{admonition} Note
:class: note
`<ms_tokenizer_path>` 替换为上一步 `snapshot_download` 打印的模型落盘路径。
```

### 安装 NPU override 模块

override 模块是本仓（Ascend/docs）里的两个普通 Python 文件，放在 torchtitan 仓库**之外**，通过 `PYTHONPATH` + `--override.imports module.function` 激活：

| 文件 | 作用 |
| --- | --- |
| `tests/torchtitan/npu_compat.py` | override 模块本身：注册一个 `npu_attention` 工厂，替换 `GQAttention.Config` 节点 |
| `tests/torchtitan/check_npu_overrides.py` | 等价性自检脚本（CPU）：断言替换进去的组件与上游组件数值等价 |

```shell #test-setup
mkdir -p "$HOME/tt-npu-overrides"
BASE="https://api.github.com/repos/Ascend/docs/contents/tests/torchtitan"
REF="${DOC_REF:-main}"
AUTH=()
[ -n "${GH_TOKEN:-}" ] && AUTH+=(-H "Authorization: Bearer $GH_TOKEN")
for f in npu_compat.py check_npu_overrides.py; do
  curl -fsSL --retry 3 --retry-connrefused --retry-delay 2 \
    -H "Accept: application/vnd.github.raw" "${AUTH[@]}" \
    "$BASE/$f?ref=$REF" -o "$HOME/tt-npu-overrides/$f"
done
```

验证 override 工厂完成注册：

```shell #test id="override-register"
cd torchtitan && PYTHONPATH="$HOME/tt-npu-overrides" python -c "import npu_compat; from torchtitan.config.override import _REGISTRY; print(sorted(_REGISTRY))"
```

输出结果如下：

```shell #test-result id="override-register"
['npu_compat.npu_attention']
```

再断言替换进去的组件与上游组件**数值等价**（CPU 即可跑，不需要 NPU）：RoPE 与上游 `ComplexRoPE` 的输出必须逐位一致，attention 与朴素实现的逐文档 causal 掩码注意力必须一致。

```shell #test id="override-equivalence"
python "$HOME/tt-npu-overrides/check_npu_overrides.py"
```

输出结果如下：

```shell #test-result id="override-equivalence"
npu override equivalence: rope OK, mask OK
```

### 单卡训练

在 1 张 NPU 上用 `torchrun` 真跑 2 步 `debugmodel`（上游自带的最小配置，~6 M 参数，单卡足够），验证完整的训练链路。`--override.imports` 激活上一步的 NPU 兼容替换，日志会打印 `Applied 6 override(s)`。多卡训练见下一节「多卡训练」（8B 规模不在本文档验证范围）：

```shell #test id="torchtitan-train-debug" load="upstream_ref>>ref"
cd torchtitan && git checkout -f <ref>
PYTHONPATH="$HOME/tt-npu-overrides" \
ASCEND_RT_VISIBLE_DEVICES=0 \
TORCHINDUCTOR_COMPILE_THREADS=1 \
torchrun --nproc_per_node=1 \
    --rdzv_backend c10d \
    --rdzv_endpoint="localhost:0" \
    -m torchtitan.train \
    --module llama3 \
    --config llama3_debugmodel \
    --override.imports npu_compat.npu_attention \
    --comm.mode default \
    --training.steps 2 \
    --training.local-batch-size 1 \
    --training.seq-len 256 \
    --metrics.log-freq 1 \
    --metrics.disable-color-printing \
    --dump-folder /tmp/torchtitan-quickstart
```

输出结果类似如下：

```shell #test-result id="torchtitan-train-debug" fuzzy='xxx' fuzzy='...'
[titan] xxx - root - INFO - torchtitan version: xxx
[titan] xxx - root - INFO - [Override] npu_compat.npu_attention: model_spec.model.layers.0.attention GQAttention.Config -> NpuGQAttention.Config
...
[titan] xxx - root - INFO - Applied 6 override(s)
[titan] xxx - root - INFO - Building llama3 debugmodel
...
[titan] xxx - root - INFO - Sleeping 2 seconds for other ranks to complete
[titan] xxx - root - INFO - Training completed
[titan] xxx - root - INFO - Process group destroyed
```

### 多卡训练

在 2 张 NPU 上用 `torchrun` 起真分布式训练，同样真跑 2 步：FSDP 把模型参数和优化器状态分片到两张卡，并叠 `--training.dtype bfloat16` 验证混合精度。`HCCL_NPU_SOCKET_PORT_RANGE` 用于错开两个 rank 的 HCCL 端口（否则会报 `EI0020` 端口冲突）：

```shell #test id="torchtitan-train-2card" load="upstream_ref>>ref"
cd torchtitan && git checkout -f <ref>
PYTHONPATH="$HOME/tt-npu-overrides" \
ASCEND_RT_VISIBLE_DEVICES=0,1 \
HCCL_NPU_SOCKET_PORT_RANGE=60000-60200 \
TORCHINDUCTOR_COMPILE_THREADS=1 \
torchrun --nproc_per_node=2 \
    --rdzv_backend c10d \
    --rdzv_endpoint="localhost:0" \
    --local-ranks-filter 0 \
    --tee 3 \
    -m torchtitan.train \
    --module llama3 \
    --config llama3_debugmodel \
    --override.imports npu_compat.npu_attention \
    --comm.mode default \
    --training.dtype bfloat16 \
    --training.steps 2 \
    --training.local-batch-size 1 \
    --training.seq-len 256 \
    --metrics.log-freq 1 \
    --metrics.disable-color-printing \
    --dump-folder /tmp/torchtitan-quickstart-2card
```

输出结果类似如下：

```shell #test-result id="torchtitan-train-2card"
[default0]:[titan] ... - root - INFO - torchtitan version: ...
[default0]:[titan] ... - root - INFO - Building device mesh with parallelism: ...
[default0]:[titan] ... - root - INFO - Applied 6 override(s)
[default0]:[titan] ... - root - INFO - Building llama3 debugmodel
...
[default0]:[titan] ... - root - INFO - Synchronizing and adjusting timeout for all ProcessGroups to 0:01:40
...
[default0]:[titan] ... - root - INFO - Sleeping 2 seconds for other ranks to complete
[default0]:[titan] ... - root - INFO - Training completed
[default0]:[titan] ... - root - INFO - Process group destroyed
```

## 外部链接

- GitHub：[pytorch/torchtitan](https://github.com/pytorch/torchtitan)
- 论文：[PyTorch Native LLM Training](https://arxiv.org/abs/2410.06511)
