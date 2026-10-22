# VBench

在单张昇腾 NPU 上运行 [VBench 官方 README 的 human_action 示例](https://github.com/Vchitect/VBench#usage)，
使用 UMT 动作识别模型评估视频是否符合给定的动作描述，并输出 JSON 结果。
本例使用官方 LaVie 的 `A person is strumming guitar`（弹吉他）描述及对应的
5 段视频，帮助你快速完成一次评测，不用于复现完整 VBench 排行榜分数。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品，至少一张可用 Ascend NPU，驱动与设备配置完好。
在容器中运行时，请先完成 NPU 设备及驱动的挂载，并确认 `npu-smi info` 正常。

### 基础软件

- Ubuntu 22.04 aarch64、Python 3.12、CANN toolkit 与驱动；
- CANN 环境已通过 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 加载；
- Git、C/C++ 编译器、CMake、FFmpeg 开发库，用于构建 CPU Decord；
- 能访问 Python 包源、GitHub、PyTorch CPU wheel 和官方 UMT 权重；
- 建议预留至少 10 GB 可写磁盘空间。

本指南使用 [workflows 的 VBench 项目](https://github.com/cosdt-ci-test/workflows/tree/abaa5c1610c57f34eeaa73208d91b6a9b1573a79/projects/vbench)
提供的 Ascend 补丁、安装脚本和示例视频。下文固定辅助仓库提交
`abaa5c1610c57f34eeaa73208d91b6a9b1573a79`，以便复现；VBench 本身仍按最新
release 安装。安装脚本会安装系统依赖，建议在具备 root 权限的 Ubuntu 容器中执行；
宿主机驱动需提前配置好。

### 本文档示例使用的版本

**配套机器**：Atlas 900 A2，单张 Ascend NPU，Ubuntu 22.04 aarch64。

**配套镜像**：

`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`

| 组件 | 版本 |
| --- | --- |
| Python | 3.12.13 |
| CANN | 9.1.0 |
| torch | 2.9.0+cpu |
| torch_npu | 2.9.0.post2 |
| torchvision | 0.24.0，CPU 构建，负责 CPU 预处理 |
| timm | 0.9.12 |
| numpy | 1.26.4 |
| Decord | 0.6.0，CPU 源码构建 |
| 评测器 | 官方 UMT `l16_ptk710_ftk710_ftk400_f16_res224.pth` |
| 样本 | 官方 LaVie `A person is strumming guitar`，原始索引 0～4 |

### 准备 Python 环境

在准备放置 `workflows` 的目录执行下列命令，获取配套工具、创建独立的 Python 环境并安装 `uv`。
使用配套依赖约束文件，使后续安装的库版本与本文使用的组合一致。

```shell
git clone https://github.com/cosdt-ci-test/workflows.git
git -C workflows fetch --depth 1 origin abaa5c1610c57f34eeaa73208d91b6a9b1573a79
git -C workflows checkout --detach FETCH_HEAD
cd workflows/projects/vbench
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip uv
export PIP_CONSTRAINT="$PWD/constraints-npu.txt"
export UV_CONSTRAINT="$PWD/constraints-npu.txt"
export ASCEND_RT_VISIBLE_DEVICES=0
```

`ASCEND_RT_VISIBLE_DEVICES=0` 选择一张物理 NPU；如需使用其他卡，请替换设备编号。
下文的 `npu:0` 表示该进程中第一张可见的 NPU。后续命令均在同一个已激活环境的
终端中执行，工作目录保持为 `projects/vbench/`。

## 检查并安装运行环境

```shell #test id="check-python"
python --version
```

输出结果如下：

```text #test-result id="check-python"
Python 3.12.13
```

先准备匹配的 PyTorch-NPU 栈；torchvision 从 PyTorch CPU 索引安装。它的图像预处理
在 CPU 上执行，UMT 前向计算在 NPU 上执行。

```shell #test-setup
set -euo pipefail
uv pip install -f https://mirrors.aliyun.com/pytorch-wheels/cpu \
    "torch==2.9.0" "torch_npu==2.9.0.post2" "numpy==1.26.4" pyyaml Pillow setuptools wheel
PIP_EXTRA_INDEX_URL= python -m pip install --no-deps \
    --index-url https://download.pytorch.org/whl/cpu "torchvision==0.24.0"
```

检查安装结果。以下代码用 Python 执行：

```python #test id="install-torch"
import torch
import torch_npu
import torchvision

assert torch.npu.is_available()
assert torch.npu.device_count() == 1
print("NPU available:", torch.npu.is_available())
print("torchvision:", torchvision.__version__)
```

输出结果如下：

```text #test-result id="install-torch"
...NPU available: True
torchvision: 0.24.0...
```

## 获取上游并应用 Ascend patch

获取 VBench 最新正式 release 的源码并应用配套 Ascend 补丁。

```{admonition} Note
:class: note
`<ref>` 替换为 VBench 当前最新 release 的标签，可从
[Releases](https://github.com/Vchitect/VBench/releases) 获取。
```

<!--
```shell #test-setup store="upstream_ref"
printf '%s\n' "${UPSTREAM_REF:?Set UPSTREAM_REF to the upstream ref}"
```
-->

```shell #test id="checkout-and-patch" load="upstream_ref>>ref"
set -euo pipefail
git init vbench-src
git -C vbench-src remote add origin https://github.com/Vchitect/VBench.git
git -C vbench-src fetch --depth 1 origin <ref>
git -C vbench-src checkout --detach FETCH_HEAD
python scripts/apply_patches.py vbench-src
git -C vbench-src diff --stat
echo "VBench Ascend patches applied"
```

输出结果如下：

```text #test-result id="checkout-and-patch"
...VBench Ascend patches applied
```

补丁为 CLI 增加 `npu:0` 设备选择和单卡启动支持，并提供本例所需的依赖集合；
不改变官方 UMT 模型、视频预处理及评分算法。应用补丁失败时，请先确认所选
VBench 版本与配套补丁兼容。

## 安装 VBench

Decord 0.6.0 没有官方 Linux aarch64 wheel，使用上游源码构建 CPU 解码器。
下面仅安装 `human_action` 评测所需依赖，不安装其他评测维度的全部依赖。

```shell #test-setup
set -euo pipefail
bash scripts/install_decord.sh
VBENCH_INSTALL_PROFILE=human_action uv pip install --no-build-isolation \
    -e ./vbench-src "gdown==6.4.2"
vbench evaluate --help >/dev/null
```

检查安装结果。以下代码用 Python 执行：

```python #test id="install-vbench"
import decord
import vbench.human_action

print("Decord:", decord.__version__)
print("VBench human_action import: OK")
```

输出结果如下：

```text #test-result id="install-vbench"
...Decord: 0.6.0
VBench human_action import: OK
```

## 准备官方权重和视频

运行配套资源准备脚本，获取官方 UMT 权重、准备 5 段示例视频，并从上游
`VBench_full_info.json` 生成本例的评测清单。
首次运行需要下载约 600 MB 权重，请确保能够访问
[官方权重仓库](https://huggingface.co/OpenGVLab/VBench_Used_Models)。

```shell #test id="prepare-assets"
set -euo pipefail
python -u scripts/prepare_assets.py vbench-src
```

输出结果如下：

```text #test-result id="prepare-assets"
...Original LaVie human_action videos: 5
```

准备完成后，视频位于 `vbench-src/sampled_videos/lavie/human_action/`，
文件名为 `A person is strumming guitar-0.mp4` 至 `A person is strumming guitar-4.mp4`；
评测清单为 `vbench-src/human_action_full_info.json`。这两个路径分别对应下文的
`--videos_path` 和 `--full_json_dir` 参数。

## 使用官方 CLI 进行 human_action 评测

通过 `--device npu:0` 选择 NPU，使用准备好的视频和评测清单运行：

```shell #test id="human-action-cli"
set -euo pipefail
vbench evaluate \
    --videos_path ./vbench-src/sampled_videos/lavie/human_action \
    --dimension human_action \
    --device npu:0 \
    --full_json_dir ./vbench-src/human_action_full_info.json \
    --output_path ./output/vbench/cli
```

输出结果如下：

```text #test-result id="human-action-cli"
...UMT device: npu:0
...done
```

## 使用官方 Python API

也可以直接调用官方 `VBench` API。两个入口使用相同的权重、视频和清单，
不必先执行 CLI 才能使用 API。将以下代码保存为 `evaluate_npu.py`，
在 `projects/vbench/` 目录运行 `python evaluate_npu.py`：

以下代码用 Python 执行：

```python #test id="human-action-api"
import torch
import torch_npu
from vbench import VBench

my_VBench = VBench(
    torch.device("npu:0"),
    "vbench-src/human_action_full_info.json",
    "output/vbench/api",
)
my_VBench.evaluate(
    videos_path="vbench-src/sampled_videos/lavie/human_action",
    name="lavie_human_action",
    dimension_list=["human_action"],
)
print("API results saved to output/vbench/api")
```

输出结果如下：

```text #test-result id="human-action-api"
...API results saved to output/vbench/api
```

## 查看结果

CLI 结果保存在 `output/vbench/cli/` 下的 `*_eval_results.json` 文件中；
上述 API 示例的结果文件为 `output/vbench/api/lavie_human_action_eval_results.json`。
其中 `human_action` 包含整体得分和逐视频结果，用于描述视频中是否识别出了指定动作。
得分可以为零，成功生成结果文件不等于动作识别正确或生成质量良好。

本文仅介绍初代 VBench 的单卡 `human_action` 评测。其他评测维度、多卡、
VBench++ 与 VBench-2.0 不在本指南范围内，不应直接照搬本例的依赖和设备参数。
