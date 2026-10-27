# DGL (Huawei Ascend NPU)

DGL 提供图数据结构、消息传递和图神经网络模块。DGL-Ascend 为这些接口增加昇腾 NPU 后端，让图结构和模型参数在 NPU 上参与计算。本文完成源码安装，并用 LightGCN 训练一个图推荐模型。

## 前置条件

### 硬件

Atlas 900 A2 训练系列产品（Ascend 910B4），并按需完成物理机或容器内的设备挂载。

### 基础软件

在运行本文档示例之前，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）

本文档示例使用 Python 3.10、CANN 9.1.0。

本文档配套镜像：`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.10`

## 加载 CANN 环境

在当前终端加载 CANN 的库路径和运行时配置，后续编译和 Python 命令才能使用 NPU：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 安装 Ascend NPU 所需依赖

`torch_npu` 让 PyTorch 使用昇腾 NPU。`torch` 与 `torch_npu` 版本须严格配套；按 [CANN 与 PyTorch 配套表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md) 选择与 CANN 匹配的组合：

```shell #test id="install-ascend-deps"
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch_npu==2.9.0.post2 --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi

python -c "import torch, torch_npu; print('torch version:', torch.__version__); print('torch_npu version:', torch_npu.__version__)"
```

输出结果如下：

```shell #test-result id="install-ascend-deps" fuzzy='...'
...
torch version: 2.9.0+cpu
torch_npu version: 2.9.0.post2
```

## 安装 DGL

**获取源码。**

```shell #test-setup
git clone https://github.com/BUPT-GAMMA/dgl-ascend.git
```

**获取子模块。** 下载编译所需的第三方源码：

```shell #test-setup
git -C dgl-ascend submodule update --init --recursive
```

**编译昇腾后端。** 构建支持昇腾 NPU 的 DGL 原生库：

```shell #test-setup
bash dgl-ascend/script/build_dgl_ascend.sh
```

**安装 Python 绑定。** 安装兼容上游代码的 Cython 版本和 Python 运行依赖，再将编译好的 DGL 原生库接入 Python 的 `dgl` 接口并构建扩展：

```shell #test-setup
pip install "Cython==3.0.12" packaging pandas "pydantic>=2.0" tqdm

cd dgl-ascend/python
python setup.py install
python setup.py build_ext --inplace
cd ../..
```

安装完成后，在当前目录导入 DGL 并查看版本：

```shell #test id="install-dgl"
python -c "import torch, torch_npu, dgl; print('dgl version:', dgl.__version__)"
```

输出结果如下：

```shell #test-result id="install-dgl" fuzzy='xxx'
dgl version: xxx
```

```{admonition} Note
:class: note
`xxx` 表示实际安装的 DGL 版本号。
```

## 用法示例

### LightGCN 图推荐训练

LightGCN 将用户和物品交互表示为二部图，通过多层邻居聚合学习推荐所需的嵌入，适合协同过滤任务。下面用 Gowalla 数据集在单卡 NPU 上训练一轮。

下载并解压数据，使用 Python 执行以下代码：

```python #test-setup
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

example_dir = Path('dgl-ascend/examples/pytorch/lightgcn')
archive = example_dir / 'gowalla.zip'
urlretrieve('https://s3.us-west-2.amazonaws.com/dgl-data/dataset/gowalla.zip', archive)
with ZipFile(archive) as dataset:
    dataset.extractall(example_dir)
```

运行训练，查看设备和每轮的平均 BPR 损失。BPR 损失用于学习用户对已交互物品与未交互物品的相对偏好：

```shell #test id="lightgcn-train"
cd dgl-ascend/examples/pytorch/lightgcn
python main.py --dataset gowalla --batch 2048 --recdim 64 --epochs 1 --device npu
```

输出结果如下：

```shell #test-result id="lightgcn-train" fuzzy='...' fuzzy='xxx'
...
device: npu
...
Average BPR Loss: xxx
```

```{admonition} Note
:class: note
`xxx` 表示本轮训练的平均 BPR 损失，以实际运行结果为准。
```

## 外部链接

- [DGL 官方仓库](https://github.com/dmlc/dgl)
- [DGL 官方文档](https://www.dgl.ai/dgl_docs/)
- [DGL-Ascend 仓库](https://github.com/BUPT-GAMMA/dgl-ascend)
- [DGL-Ascend 安装文档](https://github.com/BUPT-GAMMA/dgl-ascend/blob/master/AscendInstallation.md)
