# PyG

PyG（PyTorch Geometric）是基于 PyTorch 的图神经网络库，提供图数据结构、消息传递层、图卷积算子和常用图数据集。本文在单卡昇腾 NPU 上安装 PyG，加载 Cora 引文图数据集，训练一个两层 GCN 完成论文节点分类。

## 前置条件

### 硬件

Atlas 900 A2 训练系列产品，单卡。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit 与驱动已安装，并能 `source set_env.sh`。版本按 [昇腾软件配套清单](https://www.hiascend.com/developer/download/compatibility) 选择 |
| Python | 落在官方配套表给出的范围内，并满足 PyG 下限；当前正式版要求 `>=3.10` |
| PyTorch | 安装官方当前推荐的 `torch` 与 `torch_npu`，见 [CANN 与 PyTorch 配套表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md) 和 [PyTorch 安装包](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download) |
| PyG | 从 PyPI 安装当前正式版 |

### 本文验证环境

- 机器类型：Atlas 900 A2（Ascend 910B4，32 GB × 1），Ubuntu 22.04
- 配套镜像：swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| torch | 2.12.0+cpu |
| torch_npu | 2.12.0.post2 |

---

## 1. 加载 CANN 环境

在终端加载 CANN 环境：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:/usr/local/bin:$PATH
```

---

## 2. 检查环境

确认 NPU 在线。`npu-smi` 找不到时参阅 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)。

```shell
npu-smi info
```

输出结果类似如下：

```text
+------------------------------------------------------------------------------------------------+
| npu-smi 25.5.1                   Version: 25.5.1                                               |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B4               | OK            | 89.7        39                0    / 0             |
| 0                         | 0000:C1:00.0  | 0           0    / 0          2888 / 32768         |
+===========================+===============+====================================================+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| No running processes found in NPU 0                                                            |
+===========================+===============+====================================================+
```

---

## 3. 安装 PyTorch NPU 栈

安装官方当前推荐的 `torch` 与 `torch_npu` 配套组合，并打印版本和 NPU 可用状态：

```shell #test id="install-torch"
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  torch_npu numpy pyyaml
python -c "import numpy, yaml, torch, torch_npu; print('torch', torch.__version__); print('torch_npu', torch_npu.__version__); print('npu_available', torch.npu.is_available())"
```

<!--
完整输出较长，其中应包含。`xxx` 表示实际安装的版本号，`torch` 为 CPU 构建版本，`+cpu` 表示 NPU 支持由 `torch_npu` 提供：

```shell #test-result id="install-torch" fuzzy="..." fuzzy="xxx"
...
torch xxx+cpu
torch_npu xxx
npu_available True
```
-->

---

## 4. 安装 PyG

从 PyPI 安装 PyG 当前正式版，并安装读取 Cora 数据集用到的 `scipy`：

<!--
```shell #test-setup store="pyg_ver"
echo "${UPSTREAM_REF#v}"
```
-->

```shell #test id="install-pyg" load="pyg_ver>>ver"
python -m pip install --no-build-isolation "torch_geometric==<ver>" scipy
python -c "import torch_geometric; print('torch_geometric', torch_geometric.__version__)"
```

<!--
完整输出较长，其中应包含：

```shell #test-result id="install-pyg" load="pyg_ver>>ver"
...torch_geometric <ver>
```
-->

```{admonition} Note
:class: note
将 `<ver>` 换成 PyG 最新的 release 版本号。
```

---

## 5. 训练 GNN 模型

下面脚本在 Cora 引文图上训练两层 GCN。数据和模型放到 `npu:0`，训练 200 轮，每 50 轮打印一次损失，训练结束后打印测试集准确率。

本示例的工作目录为 `/root/pyg-qs`，先创建该目录：

```shell
mkdir -p /root/pyg-qs
```

训练脚本如下，保存为 `/root/pyg-qs/train_gnn.py`：

```python
import torch
import torch_npu
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

torch.manual_seed(42)

device = torch.device('npu:0')
dataset = Planetoid(root='/root/pyg-qs/data', name='Cora')
data = dataset[0].to(device)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


model = GCN(dataset.num_features, 16, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print(f'device: {data.x.device}')

for epoch in range(1, 201):
    pred = model(data.x, data.edge_index)
    loss = F.cross_entropy(pred[data.train_mask], data.y[data.train_mask])

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch == 1 or epoch % 50 == 0:
        print(f'Epoch {epoch:03d}, Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    pred = model(data.x, data.edge_index).argmax(dim=-1)
test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
print(f'Test accuracy: {test_acc.item():.4f}')
```

<!--
```shell #test-setup
mkdir -p /root/pyg-qs
cat > /root/pyg-qs/train_gnn.py <<'PY'
import torch
import torch_npu
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

torch.manual_seed(42)

device = torch.device('npu:0')
dataset = Planetoid(root='/root/pyg-qs/data', name='Cora')
data = dataset[0].to(device)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


model = GCN(dataset.num_features, 16, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print(f'device: {data.x.device}')

for epoch in range(1, 201):
    pred = model(data.x, data.edge_index)
    loss = F.cross_entropy(pred[data.train_mask], data.y[data.train_mask])

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch == 1 or epoch % 50 == 0:
        print(f'Epoch {epoch:03d}, Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    pred = model(data.x, data.edge_index).argmax(dim=-1)
test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
print(f'Test accuracy: {test_acc.item():.4f}')
PY
```
-->

运行训练脚本：

```shell #test id="train-gnn"
python /root/pyg-qs/train_gnn.py
```

输出结果如下。损失和测试集准确率是固定随机种子下的实际数值：

```shell #test-result id="train-gnn"
device: npu:0
Epoch 001, Loss: 1.9444
Epoch 050, Loss: 0.0062
Epoch 100, Loss: 0.0021
Epoch 150, Loss: 0.0013
Epoch 200, Loss: 0.0008
Test accuracy: 0.7840
```

---

## 相关链接

更多图模型、数据集和教程见 PyG 官方文档，其余模块的用法与社区文档相同。

- 官方文档：[PyG Documentation](https://pytorch-geometric.readthedocs.io/en/latest/)
- GitHub：[pyg-team/pytorch_geometric](https://github.com/pyg-team/pytorch_geometric)
