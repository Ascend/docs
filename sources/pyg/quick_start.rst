快速开始
==================

.. note::
    阅读本篇前，请确保已按照 :doc:`安装教程 <./install>` 准备好昇腾环境及 PyG 

本文档帮助昇腾开发者快速使用 PyG × 昇腾 进行 GNN 训练。你可以访问 `这篇官方论文 <https://arxiv.org/abs/2507.16991>`_ 获取更多信息。

概览
---------------------

PyG 包含了针对图 (Grpah) 及其他不规则结构的多种深度学习方法，这些方法来自众多已发表的论文。


训练示例
---------------------

示例训练实现了针对引用图中论文分类的图神经网络的训练。首先加载 Cora 数据集，并使用预定义的 GCNConv 创建了一个简单的两层 GCN 模型，然后开始训练。

GCN (Graph Convolutional Network) 是一种经典的图神经网络架构，适用于处理图结构数据。GCN 通过在图上进行卷积操作来捕捉节点之间的关系，从而实现节点分类、图分类等任务。

.. code-block:: shell
    :linenos:

    import torch
    from torch import Tensor
    from torch_geometric.nn import GCNConv
    from torch_geometric.datasets import Planetoid

    dataset = Planetoid(root='.', name='Cora')

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

    model = GCN(dataset.num_features, 16, dataset.num_classes)

该示例会自动下载 Cora 数据集，并使用 GCN 模型以备训练。你可以根据需要修改模型结构、训练参数等，以适应不同的任务和数据集。

也可以使用 PyG 提供的示例代码进行训练，例如使用官方提供的 GCN 示例进行训练：


.. code-block:: shell
    :linenos:

    python ./examples/gcn.py 



看到类似如下输出，loss 存在明显下降趋势说明训练成功：

.. code-block:: shell
    :linenos:

    Epoch: 001, Loss: 1.9458, Train: 0.2286, Val: 0.2580, Test: 0.2510
    Epoch: 002, Loss: 1.9432, Train: 0.3857, Val: 0.2300, Test: 0.2510
    Epoch: 003, Loss: 1.9374, Train: 0.7643, Val: 0.4500, Test: 0.4500
    Epoch: 004, Loss: 1.9290, Train: 0.7286, Val: 0.3860, Test: 0.4500
    Epoch: 005, Loss: 1.9223, Train: 0.7643, Val: 0.4080, Test: 0.4500
    Epoch: 006, Loss: 1.9178, Train: 0.7286, Val: 0.4140, Test: 0.4500
    Epoch: 007, Loss: 1.9078, Train: 0.7786, Val: 0.4380, Test: 0.4500
    Epoch: 008, Loss: 1.8984, Train: 0.7429, Val: 0.4320, Test: 0.4500
    Epoch: 009, Loss: 1.8855, Train: 0.7786, Val: 0.4040, Test: 0.4500
    Epoch: 010, Loss: 1.8776, Train: 0.7786, Val: 0.4220, Test: 0.4500
    Epoch: 011, Loss: 1.8649, Train: 0.7714, Val: 0.4360, Test: 0.4500
    Epoch: 012, Loss: 1.8561, Train: 0.8143, Val: 0.4900, Test: 0.4920
    Epoch: 013, Loss: 1.8508, Train: 0.8429, Val: 0.5400, Test: 0.5440
    Epoch: 014, Loss: 1.8339, Train: 0.8786, Val: 0.5680, Test: 0.5970
    Epoch: 015, Loss: 1.8220, Train: 0.9071, Val: 0.6140, Test: 0.6470
    Epoch: 016, Loss: 1.8028, Train: 0.9214, Val: 0.6540, Test: 0.6830
    Epoch: 017, Loss: 1.7940, Train: 0.9214, Val: 0.6800, Test: 0.7130
    Epoch: 018, Loss: 1.7733, Train: 0.9286, Val: 0.6900, Test: 0.7150
    Epoch: 019, Loss: 1.7586, Train: 0.9357, Val: 0.6920, Test: 0.7290
    Epoch: 020, Loss: 1.7426, Train: 0.9357, Val: 0.6980, Test: 0.7380
    Epoch: 021, Loss: 1.7214, Train: 0.9429, Val: 0.7040, Test: 0.7430
    Epoch: 022, Loss: 1.7060, Train: 0.9429, Val: 0.7080, Test: 0.7460
    Epoch: 023, Loss: 1.6939, Train: 0.9429, Val: 0.7200, Test: 0.7500
    Epoch: 024, Loss: 1.6736, Train: 0.9429, Val: 0.7160, Test: 0.7500
    Epoch: 025, Loss: 1.6517, Train: 0.9429, Val: 0.7180, Test: 0.7500
    Epoch: 026, Loss: 1.6458, Train: 0.9429, Val: 0.7220, Test: 0.7370
    Epoch: 027, Loss: 1.6297, Train: 0.9429, Val: 0.7240, Test: 0.7380
    Epoch: 028, Loss: 1.5822, Train: 0.9429, Val: 0.7140, Test: 0.7380
    Epoch: 029, Loss: 1.5706, Train: 0.9429, Val: 0.7120, Test: 0.7380
    Epoch: 030, Loss: 1.5858, Train: 0.9429, Val: 0.7220, Test: 0.7380
    Epoch: 031, Loss: 1.5373, Train: 0.9429, Val: 0.7300, Test: 0.7500
    Epoch: 032, Loss: 1.5358, Train: 0.9429, Val: 0.7260, Test: 0.7500
    Epoch: 033, Loss: 1.5177, Train: 0.9429, Val: 0.7300, Test: 0.7500
    Epoch: 034, Loss: 1.4543, Train: 0.9429, Val: 0.7420, Test: 0.7660
    Epoch: 035, Loss: 1.4536, Train: 0.9429, Val: 0.7520, Test: 0.7740
    Epoch: 036, Loss: 1.4642, Train: 0.9429, Val: 0.7560, Test: 0.7740
    Epoch: 037, Loss: 1.4009, Train: 0.9500, Val: 0.7620, Test: 0.7780
    Epoch: 038, Loss: 1.3986, Train: 0.9500, Val: 0.7560, Test: 0.7780
    Epoch: 039, Loss: 1.3620, Train: 0.9500, Val: 0.7520, Test: 0.7780
    Epoch: 040, Loss: 1.3841, Train: 0.9500, Val: 0.7580, Test: 0.7780
    Epoch: 041, Loss: 1.3488, Train: 0.9500, Val: 0.7700, Test: 0.7800
    Epoch: 042, Loss: 1.3262, Train: 0.9571, Val: 0.7680, Test: 0.7800
    Epoch: 043, Loss: 1.2861, Train: 0.9571, Val: 0.7760, Test: 0.7850
    Epoch: 044, Loss: 1.2833, Train: 0.9571, Val: 0.7800, Test: 0.7880
    Epoch: 045, Loss: 1.2255, Train: 0.9571, Val: 0.7660, Test: 0.7880
    Epoch: 046, Loss: 1.2127, Train: 0.9500, Val: 0.7620, Test: 0.7880
    Epoch: 047, Loss: 1.2455, Train: 0.9571, Val: 0.7660, Test: 0.7880
    Epoch: 048, Loss: 1.1698, Train: 0.9571, Val: 0.7660, Test: 0.7880
    Epoch: 049, Loss: 1.1380, Train: 0.9500, Val: 0.7680, Test: 0.7880
    Epoch: 050, Loss: 1.1567, Train: 0.9500, Val: 0.7680, Test: 0.7880
    Epoch: 051, Loss: 1.1356, Train: 0.9500, Val: 0.7680, Test: 0.7880
    Epoch: 052, Loss: 1.1302, Train: 0.9571, Val: 0.7680, Test: 0.7880
    Epoch: 053, Loss: 1.0982, Train: 0.9571, Val: 0.7640, Test: 0.7880
    Epoch: 054, Loss: 1.0880, Train: 0.9571, Val: 0.7620, Test: 0.7880
    Epoch: 055, Loss: 1.0617, Train: 0.9571, Val: 0.7580, Test: 0.7880
    Epoch: 056, Loss: 1.0410, Train: 0.9643, Val: 0.7600, Test: 0.7880
    Epoch: 057, Loss: 1.0352, Train: 0.9643, Val: 0.7620, Test: 0.7880
    Epoch: 058, Loss: 1.0271, Train: 0.9643, Val: 0.7680, Test: 0.7880
    Epoch: 059, Loss: 0.9928, Train: 0.9643, Val: 0.7680, Test: 0.7880
    Epoch: 060, Loss: 1.0205, Train: 0.9643, Val: 0.7720, Test: 0.7880
    Epoch: 061, Loss: 1.0038, Train: 0.9643, Val: 0.7740, Test: 0.7880
    Epoch: 062, Loss: 0.9809, Train: 0.9643, Val: 0.7740, Test: 0.7880
    Epoch: 063, Loss: 0.9509, Train: 0.9643, Val: 0.7740, Test: 0.7880
    Epoch: 064, Loss: 0.9133, Train: 0.9643, Val: 0.7720, Test: 0.7880
    Epoch: 065, Loss: 0.9303, Train: 0.9643, Val: 0.7740, Test: 0.7880
    Epoch: 066, Loss: 0.9378, Train: 0.9643, Val: 0.7780, Test: 0.7880
    Epoch: 067, Loss: 0.8676, Train: 0.9643, Val: 0.7840, Test: 0.8110
    Epoch: 068, Loss: 0.8609, Train: 0.9714, Val: 0.7840, Test: 0.8110
    Epoch: 069, Loss: 0.8127, Train: 0.9643, Val: 0.7880, Test: 0.8200
    Epoch: 070, Loss: 0.8994, Train: 0.9714, Val: 0.7880, Test: 0.8200
    Epoch: 071, Loss: 0.7771, Train: 0.9714, Val: 0.7920, Test: 0.8180
    Epoch: 072, Loss: 0.8375, Train: 0.9714, Val: 0.7880, Test: 0.8180
    Epoch: 073, Loss: 0.8174, Train: 0.9714, Val: 0.7900, Test: 0.8180
    Epoch: 074, Loss: 0.7833, Train: 0.9714, Val: 0.7920, Test: 0.8180
    Epoch: 075, Loss: 0.7510, Train: 0.9714, Val: 0.7900, Test: 0.8180
    Epoch: 076, Loss: 0.7898, Train: 0.9714, Val: 0.7880, Test: 0.8180
    Epoch: 077, Loss: 0.7931, Train: 0.9786, Val: 0.7840, Test: 0.8180
    Epoch: 078, Loss: 0.7608, Train: 0.9786, Val: 0.7860, Test: 0.8180
    Epoch: 079, Loss: 0.7193, Train: 0.9786, Val: 0.7840, Test: 0.8180
    Epoch: 080, Loss: 0.6972, Train: 0.9786, Val: 0.7900, Test: 0.8180
    Epoch: 081, Loss: 0.7126, Train: 0.9857, Val: 0.7860, Test: 0.8180
    Epoch: 082, Loss: 0.7176, Train: 0.9857, Val: 0.7840, Test: 0.8180
    Epoch: 083, Loss: 0.7042, Train: 0.9786, Val: 0.7800, Test: 0.8180
    Epoch: 084, Loss: 0.6833, Train: 0.9786, Val: 0.7820, Test: 0.8180
    Epoch: 085, Loss: 0.6981, Train: 0.9786, Val: 0.7880, Test: 0.8180
    Epoch: 086, Loss: 0.6565, Train: 0.9786, Val: 0.7880, Test: 0.8180
    Epoch: 087, Loss: 0.6837, Train: 0.9786, Val: 0.7860, Test: 0.8180
    Epoch: 088, Loss: 0.7371, Train: 0.9786, Val: 0.7900, Test: 0.8180
    Epoch: 089, Loss: 0.6373, Train: 0.9786, Val: 0.7940, Test: 0.8240
    Epoch: 090, Loss: 0.6574, Train: 0.9786, Val: 0.7980, Test: 0.8250
    Epoch: 091, Loss: 0.6248, Train: 0.9786, Val: 0.7980, Test: 0.8250
    Epoch: 092, Loss: 0.6330, Train: 0.9786, Val: 0.8020, Test: 0.8180
    Epoch: 093, Loss: 0.7066, Train: 0.9786, Val: 0.8000, Test: 0.8180
    Epoch: 094, Loss: 0.5868, Train: 0.9786, Val: 0.8060, Test: 0.8230
    Epoch: 095, Loss: 0.6133, Train: 0.9786, Val: 0.8040, Test: 0.8230
    Epoch: 096, Loss: 0.5794, Train: 0.9786, Val: 0.7960, Test: 0.8230
    Epoch: 097, Loss: 0.5593, Train: 0.9786, Val: 0.7880, Test: 0.8230
    Epoch: 098, Loss: 0.5757, Train: 0.9786, Val: 0.7840, Test: 0.8230
    Epoch: 099, Loss: 0.6419, Train: 0.9857, Val: 0.7820, Test: 0.8230
    Epoch: 100, Loss: 0.5809, Train: 0.9857, Val: 0.7780, Test: 0.8230


