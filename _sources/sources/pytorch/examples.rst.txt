样例
==========

这些示例将会帮助您快速了解如何在Ascend NPU上使用PyTorch的相关特性。

.. note::

   在运行下述示例之前，需要您已经安装了PyTorch-NPU环境，有关环境安装，请参考 :doc:`./install`

数据并行
-----------------------

PyTorch的数据并行主要分为以下几种：DP、DDP以及FSDP（HSDP变种），接下来将简单描述在Ascend NPU场景下如何实现上述数据并行。

DDP
^^^^^^

.. code-block:: python
    :linenos:
    :emphasize-lines: 11,12,31,35,39,47,49

    # encoding: UTF-8

    import os
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import torch.nn as nn
    import torch.optim as optim
    from torch.nn.parallel import DistributedDataParallel as DDP

    # 引入torch-npu包
    import torch_npu


    class ToyModel(nn.Module):
        def __init__(self):
            super(ToyModel, self).__init__()
            self.net1 = nn.Linear(10, 10)
            self.relu = nn.ReLU()
            self.net2 = nn.Linear(10, 5)

        def forward(self, x):
            return self.net2(self.relu(self.net1(x)))


    def setup(rank, world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

        # initialize the process group
        dist.init_process_group("hccl", rank=rank, world_size=world_size)


    def example(rank, world_size):
        device = torch.device("npu:{}".format(rank))
        # create default process group
        setup(rank, world_size)
        # create local model
        model = ToyModel().to(device)
        # construct DDP model
        ddp_model = DDP(model, device_ids=[rank])
        # define loss function and optimizer
        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

        # forward pass
        outputs = ddp_model(torch.randn(20, 10).to(device))
        # backward pass
        labels = torch.randn(20, 5).to(device)
        loss_fn(outputs, labels).backward()
        # update parameters
        optimizer.step()


    def main():
        n_npus = torch.cuda.device_count()
        assert n_npus >= 2, f"Requires at least 2 NPUs to run, but got {n_npus}"
        world_size = n_npus
        mp.spawn(example, args=(world_size,), nprocs=world_size, join=True)


    if __name__ == "__main__":
        main()

FSDP
^^^^^^

.. code-block:: python
    :linenos:
    :emphasize-lines: 11,12,31,35,39,47,49

    # encoding: UTF-8

    import os
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    # 引入torch-npu包
    import torch_npu


    class ToyModel(nn.Module):
        def __init__(self):
            super(ToyModel, self).__init__()
            self.net1 = nn.Linear(10, 10)
            self.relu = nn.ReLU()
            self.net2 = nn.Linear(10, 5)

        def forward(self, x):
            return self.net2(self.relu(self.net1(x)))


    def setup(rank, world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

        # initialize the process group
        dist.init_process_group("hccl", rank=rank, world_size=world_size)


    def example(rank, world_size):
        device = torch.device("npu:{}".format(rank))
        # create default process group
        setup(rank, world_size)
        # create local model
        model = ToyModel().to(device)
        # construct FSDP model
        ddp_model = FSDP(model, device_id=rank)
        # define loss function and optimizer
        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

        # forward pass
        outputs = ddp_model(torch.randn(20, 10).to(device))
        # backward pass
        labels = torch.randn(20, 5).to(device)
        loss_fn(outputs, labels).backward()
        # update parameters
        optimizer.step()


    def main():
        n_npus = torch.cuda.device_count()
        assert n_npus >= 2, f"Requires at least 2 NPUs to run, but got {n_npus}"
        world_size = n_npus
        mp.spawn(example, args=(world_size,), nprocs=world_size, join=True)


    if __name__ == "__main__":
        main()
