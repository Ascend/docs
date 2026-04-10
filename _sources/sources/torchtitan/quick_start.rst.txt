快速开始
==================

.. note::
    阅读本篇前，请确保已按照 :doc:`安装教程 <./install>` 准备好昇腾环境及 TorchTitan 

本文档帮助昇腾开发者快速使用 TorchTitan × 昇腾 进行LLM预训练。你可以访问 `这篇官方论文 <https://arxiv.org/abs/2410.06511>`_ 获取更多信息。

概览
---------------------

TorchTitan是一个用于LLM预训练的PyTorch原生库，用于大规模的分布式训练。TorchTitan支持不同种类的并行训练方式，包括FSDP、TP、PP、CP，也支持2D、3D的多种并行技术的组合。另外，项目还支持Float8以及混合精度训练等方式提高训练速度。



配置训练参数
---------------------

.. note::

    在 TorchTitan 中，每个训练任务的参数可以由CLI和TOML文件来进行配置。其优先级是CLI > TOML文件 > 默认值


以下选取部分可配置参数进行介绍，用户可以在 torchtitan/config_manager.py 中找到所有可配置参数的说明。

.. list-table:: TorchTitan 主要配置项参数说明
   :widths: 15 30 25
   :header-rows: 1

   * - 配置项参数
     - 描述
     - 默认数值
   * - --training.dataset
     - 使用的数据集
     - c4_test
   * - --training.batch_size
     - 批量大小
     - 8
   * - --training.seq_len
     - 序列长度
     - 2048
   * - --training.steps
     - 训练步数
     - 10000
   * - --parallelism.data_parallel_replicate_degree
     - 权重复制的数据并行程度，1表示禁用
     - 1
   * - --parallelism.data_parallel_shard_degree
     - 权重分片的数据并行程度，-1表示自动使用所有可用设备数量，1表示禁用
     - -1
   * - --parallelism.tensor_parallel_degree
     - 张量并行度，1表示禁用
     - 1
   * - --parallelism.pipeline_parallel_degree
     - 流水线并行度，1表示禁用
     - 1
   * - --parallelism.context_parallel_degree
     - 上下文并行度，1表示禁用
     - 1
   * - --float8.enable_fsdp_float8_all_gather
     - 是否在FSDP应用float8 all gather
     - false
   



训练示例
---------------------

下面示例有单机8张NPU卡下配置不同的分布式策略，多卡场景根据卡数进行调整：

DDP

.. code-block:: shell
    :linenos:

    data_parallel_replicate_degree = 8
    data_parallel_shard_degree = -1
    tensor_parallel_degree = 1
    pipeline_parallel_degree = 1
    context_parallel_degree = 1

FSDP

.. code-block:: shell
    :linenos:

    data_parallel_replicate_degree = 1
    data_parallel_shard_degree = -1 #-1为默认FSDP
    tensor_parallel_degree = 1
    pipeline_parallel_degree = 1
    context_parallel_degree = 1

TP

.. code-block:: shell
    :linenos:

    data_parallel_replicate_degree = 1
    data_parallel_shard_degree = -1
    tensor_parallel_degree = 8
    pipeline_parallel_degree = 1
    context_parallel_degree = 1

FSDP+TP

.. code-block:: shell
    :linenos:

    data_parallel_replicate_degree = 1
    data_parallel_shard_degree = -1 #自动计算剩余度数
    tensor_parallel_degree = 2 or 4
    pipeline_parallel_degree = 1
    context_parallel_degree = 1

FSDP+TP+PP

.. code-block:: shell
    :linenos:

    data_parallel_replicate_degree = 1
    data_parallel_shard_degree = -1 #自动计算剩余度数
    tensor_parallel_degree = 2 
    pipeline_parallel_degree = 2
    context_parallel_degree = 1

如需改变NPU的数量，可以在 ./run_train.sh 中进行修改

.. code-block:: shell
    :linenos:

    # use envs as local overrides for convenience
    # e.g.
    # LOG_RANK=0,1 NGPU=4 ./run_train.sh
    NGPU=${NGPU:-"8"}
    export LOG_RANK=${LOG_RANK:-0}
    CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/llama3/train_configs/debug_model.toml"}

完成配置后，即可按照debug_model.toml文件配置开始预训练，该模型较小，可以用于进行测试：

.. code-block:: shell
    :linenos:

    ./run_train.sh

或指定llama3_8b.toml文件配置开始预训练Llama3-8B模型：

.. code-block:: shell
    :linenos:

    CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" ./run_train.sh


当看到每个训练步骤的训练参数信息，即已经开始训练，看到 Training Completed 为训练成功完成。