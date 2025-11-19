分布式训练
==============

.. note::
    阅读本篇前，请确保已按照 :doc:`安装教程 <./install>` 准备好昇腾环境及 LLaMA-Factory ！

本篇为 :doc:`快速开始 <./quick_start>` 的进阶，同样首先安装 DeepSpeed 和 ModelScope：

.. code-block::

  pip install -e ".[deepspeed,modelscope]" -i https://pypi.tuna.tsinghua.edu.cn/simple

多卡 NPU 指定
--------------------------

无论是单机还是多机环境，请先使用 ``export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`` 显式指定所需 NPU 卡号，此处为 0~3 四卡 NPU。

.. note::
    
    昇腾 NPU 卡从 0 开始编号，docker 容器内也是如此；
    
    如映射物理机上的 6，7 号 NPU 卡到容器内使用，其对应的卡号分别为 0，1


或使用以下脚本自动检测并指定多卡 NPU：

.. code-block:: shell

    # ------------------------------ detect npu --------------------------------------
    # detect npu via npu-smi
    if command -v npu-smi info &> /dev/null; then
      num_npus=$(npu-smi info -l | grep "Total Count" | awk -F ":" '{print $NF}')
      npu_list=$(seq -s, 0 $((num_npus-1)))
    else
      num_npus=-1
      npu_list="-1"
    fi
    echo using npu : $npu_list
    num_gpus=$(echo $npu_list | awk -F "," '{print NF}')
    # --------------------------------------------------------------------------------
    export ASCEND_RT_VISIBLE_DEVICES=$npu_list


单机多卡微调
--------------------------

通过 ``ASCEND_RT_VISIBLE_DEVICES`` 变量显式指定多卡后，使用 torchrun 启动分布式训练，需指定 ``nproc_per_node`` 参数为 NPU 卡数量，其余参数配置与 :doc:`快速开始 <./quick_start>` 中单卡微调保持一致

.. code-block:: shell
    
    torchrun --nproc_per_node $num_npus \
        --nnodes 1 \
        --node_rank 0 \
        --master_addr 127.0.0.1 \
        --master_port 7007 \
        src/train.py <your_path>/qwen1_5_lora_sft_ds.yaml


多机多卡微调
--------------------------

在每个节点上通过 ``ASCEND_RT_VISIBLE_DEVICES`` 变量显式指定需要用到的多个 NPU 卡，如果不指定，则默认使用所有 NPU 卡（此时需要确保显存是否够用）。

其次，必须在每个节点上使用 ``export HCCL_SOCKET_IFNAME=eth0`` 来指定当前节点的 HCCL 通信网卡（请使用目标网卡名替换 ``eth0``）。

以两机环境为例，分别在主、从节点（机器）上执行如下两条命令即可启动多机训练：

.. code-block:: shell

    # 在主节点执行如下命令，设置 rank_id = 0
    FORCE_TORCHRUN=1 NNODES=2 NODE_RANK=0 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 \
    llamafactory-cli train <your_path>/qwen1_5_lora_sft_ds.yaml
    
    # 在从节点执行如下命令，设置 rank_id = 1
    FORCE_TORCHRUN=1 NNODES=2 NODE_RANK=1 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 \
    llamafactory-cli train <your_path>/qwen1_5_lora_sft_ds.yaml

.. list-table::
    :widths: 30 70  
    :header-rows: 1

    * - 变量名
      - 介绍
    * - FORCE_TORCHRUN
      - 是否强制使用torchrun
    * - NNODES
      - 节点数量
    * - NODE_RANK
      - 各个节点的rank。
    * - MASTER_ADDR
      - 主节点的地址。
    * - MASTER_PORT
      - 主节点的端口。