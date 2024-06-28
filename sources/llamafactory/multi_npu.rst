单机多卡微调
==============

.. note::
    阅读本篇前，请确保已按照 :doc:`安装教程 <./install>` 准备好昇腾环境及 LLaMA-Factory ！

本篇为 :doc:`快速开始 <./quick_start>` 的进阶，同样首先安装 DeepSpeed 和 ModelScope：

.. code-block::

  pip install -e ".[deepspeed,modelscope]" -i https://pypi.tuna.tsinghua.edu.cn/simple

多卡 NPU 指定
--------------------------

使用 ``export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`` 指定所需 NPU 卡号，此处为 0~3 四卡 NPU。

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

基于 LoRA 的模型多卡分布式微调
-------------------------------

通过 ``ASCEND_RT_VISIBLE_DEVICES`` 变量指定多卡后，使用 torchrun 启动分布式训练，需指定 ``nproc_per_node`` 参数为 NPU 卡数量，其余参数配置与 :doc:`快速开始 <./quick_start>` 中单卡微调保持一致

.. code-block:: shell
    
    torchrun --nproc_per_node $num_npus \
        --nnodes 1 \
        --node_rank 0 \
        --master_addr 127.0.0.1 \
        --master_port 7007 \
        src/train.py <your_path>/qwen1_5_lora_sft_ds.yaml
