快速开始
============

.. note::
    阅读本篇前，请确保已按照 :doc:`安装教程 <./install>` 准备好昇腾环境及 LLaMA-Factory ！
    
本教程聚焦大语言模型（Large Language Model，LLM）的微调过程，以 Qwen1.5-7B 模型为例，讲述如何使用 LLaMA-Factory 在昇腾 NPU 上进行 LoRA 微调及推理。

本篇将使用到 DeepSpeed 和 ModelScope，请使用以下指令安装：

.. code-block::

  pip install -e '.[deepspeed,modelscope]'

环境变量配置
-------------

通过环境变量设置单卡 NPU，并使用 ModelScope 下载模型/数据集：

.. code-block:: shell
  
  export ASCEND_RT_VISIBLE_DEVICES=0
  export USE_MODELSCOPE_HUB=1

多卡 NPU 用户请参考 :ref:`multi_npu`

基于 LoRA 的模型微调
------------------------

.. <!-- TODO: 确认是否只有这两个是必须指定的参数 -->

使用 torchrun 启动微调，通过 ``model_name_or_path`` 指定模型， ``output_dir`` 参数指定输出文件保存路径。
微调涉及的所有参数均在 :ref:`qwen_yaml` 中设置。

.. code-block:: shell
    
    torchrun --nproc_per_node 1 \
        --nnodes 1 \
        --node_rank 0 \
        --master_addr 127.0.0.1 \
        --master_port 7007 \
        src/train.py <your_path>/qwen1_5_lora_sft_ds.yaml

.. note::

  ``nproc_per_node, nnodes, node_rank, master_addr, master_port`` 为 torchrun 所需参数，其详细含义可参考 `PyTorch 官方文档 <https://pytorch.org/docs/stable/elastic/run.html>`_。

动态合并 LoRA 的推理
---------------------

经 LoRA 微调后，通过 ``llamafactory-cli chat`` 使用微调后的模型进行推理，指定 ``adapter_name_or_path`` 参数为 LoRA 微调模型的存储路径：

.. code-block:: shell

    llamafactory-cli chat --model_name_or_path qwen/Qwen1.5-7B \
                --adapter_name_or_path saves/Qwen1.5-7B/lora/sft \
                --template qwen \
                --finetuning_type lora

.. note::
  确保微调及推理阶段使用同一 ``template`` 参数

接下来即可在终端使用微调的模型进行问答聊天了！如下图所示，为在 NPU 成功推理的样例：

.. figure:: ./images/chat.png
  :align: left

.. note::
  第一轮问答会有一些 warning 告警，这是由于 transformers 库更新所致，不影响推理的正常运行，请忽略

完整脚本
-----------

推理及微调脚本
~~~~~~~~~~~~~~~~

使用 Qwen1.5-7B 模型微调和推理的完整脚本如下：

.. code-block:: shell

    # use modelscope
    export USE_MODELSCOPE_HUB=1

    # specify NPU
    export ASCEND_RT_VISIBLE_DEVICES=0 

    ### qwen/Qwen1.5-7B
    ### finetune
    torchrun --nproc_per_node 1 \
        --nnodes 1 \
        --node_rank 0 \
        --master_addr 127.0.0.1 \
        --master_port 7007 \
        src/train.py <your_path>/qwen1_5_lora_sft_ds.yaml

    ### inference -- chat
    llamafactory-cli chat --model_name_or_path qwen/Qwen1.5-7B \
                --adapter_name_or_path saves/Qwen1.5-7B/lora/sft \
                --template qwen \
                --finetuning_type lora

.. _qwen_yaml:

yaml 配置文件
~~~~~~~~~~~~~~~~

完整 qwen1_5_lora_sft_ds.yaml：

.. literalinclude:: ./qwen1_5_lora_sft_ds.yaml
    :language: yaml
    :linenos:


.. _multi_npu:

多卡 NPU 的使用
-----------------

使用以下脚本自动检测并指定多卡 NPU：

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

或使用 ``export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`` 指定所需 NPU 卡号，此处为 0~3 四卡 NPU。

.. note::
    
    昇腾 NPU 卡从 0 开始编号，docker 容器内也是如此；
    
    如映射物理机上的 6，7 号 NPU 卡到容器内使用，其对应的卡号分别为 0，1
