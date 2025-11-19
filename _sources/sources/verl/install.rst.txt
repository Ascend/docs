安装指南
==============

本教程面向使用 verl & Ascend 的开发者，帮助完成昇腾环境下 verl 的安装。

昇腾环境安装
------------

请根据已有昇腾产品型号及 CPU 架构等按照 :doc:`快速安装昇腾环境指引 <../ascend/quick_install>` 进行昇腾环境安装。

.. warning::
  CANN 最低版本为 8.3.RC1，安装 CANN 时，请同时安装 Kernel 算子包以及 nnal 加速库软件包。

Python 环境创建
----------------------

.. code-block:: shell
    :linenos:

    # 创建名为 verl 的 python 3.11 的虚拟环境
    conda create -y -n verl python==3.11
    # 激活虚拟环境
    conda activate verl

Torch 安装创建
----------------------

.. code-block:: shell
    :linenos:

    # 安装 torch 2.7.1 及 torch-npu 2.7.1 的 CPU 版本
    pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu

    # 安装 torch-npu 2.7.1
    pip install torch-npu==2.7.1

vllm & vllm-ascend 安装
----------------------


方法一：使用以下命令编译安装 vllm 和 vllm-ascend。请注意根据机器类型区分安装方式。

.. code-block:: shell
    :linenos:

    # vllm
    git clone -b v0.11.0 --depth 1 https://github.com/vllm-project/vllm.git
    cd vllm
    pip install -r requirements-build.txt

    # for Atlas 200T A2 Box16
    VLLM_TARGET_DEVICE=empty pip install -e . --extra-index https://download.pytorch.org/whl/cpu/

    # for Atlas 900 A2 PODc or Atlas 800T A3
    VLLM_TARGET_DEVICE=empty pip install -e .

.. code-block:: shell
    :linenos:

    # vllm-ascend
    git clone -b v0.11.0rc1 --depth 1 https://github.com/vllm-project/vllm-ascend.git
    cd vllm-ascend
    pip install -e .


方法二：使用以下命令直接安装预编译好的 vllm 和 vllm-ascend。

.. code-block:: shell
    :linenos:

    # Install vllm-project/vllm. The newest supported version is v0.11.0.
    pip install vllm==0.11.0

    # Install vllm-project/vllm-ascend from pypi.
    pip install vllm-ascend==0.11.0rc1

安装 verl
----------------------  

使用以下指令安装 verl 及相关依赖：

.. code-block:: shell
    :linenos:

    git clone https://github.com/volcengine/verl.git
    cd verl

    # Install verl NPU dependencies
    pip install -r requirements-npu.txt
    pip install -e .


其他第三方库说明
----------------------

+----------------------+---------------------------+
| Software             | Description               |
+======================+===========================+
| transformers         | >=v4.57.1                 |
+----------------------+---------------------------+
| flash_attn           | not supported             |
+----------------------+---------------------------+
| liger-kernel         | not supported             |
+----------------------+---------------------------+


1. 支持通过 transformers 使能 –flash_attention_2， transformers 需大于等于 4.57.1版本。

2. 不支持通过 flash_attn 使能 flash attention 加速。

3. 不支持 liger-kernel 使能。

4. 针对 x86 服务器，需要安装 cpu 版本的 torchvision。
   
.. code-block:: shell
    :linenos:

    pip install torchvision==0.20.1+cpu --index-url https://download.pytorch.org/whl/cpu