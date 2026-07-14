安装指南
==============

本教程面向使用 OpenCompass & 昇腾的开发者，帮助完成昇腾环境下 OpenCompass 的安装。

昇腾环境安装
------------

请根据已有昇腾产品型号及CPU架构等按照 :doc:`快速安装昇腾环境指引 <../ascend/quick_install>` 进行昇腾环境安装。

.. warning::
  CANN 最低版本为 8.0.rc1，安装 CANN 时，请同时安装 Kernel 算子包。

Python 环境创建
----------------------

.. code-block:: shell
    :linenos:
  
    # 创建 python 3.10 的虚拟环境
    conda create -y -n opencompass python=3.10
    # 激活虚拟环境
    conda activate opencompass


OpenCompass 安装
----------------------

使用以下指令安装 OpenCompass:

.. code-block:: shell
    :linenos:

      pip install -U opencompass -i https://pypi.tuna.tsinghua.edu.cn/simple

    ## Full installation (with support for more datasets)
    # pip install "opencompass[full]"

    ## Environment with model acceleration frameworks
    ## Manage different acceleration frameworks using virtual environments
    ## since they usually have dependency conflicts with each other.
    # pip install "opencompass[lmdeploy]"
    # pip install "opencompass[vllm]"

    ## API evaluation (i.e. Openai, Qwen)
    # pip install "opencompass[api]"

torch-npu 安装
----------------------

按照 :doc:`torch-npu 安装指引 <../pytorch/install>` 安装 2.1.0 版本 torch 和 torch-npu，或使用以下指令快速安装：

.. code-block:: shell
    :linenos:

    # install the dependencies
    pip3 install attrs numpy==1.26.4 decorator sympy cffi pyyaml pathlib2 psutil protobuf scipy requests absl-py wheel typing_extensions -i https://pypi.tuna.tsinghua.edu.cn/simple
    # install torch and torch-npu
    pip install torch==2.1.0 torch-npu==2.1.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

安装校验
----------------------

使用以下 Python 脚本对 open_clip 的安装进行校验，正确打印 open_clip 的版本号和 NPU 卡号说明安装成功。

.. code-block:: python
    :linenos:
    :emphasize-lines: 2

    import torch
    import opencompass

    print("opencompass version: ", opencompass.__version__)
    print("NPU devices: ", torch.npu.current_device())

正确回显如下（单卡 NPU 环境）：

.. code-block:: shell

  opencompass version:  0.3.3
  NPU devices: 0
