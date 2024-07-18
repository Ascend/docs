安装指南
==============

本教程面向使用 pytorch-image-models (timm) & 昇腾的开发者，帮助完成昇腾环境下 timm 的安装。

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
    conda create -n <your_env_name> python=3.10
    # 激活虚拟环境
    conda activate <your_env_name>


timm 安装
----------------------

使用以下指令安装 timm：

.. code-block:: shell
    :linenos:

    pip install timm

torch-npu 安装
----------------------

按照 :doc:`torch-npu 安装指引 <../pytorch/install>` 安装 2.2.0 版本 torch 和 torch-npu，或使用以下指令快速安装：

.. code-block:: shell
    :linenos:

    # install the dependencies
    pip3 install attrs numpy==1.26.4 decorator sympy cffi pyyaml pathlib2 psutil protobuf scipy requests absl-py wheel typing_extensions
    # install torch and torch-npu
    pip install torch==2.2.0 torch-npu==2.2.0

安装校验
----------------------

使用以下 Python 脚本对 open_clip 的安装进行校验，正确打印 open_clip 的版本号和 NPU 卡号说明安装成功。

.. code-block:: python
    :linenos:
    :emphasize-lines: 2

    import torch
    import torch_npu
    import timm

    print("timm version:", timm.version.__version__)
    print("NPU devices:", torch.npu.current_device())

正确回显如下（单卡 NPU 环境）：

.. code-block:: shell

  timm version: 1.0.8.dev0
  NPU devices: 0
