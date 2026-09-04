安装指南
==============

本教程面向使用 WeNet & 昇腾的开发者，帮助完成昇腾环境下 WeNet 的安装。

昇腾环境安装
------------

请根据已有昇腾产品型号及CPU架构等按照 :doc:`快速安装昇腾环境指引 <../ascend/quick_install>` 进行昇腾环境安装。

.. warning::
  CANN 最低版本为 8.0.rc1，安装 CANN 时，请同时安装 Kernel 算子包。

Python 环境创建
----------------------

.. code-block:: shell
    :linenos:
  
    # 创建名为 wenet 的 python 3.10 的虚拟环境
    conda create -y -n wenet python=3.10
    # 激活虚拟环境
    conda activate wenet


WeNet 安装
----------------------

使用以下指令安装带有 torch-npu 的 WeNet 及训练相关依赖：

.. code-block:: shell
    :linenos:

    # 安装带有 torch-npu 的 WeNet
    pip install -e .[torch-npu]

    # 安装 WeNet 训练相关依赖
    pip install -r requirements.txt

请遵循以下 torch-npu 相关库的版本控制：

+------------+------------------+-----------+
| Requirement|  Minimum         | Recommend |
+============+==================+===========+
| CANN       | 8.0.RC2.alpha003 | latest    |
+------------+------------------+-----------+
|   torch    | 2.1.0            |   2.2.0   |
+------------+------------------+-----------+
|  torch-npu | 2.1.0            |   2.2.0   |
+------------+------------------+-----------+
| torchaudio | 2.1.0            |   2.2.0   |
+------------+------------------+-----------+
| deepspeed  | 0.13.2           |   latest  |
+------------+------------------+-----------+



安装校验
----------------------

使用以下 Python 脚本对 open_clip 的安装进行校验，正确打印 open_clip 的版本号和 NPU 卡号说明安装成功。

.. code-block:: python
    :linenos:
    :emphasize-lines: 5,6

    import torch
    import torch_npu
    import timm

    print("timm version:", timm.version.__version__)
    print("NPU devices:", torch.npu.current_device())

正确回显如下（单卡 NPU 环境）：

.. code-block:: shell

  timm version: 1.0.8.dev0
  NPU devices: 0
