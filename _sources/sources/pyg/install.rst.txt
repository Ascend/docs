安装指南
==============

本教程面向使用 PyG & 昇腾的开发者，帮助完成昇腾环境下 PyG 的安装。

昇腾环境安装
------------

请根据已有昇腾产品型号及CPU架构等按照 :doc:`快速安装昇腾环境指引 <../ascend/quick_install>` 进行昇腾环境安装。

.. warning::
  CANN 最低版本为 8.0.rc1，安装 CANN 时，请同时安装 Kernel 算子包。

Python 环境创建
----------------------

.. code-block:: shell
    :linenos:
  
    # 创建 python 3.11 的虚拟环境
    conda create -y -n pyg python=3.11
    # 激活虚拟环境
    conda activate pyg


Torch 安装
----------------------

使用以下指令安装 torch 和 torch-npu:

.. code-block:: shell
    :linenos:

    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
    pip install torch_npu==2.6.0


PyG 安装
----------------------

使用以下指令安装 PyG:

.. code-block:: shell
    :linenos:

    pip install torch_geometric



进阶库安装 (Optional)
----------------------
PyG 还依赖一些进阶库，如 torch-scatter、torch-sparse 等。可以使用以下指令安装：

.. code-block:: shell
    :linenos:

    pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.6.0+cpu.html

