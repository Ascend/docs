安装指南
===============

本教程面向使用 TRL (Transformer Reinforcement Learning) & 昇腾的开发者，帮助完成昇腾环境下 TRL 的安装。

昇腾环境安装
---------------

请根据已有昇腾产品型号及 CPU 架构等按照 :doc:`快速安装昇腾环境指引 <../ascend/quick_install>` 进行昇腾环境安装。

.. warning::
  CANN 最低版本为 8.0.rc1，安装 CANN 时，请同时安装 Kernel 算子包。

TRL 下载安装
---------------

1. 安装项目所需依赖

.. code-block:: shell

    pip install trl -i https://pypi.tuna.tsinghua.edu.cn/simple

另外，本项目需要手动安装 transformers 仓库的最新 main 分支，否则可能会出现如下错误：
  
.. image:: ./images/image.png

.. code-block:: shell

    pip install git+https://github.com/huggingface/transformers.git

2. 安装 torch_npu

.. code-block:: shell

    pip install torch==2.1.0 torch_npu==2.1.0.post6 -i https://pypi.tuna.tsinghua.edu.cn/simple

.. hint::
  torch_npu 的版本需要匹配 torch 的版本，详细信息请参考：`Ascend Extension for PyTorch <https://github.com/Ascend/pytorch>`_。
