安装指南
=====================================================

这里是安装 huggingface/kernels 需要注意的一些事项。


昇腾环境安装
-----------------------------------------------------

请根据已有昇腾产品型号及 CPU 架构等按照 :doc:`快速安装昇腾环境指引 <../ascend/quick_install>` 进行昇腾环境安装。


安装 torch 和 torch_npu
-----------------------------------------------------

.. code-block:: shell
    :linenos:

    # 安装 torch
    pip install torch==2.7.1

    # 安装 torch_npu
    pip install torch_npu==2.7.1

.. warning::
    torch 和 torch_npu 的最低版本为 v2.5.1，请确认安装的版本大于等于此版本，这里推荐的版本是 v2.7.1。


安装 kernels 包
------------------------------------------------------

.. code-block:: shell
    :linenos:

    # 安装 kernels 包
    pip install kernels

.. warning::
    kernels 支持 NPU 的最低版本为 v0.11.0，请确认安装的版本大于等于此版本。
