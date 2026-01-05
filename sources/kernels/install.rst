安装指南
======================

这里是安装 huggingface/kernels 需要注意的一些注意事项。


昇腾环境安装
----------------------

请根据已有昇腾产品型号及 CPU 架构等按照 :doc:`快速安装昇腾环境指引 <../ascend/quick_install>` 进行昇腾环境安装。


安装 Kernels 包
----------------------

.. code-block:: shell
    :linenos:

    # 安装 kernels 包
    pip install kernels

.. warning::
    kernels 支持NPU的最低版本为0.11.0
