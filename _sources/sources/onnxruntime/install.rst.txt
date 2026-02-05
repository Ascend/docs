安装指南
===========

本教程面向使用 ONNX Runtime & Ascend NPU 的开发者，帮助完成昇腾环境下 ONNX Runtime 的安装。

.. note::
    
    阅读本篇前，请确保已按照 :doc:`安装教程 <../ascend/quick_install>` 准备好昇腾环境！

ONNX Runtime 安装
-------------------

ONNX Runtime 目前提供了 源码编译 和 二进制包 两种安装方式，其中二进制包当前只支持Python。

从源码安装
^^^^^^^^^^^^

.. code-block:: shell
    :linenos:

    # Default path, change it if needed.
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    ./build.sh --config <Release|Debug|RelWithDebInfo> --build_shared_lib --parallel --use_cann


从pip安装
^^^^^^^^^^^^

.. code-block:: shell
    :linenos:

    pip3 install onnxruntime-cann
