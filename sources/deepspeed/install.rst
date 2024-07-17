安装DeepSpeed
==============

.. note:: 
    在本示例之前，请确保已经安装了 `昇腾环境 <../ascend/quick_install.html>`_ 和 `PyTorch <../pytorch/install.html>`_ 环境。

1. 安装DeepSpeed
-----------------
安装DeepSpeed最简单的方式是通过 ``pip`` 。

.. code-block:: shell
    :linenos:

    pip install deepspeed


2. 通过源码安装
------------------
从 `GitHub <https://github.com/microsoft/DeepSpeed>`_ 克隆DeepSpeed项目后，可以通过 ``pip`` 来通过源码编译。

.. code-block:: shell
    :linenos:

    pip install .


3. 预编译DeepSpeed算子（可选）
----------------------------------
如果不想使用JIT编译模式，而想要预编译DeepSpeed算子，可以通过设置环境变量的方式完成算子的预编译。

.. code-block:: shell
    :linenos:

    DS_BUILD_OPS=1 pip install deepspeed

4. 安装验证
-----------

安装完成后，可以通过 ``ds_report`` 命令查看安装结果

.. code-block:: shell
    :linenos:

    --------------------------------------------------
    DeepSpeed C++/CUDA extension op report
    --------------------------------------------------
    NOTE: Ops not installed will be just-in-time (JIT) compiled at
        runtime if needed. Op compatibility means that your system
        meet the required dependencies to JIT install the op.
    --------------------------------------------------
    JIT compiled ops requires ninja
    ninja .................. [OKAY]
    --------------------------------------------------
    op name ................ installed .. compatible
    --------------------------------------------------
    deepspeed_not_implemented  [NO] ....... [OKAY]
    async_io ............... [NO] ....... [OKAY]
    cpu_adagrad ............ [NO] ....... [OKAY]
    cpu_adam ............... [NO] ....... [OKAY]
    cpu_lion ............... [NO] ....... [OKAY]
    fused_adam ............. [NO] ....... [OKAY]
    transformer_inference .. [NO] ....... [OKAY]
    --------------------------------------------------
    DeepSpeed general environment info:
    torch install path ............... ['/root/miniconda3/envs/ds/lib/python3.10/site-packages/torch']
    torch version .................... 2.2.0
    deepspeed install path ........... ['/root/miniconda3/envs/ds/lib/python3.10/site-packages/deepspeed']
    deepspeed info ................... 0.14.4, unknown, unknown
    deepspeed wheel compiled w. ...... torch 2.2
    torch_npu install path ........... ['/root/miniconda3/envs/ds/lib/python3.10/site-packages/torch_npu']
    torch_npu version ................ 2.2.0
    ascend_cann version .............. 8.0.RC2.alpha002
    shared memory (/dev/shm) size .... 20.00 GB