安装指南
==============

本教程面向在昇腾 NPU 上使用 `Liger-Kernel <https://github.com/linkedin/Liger-Kernel>`_ 的开发者，帮助完成 NPU 环境下的安装与验证。

Liger-Kernel 从 **v0.8.0** 起正式支持昇腾 NPU 后端，通过 Triton-Ascend 提供 RMSNorm、RoPE、SwiGLU、CrossEntropy 等融合算子，用于加速 LLM 训练并降低显存占用。

昇腾环境安装
---------------

请根据已有昇腾产品型号及 CPU 架构等，按照 :doc:`快速安装昇腾环境指引 <../ascend/quick_install>` 完成 CANN、驱动与固件的安装。

.. warning::

   安装 CANN 时请同时安装 ops 算子包。torch_npu 与 CANN 版本需匹配，详见 :doc:`PyTorch 安装指引 <../pytorch/install>`。


安装依赖
---------------

安装 Liger-Kernel
-------------------

从源码安装：

.. code-block:: shell

    git clone https://github.com/linkedin/Liger-Kernel.git
    cd Liger-Kernel
    pip install -e ".[dev]"

源码安装时，``setup.py`` 会自动检测昇腾 NPU 并安装对应依赖（``torch==2.6.0``、``torch_npu==2.6.0``、``triton-ascend==3.2.0``）。


验证安装
---------------

使用以下脚本验证环境是否就绪：

.. code-block:: python

    import torch
    import torch_npu
    from liger_kernel.transformers import LigerRMSNorm

    assert torch.npu.is_available(), "NPU 不可用"
    print("NPU device:", torch.npu.current_device())

    norm = LigerRMSNorm(128).npu()
    x = torch.randn(2, 16, 128, dtype=torch.bfloat16, device="npu")
    y = norm(x)
    print("LigerRMSNorm output shape:", y.shape)

正确回显应包含 NPU 设备号与张量 shape，且无报错。
