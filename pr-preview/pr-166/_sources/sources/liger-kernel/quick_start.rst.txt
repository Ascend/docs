快速上手
==============

.. note::

   阅读本篇前，请确保已按照 :doc:`安装指南 <./install>` 完成昇腾环境与 Liger-Kernel 的安装。

Liger-Kernel 提供面向 LLM 训练的高效 Triton 融合算子。在昇腾 NPU 上，只需一行 Patch 代码即可将 Hugging Face 模型中的关键算子替换为 Liger-Kernel 优化实现，从而提升训练吞吐并降低显存占用。


方式一：模型 Patch API
------------------------------------------------

如需更细粒度控制启用的算子，可在加载模型前调用对应 Patch 函数：

.. code-block:: python

    import torch
    import torch_npu
    from transformers import AutoModelForCausalLM
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3

    # 在实例化模型前调用，自动替换 RoPE、RMSNorm、SwiGLU、CrossEntropy 等算子
    apply_liger_kernel_to_qwen3(
        rope=True,
        rms_norm=True,
        swiglu=True,
        cross_entropy=True,
        fused_linear_cross_entropy=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "/home/model/Qwen3-0.6B",
        dtype=torch.bfloat16,
        device_map="npu",
    )

# 打印模型结构，检查是否替换成功
print(model)

其他常用模型可使用对应的 Patch API，例如 ``apply_liger_kernel_to_llama``、``apply_liger_kernel_to_qwen3``、``apply_liger_kernel_to_mistral`` 等，完整列表见 `Liger-Kernel 官方文档 <https://linkedin.github.io/Liger-Kernel/>`_。

.. hint::

   使用 ``device_map="npu"`` 需要安装 ``accelerate``（``pip install accelerate``）。也可在加载模型后手动迁移：``model = model.to("npu")``。

方式二：单独使用算子模块
------------------------------------------------

也可以将单个 Liger 算子嵌入自定义模型：

.. code-block:: python

    import torch
    import torch_npu
    from liger_kernel.transformers import LigerRMSNorm, LigerFusedLinearCrossEntropyLoss

    hidden_size = 4096
    vocab_size = 32000

    norm = LigerRMSNorm(hidden_size).npu()
    loss_fn = LigerFusedLinearCrossEntropyLoss()

    x = torch.randn(4, 128, hidden_size, dtype=torch.bfloat16, device="npu", requires_grad=True)
    target = torch.randint(vocab_size, (4, 128), device="npu")
    weight = torch.randn(vocab_size, hidden_size, dtype=torch.bfloat16, device="npu", requires_grad=True)

    hidden = norm(x)
    # LigerFusedLinearCrossEntropyLoss 的输入需为二维张量 (batch*seq, hidden_size)
    loss = loss_fn(weight, hidden.reshape(-1, hidden_size), target.reshape(-1))
    loss.backward()

训练框架集成
------------------------------------------------

Liger-Kernel 已集成到多个训练框架，在 NPU 上可通过框架配置启用：

- `LLaMA-Factory <https://github.com/hiyouga/LLaMA-Factory>`_：训练参数中设置 ``--use_liger_kernel true``
- `VeOmni <https://github.com/ByteDance-Seed/VeOmni>`_：模型 Patch 中启用 Liger-Kernel
- `verl <https://github.com/volcengine/verl>`_：配置 ``model.use_liger=true``
