FAQ
=======

设备指定
--------

**Q：为什么我的 NPU 卡没调用起来？**

1. 通过 ``ASCEND_RT_VISIBLE_DEVICES`` 环境变量指定昇腾 NPU 卡，如 ``ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`` 指定使用 0，1，2，3四张 NPU 卡进行微调/推理。

.. hint::
    
    昇腾 NPU 卡从 0 开始编号，docker 容器内也是如此；
    如映射物理机上的 6，7 号 NPU 卡到容器内使用，其对应的卡号分别为 0，1

2. 检查是否安装 torch-npu，建议通过 ``pip install -e '.[torch-npu,metrics]'`` 安装 LLaMA-Factory。

推理报错
----------

**Q：使用昇腾 NPU 推理报错 RuntimeError: ACL stream synchronize failed, error code:507018**

A：设置 do_sample: false，取消随机抽样策略

关联 issues：

- https://github.com/hiyouga/LLaMA-Factory/issues/3840

微调/训练报错
--------------

**Q：使用 ChatGLM 系列模型微调/训练模型时，报错 NotImplementedError: Unknown device for graph fuser**

A：在 modelscope 或 huggingface 下载的 repo 里修改 ``modeling_chatglm.py`` 代码，取消 torch.jit 装饰器注释

关联 issues：

- https://github.com/hiyouga/LLaMA-Factory/issues/3788
- https://github.com/hiyouga/LLaMA-Factory/issues/4228


**Q：微调/训练启动后，HCCL 报错，包含如下关键信息：**

.. code-block:: shell

        RuntimeError: [ERROR] HCCL error in: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp:64
    [ERROR] 2024-05-21-11:57:54 (PID:927000, Device:3, RankID:3) ERR02200 DIST call hccl api failed.
    EJ0001: 2024-05-21-11:57:54.167.645 Failed to initialize the HCCP process. Reason: Maybe the last training process is running.
            Solution: Wait for 10s after killing the last training process and try again.
            TraceBack (most recent call last):
            tsd client wait response fail, device response code[1]. unknown device error.[FUNC:WaitRsp][FILE:process_mode_manager.cpp][LINE:290]
            Fail to get sq reg virtual addr, deviceId=3, sqId=40.[FUNC:Setup][FILE:stream.cc][LINE:1102]
            stream setup failed, retCode=0x7020010.[FUNC:SyncGetDevMsg][FILE:api_impl.cc][LINE:4643]
            Sync get device msg failed, retCode=0x7020010.[FUNC:GetDevErrMsg][FILE:api_impl.cc][LINE:4704]
            rtGetDevMsg execute failed, reason=[driver error:internal error][FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:53]

A：杀掉 device 侧所有进程，等待 10s 后重新启动训练。

关联 issues：

- https://github.com/hiyouga/LLaMA-Factory/issues/3839

.. **Q：微调 ChatGLM3 使用 fp16 报错 Gradient overflow. Skipping step Loss scaler reducing loss scale to ...；使用 bf16 时 'loss': 0.0, 'grad_norm': nan**
.. https://github.com/hiyouga/LLaMA-Factory/issues/3308


**Q：使用 TeleChat 模型在昇腾 NPU 推理时，报错 AssertionError： Torch not compiled with CUDA enabled**

A：此问题一般由代码中包含 cuda 相关硬编码造成，根据报错信息，找到 cuda 硬编码所在位置，对应修改为 NPU 代码。如 ``.cuda()`` 替换为 ``.npu()`` ； ``.to("cuda")`` 替换为  ``.to("npu")`` 

**Q：模型微调遇到报错 DeviceType must be NPU. Actual DeviceType is: cpu，例如下列报错信息**

.. code-block:: shell

    File "/usr/local/pyenv/versions/3.10.13/envs/x/lib/python3.10/site-packages/transformers-4.41.1-py3.10.egg/transformers/generation/utils.py", line 1842, in generate
        result = self._sample(
    File "/usr/local/pyenv/versions/3.10.13/envs/x/lib/python3.10/site-packages/transformers-4.41.1-py3.10.egg/transformers/generation/utils.py", line 2568, in _sample
        next_tokens = next_tokens * unfinished_sequences + \
    RuntimeError: t == c10::DeviceType::PrivateUse1 INTERNAL ASSERT FAILED at "third_party/op-plugin/op_plugin/ops/base_ops/opapi/MulKernelNpuOpApi.cpp":26, please report a bug to PyTorch. DeviceType must be NPU. Actual DeviceType is: cpu
    [ERROR] 2024-05-29-17:04:48 (PID:70209, Device:0, RankID:-1) ERR00001 PTA invalid parameter

A：此类报错通常为部分 Tensor 未放到 NPU 上，请确保报错中算子所涉及的操作数均在 NPU 上。如上面的报错中，MulKernelNpuOpApi 算子为乘法算子，应确保 next_tokens 和 unfinished_sequences 均已放在 NPU 上。


.. **Q：单卡 NPU 情况下，使用 DeepSpeed 训练模型，报错 AttributeError :'GemmaForCausalLM'obiect has no attribute"save checkpoint"，此处 GemmaForCausalLM 还可能为其他模型，详细报错如下图**

**Q：单卡 NPU 情况下，使用 DeepSpeed 训练模型，报错 AttributeError :'GemmaForCausalLM'obiect has no attribute"save checkpoint"，此处 GemmaForCausalLM 还可能为其他模型**

.. .. figure:: ./images/lf-bugfix.png
..   :align: center

A：此问题一般为使用 ``python src/train.py`` 启动训练脚本或使用 ``llamafactory-cli train`` 的同时设置环境变量 ``FORCE_TORCHRUN`` 为 false 或 0 时出现。
由于 DeepSpeed 只对分布式 launcher 启动的程序中的模型用 ``DeepSpeedEngine`` 包装，包装后才有 ``save_checkpoint`` 等方法。
因此使用 ``torchrun`` 启动训练即可解决问题，即:

.. code-block:: shell

    torchrun --nproc_per_node $NPROC_PER_NODE \
            --nnodes $NNODES \
            --node_rank $RANK \
            --master_addr $MASTER_ADDR \
            --master_port $MASTER_PORT \
            src/train.py 

同时使用 ``llamafactory-cli train`` 和 DeepSpeed 时，LLaMA-Factory 将自动设置  ``FORCE_TORCHRUN`` 为 1，启动分布式训练。如果您的代码中没有这个功能，请更新 LLaMA-Factory 为最新代码。

关联 issue 及 PR：

- https://github.com/hiyouga/LLaMA-Factory/issues/4077
- https://github.com/hiyouga/LLaMA-Factory/pull/4082



问题反馈
----------

如果您遇到任何问题，欢迎在 `官方社区 <https://github.com/hiyouga/LLaMA-Factory/issues/>`_ 提 issue，或在 `LLAMA-Factory × 昇腾交流群 <https://github.com/hiyouga/LLaMA-Factory/blob/main/assets/wechat_npu.jpg>`_ 内提问，我们将第一时间进行响应。

*持续更新中 ...*

