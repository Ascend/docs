快速使用
===============================


写在前面: kernels 是什么
-------------------------------

kernels 允许 transformers 库（理论上所有 Python 模型库都可以）直接从 `HuggingFace-Hub <https://huggingface.co/>`_ 动态加载计算内核。与传统的直接使用 Python 计算内核的区别在于其具备以下特性：

- 易移植：从 PYTHONPATH 之外的路径加载内核。你不必再针对每个依赖 Transformers 的上层库中做MonkeyPatch。
- 版本的扩展性：你可以为同一 Python 模块加载某一内核的多个版本。
- 版本的兼容性：kernels 为加载 HuggingFace-Hub 中的计算内核制定了一套标准文件路径命名。该命名使用torch, cuda/cann, ABIs, linux name 和 os作为关键字。这使得在向 HuggingFace-Hub 贡献时，必须保证计算内核在特定关键字排列组合下对应版本的兼容性。

transformers 在 v4.54.0 的 release 中首次介绍了 kernels 的集成，并将后续计算加速内核的支持都放在了这里。如 GPT-OSS 的flash-attention-3 就是通过 kernels 支持的。


提供一个简单的样例，验证 kernels 的安装是否成功
------------------------------------------------------

.. code-block:: python

    import torch
    from kernels import get_kernel

    # Download optimized kernels from the Hugging Face hub
    activation = get_kernel("kernels-community/activation")

    # Create a random tensor
    x = torch.randn((10, 10), dtype=torch.float16, device="cuda")

    # Run the kernel
    y = torch.empty_like(x)
    activation.gelu_fast(y, x)
    print(y)


transformers + kernels 的使用
-------------------------------

kernels 同 transformers 都属于 huggingface 生态的一部分，因此 transformers 库率先内置支持了 kernels。所以这里我们提供一个 transformers + kernels 的使用示例，以便更好地理解 kernels 的使用优势。

kernels 支持 remote 和 local 两种内核加载方式。remote 方式是指运行时直接从 HuggingFace-Hub 下载内核并加载，而 local 方式是指事先从 HuggingFace-Hub 下载，并将其搬运至你指定的本地路径后加载内核，适用于网络受限或需要离线部署的场景。下面的示例将展示如何使用这两种方式加载内核。


Remote 加载内核示例
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

示例提供了计时统计，你可以通过注释 use_kernels 入参来对比使用 kernels 前后的性能差异。

.. code-block:: python

    import time
    import logging
    from transformers import AutoModelForCausalLM, AutoTokenizer


    # Set the level to `DEBUG` to see which kernels are being called.
    logging.basicConfig(level=logging.DEBUG)

    model_name = "Qwen/Qwen3-0.6B"

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        use_kernels=True,
    )

    # Prepare the model input
    prompt = "What is the result of 100 + 100?"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Warm_up
    for _ in range(2):
        generated_ids = model.generate(**model_inputs, max_new_tokens=32768)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

    # Print Runtime
    for _ in range(5):
        start_time = time.time()
        generated_ids = model.generate(**model_inputs, max_new_tokens=32768)
        print("runtime: ", time.time() - start_time)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        print("content:", content)


Local 加载内核示例
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

示例提供了计时统计，你可以通过注释 kernel_config 入参来对比使用 kernels 前后的性能差异。

.. code-block:: python

    import time
    import logging
    from transformers import AutoModelForCausalLM, AutoTokenizer, KernelConfig


    # Set the level to `DEBUG` to see which kernels are being called.
    logging.basicConfig(level=logging.DEBUG)

    model_name = "/root/Qwen3"

    kernel_mapping = {
        "RMSNorm":
            "/kernels-ext-npu/rmsnorm:rmsnorm",
    }

    kernel_config = KernelConfig(kernel_mapping, use_local_kernel=True)

    # Load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        kernel_config=kernel_config
    )

    # Prepare the model input
    prompt = "What is the result of 100 + 100?"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Warm_up
    for _ in range(2):
        generated_ids = model.generate(**model_inputs, max_new_tokens=32768)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

    # Print Runtime
    for _ in range(5):
        start_time = time.time()
        generated_ids = model.generate(**model_inputs, max_new_tokens=32768)
        print("runtime: ", time.time() - start_time)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        print("content:", content)

.. warning::
    使用 local 本地加载时，transformers 需要从 main 源码编译安装，因为此部分代码尚未发布在 release 版本。
