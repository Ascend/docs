快速使用文档
===============================


写在前面: kernels是什么
-------------------------------

 kernels 允许 transformers 库（理论上所有 Python 模型库都可以）直接从 `HuggingFace-Hub <https://huggingface.co/>`_ 动态加载计算内核。HuggingFace-Hub 加载内核与传统的直接使用 Python 计算内核的区别在于其具备以下特性：

- 易移植：从 PYTHONPATH 之外的路径加载内核。你不必再针对每个依赖 Transformers 的上层库中做MonkeyPatch。
- 版本的扩展性：你可以为同一 Python 模块加载某一内核的多个版本。
- 版本的兼容性：kernels 为加载 HuggingFace-Hub 中的计算内核制定了一套标准文件路径命名。该命名使用torch, cuda/cann, ABIs, linux name 和 os作为关键字。这使得在向 HuggingFace-Hub 贡献时，必须保证计算内核在特定关键字排列组合下对应版本的兼容性。

transformers 在 v4.54.0 的 release 中首次介绍了 kernels 的集成，并将后续计算加速内核的支持都放在了这里。如 GPT-OSS 的flash-attention-3 就是通过 kernels 支持的。 
