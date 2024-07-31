安装指南
==============

本教程面向使用 Diffusers & 昇腾开发者，帮助完成昇腾环境下 Diffusers 的安装。

昇腾环境安装
------------

请根据已有昇腾产品型号及CPU架构等按照 :doc:`快速安装昇腾环境指引 <../ascend/quick_install>` 进行昇腾环境安装，或直接获取对应产品的昇腾环境镜像 `cosdt/cann <https://hub.docker.com/r/cosdt/cann/tags>`_ 。

.. warning::
  CANN 最低版本为 8.0.rc1，安装 CANN 时，请同时安装 Kernel 算子包。

Diffusers 安装
------------------

Python 环境创建
------------------

.. code-block:: shell
    :linenos:
  
    # 创建名为 diffusers 的 python 3.10 的虚拟环境
    conda create -y -n diffusers python=3.10
    # 激活虚拟环境
    conda activate diffusers


pip 安装
------------------

通过以下指令安装 Diffusers 及 torch-npu：

.. code-block:: shell
    :linenos:

    pip install diffusers torch==2.2.0 torch-npu==2.2.0 torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple


安装校验
------------------

执行以下代码，若无任何报错，仅打印模型下载过程，即说明安装成功：

.. code-block:: python
    :linenos:

    from diffusers import DiffusionPipeline
    import torch

    pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipeline.to("npu")
