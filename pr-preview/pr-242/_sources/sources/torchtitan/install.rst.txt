安装指南
==============

本教程面向使用 TorchTitan & 昇腾的开发者，帮助完成昇腾环境下 TorchTitan 的安装。

昇腾环境安装
------------

请根据已有昇腾产品型号及CPU架构等按照 :doc:`快速安装昇腾环境指引 <../ascend/quick_install>` 进行昇腾环境安装。

.. warning::
  CANN 最低版本为 8.0.rc1，安装 CANN 时，请同时安装 Kernel 算子包。

Python 环境创建
----------------------

.. code-block:: shell
    :linenos:
  
    # 创建 python 3.10 的虚拟环境
    conda create -y -n torchtitan python=3.10
    # 激活虚拟环境
    conda activate torchtitan


TorchTitan 安装
----------------------

使用以下指令安装 TorchTitan:

.. code-block:: shell
    :linenos:

    git clone https://github.com/pytorch/torchtitan
    cd torchtitan
    pip install -r requirements.txt


torch-npu 安装
----------------------

按照 :doc:`torch-npu 安装指引 <../pytorch/install>` 本项目推荐安装 2.6.0 及以上版本 torch 和 torch-npu，并使用以下指令进行校验：

.. code-block:: shell
    :linenos:

    import torch
    # import torch_npu 较高版本torch无需显式import torch_npu

    x = torch.randn(2, 2).npu()
    y = torch.randn(2, 2).npu()
    z = x.mm(y)

    print(z)

程序能够成功打印矩阵Z的值即为安装成功。

下载Tokenizer
----------------------

TorchTitan 目前支持直接训练的模型主要包括有 Llama 3模型，Llama 4模型等一系列模型。若要训练这些模型，需要下载一个 tokenizer.model 文件。以下以llama3_8b模型为例进行简介。请按照 `Meta - Llama <https://huggingface.co/meta-llama/Llama-3.1-8B>`_  官方代码仓库中的说明操作，以确保能够获取 Llama 模型的权重。
确认获得访问权限后，可以运行以下命令将 Llama 3.1 的分词器（tokenizer）下载到本地机器。

.. code-block:: python
    :linenos:

    # Get your HF token from https://huggingface.co/settings/tokens

    # Llama 3.1 tokenizer.model
    python scripts/download_tokenizer.py --repo_id meta-llama/Meta-Llama-3.1-8B --tokenizer_path "original" --hf_token=...

在本地机器运行（8卡 NPU 环境）：

.. code-block:: shell

  CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" ./run_train.sh
