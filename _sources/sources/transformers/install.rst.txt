安装指南
===========

本文将介绍如何在昇腾环境下使用transfomers，帮助开发者完成transformers的安装。

.. note:: 

    请确保环境安装了对应的固件和驱动，详情请参考 `快速安装昇腾环境 <../ascend/quick_install.html>`_。

创建虚拟环境
--------------------

首先需要安装并激活python环境：

.. code-block:: shell

    conda create -n your_env_name python=3.10
    conda activate your_env_name

同时安装依赖库：

.. code-block:: shell

    # install torch
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==2.2.0

    # install torch-npu
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple torch-npu==2.2.0

安装transformers
----------------------

直接使用pip命令进行安装：

.. code-block:: shell

    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers

验证安装
--------------------

.. code-block:: python 

    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    import torch
    import torch_npu

    # 检查 NPU 是否可用
    if torch.npu.is_available():
        device = torch.device("npu:0")
        print("NPU is available. Using NPU.")
    else:
        device = torch.device("cpu")
        print("NPU is not available. Using CPU.")

    model_id = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)

    model.to(device)

    nlp_pipeline = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.npu.is_available() else -1
    )

    #分析句子情感并输出
    result = nlp_pipeline("This is a test sentence.")
    print(result)


如果成功运行并输出下面内容，则安装成功：

.. code-block:: shell 

    NPU is available. Using NPU.
    Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    [{'label': 'POSITIVE', 'score': 0.9998704791069031}]

卸载transformers
---------------------

.. code-block:: shell 

    pip uninstall transformers


