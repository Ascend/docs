模型推理 
==================

推理以transformers的pipeline为中心进行介绍

pipeline 使用
----------------------

以使用pipeline()进行自动语音识别或语音转文本为例

创建pipeline, 并输入音频文件

.. code-block:: python
    :linenos:

    from transformers import pipeline

    transcriber = pipeline(task="automatic-speech-recognition")
    transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")

获得以下结果

.. code-block:: shell
    :linenos:

    {'text': 'I HAVE A DREAM BUT ONE DAY THIS NATION WILL RISE UP LIVE UP THE TRUE MEANING OF ITS TREES'}


文本推理
-------------------

文本推理可以根据标签对文本进行分类

>>> from transformers import pipeline
    classifier = pipeline(model="facebook/bart-large-mnli")
    classifier(
        "I have a problem with my iphone that needs to be resolved asap!!",
        candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"],
    )
    #以下为输出示例
    #{'sequence': 'I have a problem with my iphone that needs to be resolved asap!!', 'labels': ['urgent', 'phone', 'computer', 'not urgent', 'tablet'], 'scores': [0.504, 0.479, 0.013, 0.003, 0.002]}

多模态
----------------------

VQA使用图像和关于该图像的问题进行提问, 图像可以是URL或图像的本地路径

.. code-block:: shell
    :linenos:

    from transformers import pipeline
    vqa = pipeline(model="impira/layoutlm-document-qa")
    output = vqa(
        image="https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png",
        question="What is the invoice number?",
    )
    output[0]["score"] = round(output[0]["score"], 3)

    #以下为输出示例
    #[{'score': 0.425, 'answer': 'us-001', 'start': 16, 'end': 16}]

大模型上使用pipeline
------------------------------

在大模型上运行pipeline需要用到accelerate

.. code-block:: shell
    :linenos:

    pip install accelerate

加载模型并获取结果

.. code-block:: shell
    :linenos:

    import torch
    import torch_npu
    from transformers import pipeline

    pipe = pipeline(model="facebook/opt-1.3b", torch_dtype=torch.bfloat16, device_map="npu:0")
    output = pipe("tell a cold joke!", do_sample=True, top_p=0.95)
    print(output)

    #以下为输出示例
    #[{'generated_text': "tell a cold joke!\nI'll give you one. Are you cold, or are you warm"}]


