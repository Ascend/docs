推理 
==================

.. note::

    阅读本篇前，请确保已按照 :doc:`安装指南 <./install>` 准备好昇腾环境及transformers！

在推理阶段，训练好的模型被用于对图像、语音或文本进行分类，也可以用于语言生成、翻译等。

本文的模型推理以transformers的pipeline为中心进行介绍，pipelines可以自动加载模型和能够进行任务推理的预处理类，使任何模型进行任何语言、计算机视觉、语音以及多模态任务的推理变得非常简单。

pipeline 抽象类
------------------

pipeline 抽象类是所有其他 pipeline 的封装，可以像其他任何 pipeline 一样实例化。

pipeline 参数由 task、tokenizer、model、optional 组成：

- task 将确定返回哪一个 pipeline，比如 text-classification 将会返回 TextClassificationPipeline，image-to-image 将会返回 ImageToImagePipeline。

- tokenizer分词器是用来将输入进行编码，str或者PreTrainedTokenizer，如果未提供将使用model参数，如果model也未提供或者非str,将使用config参数，如果config参数也未提供或者非str，将提供task的默认tokenizer。

- model是模型，str或者PreTrainedModel，一般为有.bin模型文件的目录。

- optional其他参数包括，config、feature_extractor、device、device_map等。


pipeline 使用
----------------------

pipeline适用于音频、计算机视觉、自然语言处理和多模态任务，下面将介绍它在各场景的使用方式。

音频
<<<<<<<<<<<<<

音频识别
>>>>>>>>>>>>

用于提取某些音频中包含的文本，如下创建pipeline，并输入音频文件：

.. code-block:: python
    :linenos:

    from transformers import pipeline

    transcriber = pipeline(task="automatic-speech-recognition")
    transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")

    #以下为输出示例
    {'text': 'I HAVE A DREAM BUT ONE DAY THIS NATION WILL RISE UP LIVE UP THE TRUE MEANING OF ITS TREES'}

文本转音频
>>>>>>>>>>>

根据输入文本和可选的其他条件输入生成音频文件：

.. code-block:: python
    :linenos:

    from transformers import pipeline

    pipe = pipeline(model="suno/bark-small")
    output = pipe("Hey it's HuggingFace on the phone!")

    audio = output["audio"]
    sampling_rate = output["sampling_rate"]

计算机视觉
<<<<<<<<<<<<<<<<<

图像分类
>>>>>>>>>>>>>>

图像分类可以识别图片特征，并给出分类标签和置信度得分：

.. code-block:: python
    :linenos:

    from transformers import pipeline

    classifier = pipeline(model="microsoft/beit-base-patch16-224-pt22k-ft22k")
    classifier("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")

    #以下为输出示例
    [{'score': 0.442, 'label': 'macaw'}, {'score': 0.088, 'label': 'popinjay'}, {'score': 0.075, 'label': 'parrot'}, {'score': 0.073, 'label': 'parodist, lampooner'}, {'score': 0.046, 'label': 'poll, poll_parrot'}]

图像转图像
>>>>>>>>>>>>>

它可以将图像根据信息生成新图像，以下示例通过图像超分辨率模型将低分辨率图像放大并增强其细节，使其看起来更清晰：

.. code-block:: python
    :linenos:

    from PIL import Image
    import requests
    from transformers import pipeline

    upscaler = pipeline("image-to-image", model="caidas/swin2SR-classical-sr-x2-64")
    img = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
    img = img.resize((64, 64))
    upscaled_img = upscaler(img) #超分辨率处理
    print(img.size)          
    print(upscaled_img.size) 

    #以下为输出示例
    (64, 64)    # 输出原图像的尺寸 
    (144, 144)  # 输出处理后图像的尺寸

自然语言处理
<<<<<<<<<<<<<<<<<

文本分类
>>>>>>>>>>>>>>>>>>>

根据标签对文本进行分类:

.. code-block:: shell
    :linenos:

    from transformers import pipeline
    classifier = pipeline(model="meta-llama/Meta-Llama-3-8B-Instruct")
    classifier(
        "I have a problem with my iphone that needs to be resolved asap!!",
        candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"],
    )
    #以下为输出示例
    #{'sequence': 'I have a problem with my iphone that needs to be resolved asap!!', 'labels': ['urgent', 'phone', 'computer', 'not urgent', 'tablet'], 'scores': [0.504, 0.479, 0.013, 0.003, 0.002]}

文本生成
>>>>>>>>>>>>>>>>>

根据文本生成对话响应：

.. code-block:: shell
    :linenos:

    from transformers import pipeline

    generator = pipeline(model="HuggingFaceH4/zephyr-7b-beta")
    # Zephyr-beta is a conversational model, so let's pass it a chat instead of a single string
    generator([{"role": "user", "content": "What is the capital of France? Answer in one word."}], do_sample=False, max_new_tokens=2)

    #以下为输出示例
    [{'generated_text': [{'role': 'user', 'content': 'What is the capital of France? Answer in one word.'}, {'role': 'assistant', 'content': 'Paris'}]}]

多模态
<<<<<<<<<<<<<<

视觉问答
>>>>>>>>>>>>>

VQA使用图像和关于该图像的问题进行提问，图像可以是URL或图像的本地路径:

.. code-block:: shell
    :linenos:

    from transformers import pipeline
    vqa = pipeline(model="meta-llama/Meta-Llama-3-8B-Instruct")
    output = vqa(
        image="https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png",
        question="What is the invoice number?",
    )
    output[0]["score"] = round(output[0]["score"], 3)

    #以下为输出示例
    #[{'score': 0.425, 'answer': 'us-001', 'start': 16, 'end': 16}]

图像转文本
>>>>>>>>>>>>>>>>>>>>

用于预测给定图像的主题：

.. code-block:: shell
    :linenos:

    from transformers import pipeline

    captioner = pipeline(model="ydshieh/vit-gpt2-coco-en")
    captioner("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")

    #以下为输出示例
    [{'generated_text': 'two birds are standing next to each other '}]
