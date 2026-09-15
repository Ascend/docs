微调预训练模型
==================

.. note::

    阅读本篇前，请确保已按照 :doc:`安装指南 <./install>` 准备好昇腾环境及transformers！

大模型微调本质是利用特定领域的数据集对已预训练的大模型进行进一步训练的过程。它旨在优化模型在特定任务上的性能，使模型能够更好地适应和完成特定领域的任务。
本文在使用transformers库选定相关数据集和预训练模型的基础上，通过超参数调优完成对模型的微调。

前置准备
-----------------

安装必要库
<<<<<<<<<<<<<<<

.. code-block:: shell
    :linenos:

    pip install transformers datasets evaluate accelerate scikit-learn

加载数据集
<<<<<<<<<<<<<<<<<<<

模型训练需要使用数据集，这里使用 `Yelp Reviews dataset <https://huggingface.co/datasets/Yelp/yelp_review_full>`_ ：

.. code-block:: python
    :linenos:

    from datasets import load_dataset

    # load_dataset 会自动下载数据集并将其保存到本地路径中
    dataset = load_dataset("yelp_review_full")
    #输出数据集的第100条数据
    dataset["train"][100]

输出如下:

.. code-block:: shell

    {'label': 0, 'text': 'My expectations for McDonalds are t rarely high. But for one to still fail so spectacularly...that takes something special!\\n
    The cashier took my friends\'s order, then promptly ignored me. I had to force myself in front of a cashier who opened his register to wait on the 
    person BEHIND me. I waited over five minutes for a gigantic order that included precisely one kid\'s meal. After watching two people who ordered after 
    me be handed their food, I asked where mine was. The manager started yelling at the cashiers for \\"serving off their orders\\" when they didn\'t have 
    their food. But neither cashier was anywhere near those controls, and the manager was the one serving food to customers and clearing the boards.\\nThe 
    manager was rude when giving me my order. She didn\'t make sure that I had everything ON MY RECEIPT, and never even had the decency to apologize that 
    I felt I was getting poor service.\\nI\'ve eaten at various McDonalds restaurants for over 30 years. I\'ve worked at more than one location. I expect 
    bad days, bad moods, and the occasional mistake. But I have yet to have a decent experience at this store. It will remain a place I avoid unless someone 
    in my party needs to avoid illness from low blood sugar. Perhaps I should go back to the racially biased service of Steak n Shake instead!'}


预处理数据集
<<<<<<<<<<<<<<<<<

预处理数据集需要使用AutoTokenizer，它用来自动获取与模型匹配的分词器，分词器根据规则将文本拆分为标记，并转换为张量作为模型输入，
下面用到了Meta-Llama-3-8B-Instruct模型，下载模型请转至 `模型获取 <./modeldownload.html>`_，以下是一个示例：

.. code-block:: python
    :linenos:

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    #使用分词器处理文本
    encoded_input = tokenizer("Do not meddle in the affairs of wizards, for they are subtle and quick to anger.")
    print(encoded_input)

输出如下:

.. code-block:: shell

    {'input_ids': [128000, 5519, 539, 1812, 91485, 304, 279, 22747, 315, 89263, 11, 369, 814, 527, 27545, 323, 4062, 311, 19788, 13],
     'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

接着使用dataset.map方法对数据集进行预处理：

.. code-block:: python
    :linenos:

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

初次进行预处理需要一定时间，内容如下：

.. code-block:: shell
    :linenos:

    Asking to pad to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no padding.
    Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
    Map: 100%|████████████████████████████████████████████████████████████████████████| 650000/650000 [03:27<00:00, 3139.47 examples/s]
    Map: 100%|██████████████████████████████████████████████████████████████████████████| 50000/50000 [00:15<00:00, 3156.92 examples/s]

训练全部的数据集会耗费更长的时间，通常将其划分为较小的训练集和验证集，以提高训练速度：

.. code-block:: python
    :linenos:

    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
    
    # 下面是加载全训练集和验证集
    # full_train_dataset = tokenized_datasets["train"]
    # full_eval_dataset = tokenized_datasets["test"]

训练
------------

加载模型
<<<<<<<<<

使用AutoModelForCausalLM将自动加载模型：

.. code-block:: python
    :linenos:

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

超参数调优
<<<<<<<<<<<<<<<<<<<<<

超参数调优用于激活不同训练选项的标志，它定义了关于模型的更高层次的概念，例如模型复杂程度或学习能力，下边使用TrainingArguments类来加载：

.. code-block:: python
    :linenos:

    from transformers import TrainingArguments

    training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")

模型评估
<<<<<<<<<<<<<

模型评估用于衡量模型在给定数据集上的表现，包括准确率，完全匹配速率，平均并交集点等，下面是使用方式：

.. code-block:: python
    :linenos:

    import 
    import sklearn
    import evaluate

    metric = evaluate.load("accuracy")

    #计算预测的准确性,并将预测传递给compute
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)


Trainer
<<<<<<<

使用已加载的模型、训练参数、训练和测试数据集以及评估函数创建一个Trainer对象，并调用trainer.train()来微调模型：

.. code-block:: python
    :linenos:
    
    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()


预训练全流程
-------------------

.. code-block:: python
    :linenos:

    import torch
    import torch_npu
    import numpy as np
    import sklearn
    import evaluate
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from datasets import load_dataset

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    device = "npu:0" if torch.npu.is_available() else "cpu"
    
    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).to(device)

    dataset = load_dataset("yelp_review_full")

    #分词函数
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

    # 加载评估指标
    metric = evaluate.load("accuracy")

    # 定义评估指标的计算函数
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()


训练完成后得到以下结果：

.. code-block:: shell
    :linenos:

    |█████████████████████████████████| [375/375 06:21, Epoch 3/3]

    =====  =============  ===============  ======
    Epoch  Training Loss  Validation Loss  Accuracy
    =====  =============  ===============  ======
    1	    No log	    1.155628	0.499000
    2	    No log	    0.994618	0.574000
    3	    No log	    1.026123	0.590000
    =====  =============  ===============  ======

    TrainOutput(global_step=375, training_loss=1.0557311197916666, metrics={'train_runtime': 384.55, 'train_samples_per_second': 7.801, 
    'train_steps_per_second': 0.975, 'total_flos': 789354427392000.0, 'train_loss': 1.0557311197916666, 'epoch': 3.0})
