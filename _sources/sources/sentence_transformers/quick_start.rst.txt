快速开始
===============

.. note::

    阅读本篇前，请确保已按照 :doc:`安装教程 <./install>` 准备好昇腾环境及 sentence-transformers ！

本教程以 `all-MiniLM-L6-v2` 模型为例，讲述如何使用 sentence-transformers 在昇腾 NPU 上实现文本数据的 Embedding。

前置准备
---------------

本篇样例代码为 sentence-transformers 的官方样例，需提前进行下载：

.. code-block::

    git clone https://github.com/UKPLab/sentence-transformers.git

使用模型
---------------

进入 sentence-transformers 项目目录，依次执行如下命令：

.. code-block::

    cd examples/applications/computing-embeddings
    python computing_embeddings.py

出现如下日志则代表执行成功：

::
    
    2024-10-15 08:11:36 - Use pytorch device_name: npu
    2024-10-15 08:11:36 - Load pretrained SentenceTransformer: all-MiniLM-L6-v2
    [W compiler_depend.ts:623] Warning: expandable_segments currently defaults to false. You can enable this feature by `export PYTORCH_NPU_ALLOC_CONF = expandable_segments:True`. (function operator())
    Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.61it/s]
    Sentence: This framework generates embeddings for each input sentence
    Embedding: [-0.01375547 -0.04301599 -0.01562478 ...  0.10029524  0.12379668 -0.04230832]

    Sentence: Sentences are passed as a list of string.
    Embedding: [ 0.05640831  0.05488579  0.03137118 ...  0.06652435  0.08493122 -0.03337045]

    Sentence: The quick brown fox jumps over the lazy dog.
    Embedding: [0.04393559 0.05903088 0.04824848 ... 0.05215353 0.05615513 0.10205095]

可以看到该模型成功生成了这些句子对应的 Embedding 向量。
