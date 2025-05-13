快速开始
============

.. note::
    阅读本篇前，请确保已按照 :doc:`安装指南 <./install>` 准备好昇腾环境及 Accelerate !
    
本教程以一个简单的 NLP 模型为例，讲述如何使用 Accelerate 在昇腾 NPU 上进行模型的训练。

前置准备
------------

本篇将使用到 HuggingFace 其他工具链及 scikit-learn 库，请使用以下指令安装：

.. code-block::

  pip install datasets evaluate transformers scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple

本篇样例代码为 Accelrate 官方样例，需提前进行下载

.. code-block::

  git clone https://github.com/huggingface/accelerate.git

模型训练
------------

.. code-block::
  :linenos:

  # 替换HF域名，方便国内用户进行数据及模型的下载
  export HF_ENDPOINT=https://hf-mirror.com
  # 进入项目目录     
  cd accelerate/examples
  # 模型训练
  python nlp_example.py

出现如下日志代表训练成功：

::

    Downloading builder script: 5.75kB [00:01, 3.69kB/s]                                                                                                  
    tokenizer_config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████| 49.0/49.0 [00:00<00:00, 237kB/s]
    config.json: 570B [00:00, 2.23MB/s]                                                                                                                   
    vocab.txt: 79.5kB [00:12, 3.45kB/s]Error while downloading from https://hf-mirror.com/bert-base-cased/resolve/main/vocab.txt: HTTPSConnectionPool(host='hf-mirror.com', port=443): Read timed out.
    Trying to resume download...
    vocab.txt: 213kB [00:07, 15.5kB/s]]
    vocab.txt: 91.4kB [00:32, 2.81kB/s]
    tokenizer.json: 436kB [00:19, 22.8kB/s] 
    Downloading readme: 35.3kB [00:01, 26.4kB/s]
    Downloading data: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 649k/649k [00:02<00:00, 288kB/s]
    Downloading data: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 75.7k/75.7k [00:00<00:00, 77.8kB/s]
    Downloading data: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 308k/308k [00:01<00:00, 204kB/s]
    Generating train split: 100%|███████████████████████████████████████████████████████████████████████████| 3668/3668 [00:00<00:00, 27701.23 examples/s]
    Generating validation split: 100%|████████████████████████████████████████████████████████████████████████| 408/408 [00:00<00:00, 73426.42 examples/s]
    Generating test split: 100%|███████████████████████████████████████████████████████████████████████████| 1725/1725 [00:00<00:00, 246370.91 examples/s]
    Map: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 3668/3668 [00:01<00:00, 3378.05 examples/s]
    Map: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 408/408 [00:00<00:00, 3553.72 examples/s]
    Map: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1725/1725 [00:00<00:00, 5109.03 examples/s]
    model.safetensors: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 436M/436M [02:42<00:00, 2.68MB/s]
    Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-cased and are newly initialized: ['classifier.bias', 'classifier.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
      - Avoid using `tokenizers` before the fork if possible
      - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    You're using a BertTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
    epoch 0: {'accuracy': 0.8014705882352942, 'f1': 0.8439306358381503}
    epoch 1: {'accuracy': 0.8578431372549019, 'f1': 0.8975265017667845}
    epoch 2: {'accuracy': 0.8700980392156863, 'f1': 0.9087779690189329}
