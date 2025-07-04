快速开始
==================

.. note::

    阅读本篇前，请确保已按照 :doc:`安装教程 <./install>` 准备好昇腾环境及 torchchat ！
    
    本篇教程将介绍如何使用 torchchat 进行快速开发，帮助您快速上手 torchchat。

查看帮助
-----------------
torchchat 提供了帮助命令，帮助用户快速了解 torchchat 的使用方法。

.. code-block:: shell
    :linenos:

    # 查看帮助
    python torchchat.py --help

输出列表:

.. code-block:: shell
    :linenos:

    usage: torchchat [-h] {chat,browser,generate,export,eval,download,list,remove,where,server} ...

    positional arguments:
      {chat,browser,generate,export,eval,download,list,remove,where,server}
                            The specific command to run
        chat                Chat interactively with a model via the CLI
        generate            Generate responses from a model given a prompt
        browser             Chat interactively with a model in a locally hosted browser
        export              Export a model artifact to AOT Inductor or ExecuTorch
        download            Download model artifacts
        list                List all supported models
        remove              Remove downloaded model artifacts
        where               Return directory containing downloaded model artifacts
        server              [WIP] Starts a locally hosted REST server for model interaction
        eval                Evaluate a model via lm-eval

    options:
      -h, --help            show this help message and exit

主要命令:

Inference (chat, generate, browser)
    - chat: 交互式聊天
    - generate: 生成响应
    - browser: 在本地浏览器中交互式聊天

Inventory Management (download, list, remove, where)
    - download: 下载模型
    - list: 列出所有支持的模型
    - remove: 删除下载的模型
    - where: 返回包含下载的模型的目录

下载模型
-----------------
torchchat 大多数模型使用 Hugging Face 作为分发渠道，因此需要创建一个 Hugging Face 帐户。
按照 `此处 <https://huggingface.co/docs/hub/en/security-tokens>`_ 的说明使用write角色创建 Hugging Face 用户访问令牌。

登录 huggingface:

.. code-block:: shell
    :linenos:

    # 登录 huggingface
    huggingface-cli login

查看支持模型列表:

.. code-block:: shell
    :linenos:

    # 查看支持的模型列表
    python torchchat.py list

下载模型:

.. code-block:: shell
    :linenos:

    # 下载模型
    python torchchat.py download <model_name>
    
    # 例如下载 LLaMA 模型
    python torchchat.py download llama3.1

模型推理
-----------------
torchchat 支持多种推理方式，用户可以根据自己的需求选择合适的推理方式。

Chat
以互动方式与模型进行聊天:

.. code-block:: shell
    :linenos:

    # 交互式聊天
    python torchchat.py chat llama3.1

Generate
根据输入提示生成文本:

.. code-block:: shell
    :linenos:

    # 生成响应
    python torchchat.py generate llama3.1 --prompt "write me a story about a boy and his bear"

生成结果示例:

.. code-block:: shell
    :linenos:

    Using device=npu Ascend910B3
    Loading model...
    Time to load model: 4.42 seconds
    -----------------------------------------------------------
    write me a story about a boy and his bear friend
    Once upon a time, in a dense forest, there lived a young boy named Timmy. Timmy was a curious and adventurous boy who loved exploring the woods behind his village. One day, while wandering deeper into the forest than he had ever gone before, Timmy stumbled upon a magnificent brown bear. The bear was enormous, with a thick coat of fur and piercing yellow eyes. At first, Timmy was frightened, but to his surprise, the bear didn't seem to be threatening him. Instead, the bear gently approached Timmy and began to sniff him.

    As the days passed, Timmy and the bear, whom he named Boris, became inseparable friends. Boris was unlike any bear Timmy had ever seen before. He was incredibly intelligent and could understand human language. Boris would often sit by Timmy's side as he read books or helped with his chores. The villagers were initially wary of Boris, but as they saw how kind and gentle he was, they grew
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                
    Generated 199 tokens                 
    Time for inference 1: 13.3118 sec total                 
    Time to first token: 0.6189 sec with parallel prefill.                

        Total throughput: 15.0242 tokens/sec, 0.0666 s/token                 
    First token throughput: 1.6157 tokens/sec, 0.6189 s/token                 
    Next token throughput: 15.6781 tokens/sec, 0.0638 s/token                     

    Bandwidth achieved: 241.30 GB/s
    *** This first iteration will include cold start effects for dynamic import, hardware caches. ***

    ========================================


    Warning: Excluding compile in calculations                 
        Average tokens/sec (total): 15.02                 
    Average tokens/sec (first token): 1.62                 
    Average tokens/sec (next tokens): 15.68 
                    
    Memory used: 17.23 GB

如上所示，torchchat 会对输入的文本进行处理，并生成相应的文本，同时输出生成的文本长度、推理时间、带宽等信息，方便用户进行性能分析。
    
    以上是 torchchat 的快速开始教程，更多其他昇腾原生支持功能请参考 :doc:`昇腾开源 <../../index>` 。