快速开始
==================

.. note::
    阅读本篇前，请确保已按照 :doc:`安装教程 <./install>` 准备好昇腾环境及 OpenCompass ！

本文档帮助昇腾开发者快速使用 OpenCompass × 昇腾 进行训练和推理。

概览
---------------------

在 OpenCompass 中评估一个模型通常包括以下几个阶段：配置 -> 推理 -> 评估 -> 可视化。

配置：这是整个工作流的起点。您需要配置整个评估过程，选择要评估的模型和数据集。此外，还可以选择评估策略、计算后端等，并定义显示结果的方式。

推理与评估：在这个阶段，OpenCompass 将会开始对模型和数据集进行并行推理和评估。推理阶段主要是让模型从数据集产生输出，而评估阶段则是衡量这些输出与标准答案的匹配程度。这两个过程会被拆分为多个同时运行的“任务”以提高效率，但请注意，如果计算资源有限，这种策略可能会使评测变得更慢。如果需要了解该问题及解决方案，可以参考
`FAQ: 效率： <https://opencompass.readthedocs.io/en/latest/get_started/faq.html#efficiency>`_

可视化：评估完成后，OpenCompass 将结果整理成易读的表格，并将其保存为 CSV 和 TXT 文件。你也可以激活飞书状态上报功能，此后可以在飞书客户端中及时获得评测状态报告。

接下来，我们将展示 OpenCompass 的基础用法，展示基座模型 ``InternLM2-1.8B`` 和对话模型 ``InternLM2-Chat-1.8B``、``Qwen2-1.5B-Instruct`` 在 GSM8K 和 MATH 下采样数据集上的评估。它们的配置文件可以在 ``configs/eval_chat_demo.py`` 和 ``configs/eval_base_demo.py`` 中找到。

在运行此实验之前，请确保您已在本地安装了 ``opencompass`` && ``torch-npu``。

本文参考：
`OpenCompass官方文档 <https://opencompass.readthedocs.io/zh-en>`_


配置评估任务
~~~~~~~~~~~~~~~

.. note::

    在 OpenCompass 中，每个评估任务由待评估的模型和数据集组成。评估的入口点是 run.py。用户可以通过命令行或配置文件选择要测试的模型和数据集。

对于对话模型：

.. code-block:: shell
    :linenos:

    python run.py \
    --models hf_internlm2_chat_1_8b hf_qwen2_1_5b_instruct \
    --datasets demo_gsm8k_chat_gen demo_math_chat_gen \
    --debug

对于基座模型：

.. code-block:: shell
    :linenos:

    python run.py \
    --models hf_internlm2_1_8b hf_qwen2_1_5b \
    --datasets demo_gsm8k_base_gen demo_math_base_gen \
    --debug

.. list-table:: opencompass run.py 参数说明
   :widths: 15 30 25
   :header-rows: 1

   * - 命令行参数
     - 描述
     - 样例数值
   * - --hf-type
     - HuggingFace 模型类型，可选值为 chat 或 base
     - chat
   * - --hf-path
     - HuggingFace 模型路径
     - internlm/internlm2-chat-1_8b
   * - --model-kwargs
     - 构建模型的参数
     - device_map=’auto’
   * - --tokenizer-path
     - HuggingFace tokenizer 路径（如果与模型路径相同，可以省略）
     - internlm/internlm2-chat-1_8b
   * - --tokenizer-kwargs
     - 构建 tokenizer 的参数
     - padding_side=’left’ truncation=’left’ trust_remote_code=True
   * - --generation-kwargs
     - 生成的参数
     - do_sample=True top_k=50 top_p=0.95
   * - --max-seq-len
     - 模型可以接受的最大序列长度
     - 2048
   * - --max-out-len
     - 生成的最大 token 数
     - 100
   * - --min-out-len
     - 生成的最小 token 数
     - 1
   * - --batch-size
     - 批量大小
     - 64
   * - --hf-num-gpus
     - 运行一个模型实例所需的 GPU 数量
     - 1
   * - --stop-words
     - 停用词列表
     - ‘<|im_end|>’ ‘<|im_start|>’
   * - --pad-token-id
     - 填充 token 的 ID
     - 0
   * - --peft-path
     - (例如) LoRA 模型的路径
     - internlm/internlm2-chat-1_8b
   * - --peft-kwargs
     - (例如) 构建 LoRA 模型的参数
     - trust_remote_code=True



启动评估
~~~~~~~~~~~~~~~

由于 OpenCompass 默认并行启动评估过程，我们可以在第一次运行时以 ``--debug`` 模式启动评估，并检查是否存在问题。包括在前述的所有文档中，我们都使用了 ``--debug`` 开关。在 ``--debug`` 模式下，任务将按顺序执行，并实时打印输出。

.. code-block:: shell
    :linenos:

    # train on multi-npu
    python run.py configs/eval_chat_demo.py -w outputs/demo --debug


对话默写 ‘internlm/internlm2-chat-1_8b’ 和 ‘Qwen/Qwen2-1.5B-Instruct’ 将在首次运行期间从 HuggingFace 自动下载。 如果一切正常，您应该看到屏幕上显示 “Starting inference process”，且进度条开始前进：

.. code-block:: shell
    :linenos:

    # train on multi-npu
    [2023-07-12 18:23:55,076] [opencompass.openicl.icl_inferencer.icl_gen_inferencer] [INFO] Starting inference process...

然后，您可以按 Ctrl+C 中断程序，并以正常模式运行以下命令：

.. code-block:: shell
    :linenos:

    # train on multi-npu
    python run.py configs/eval_chat_demo.py -w outputs/demo

在正常模式下，评估任务将在后台并行执行，其输出将被重定向到输出目录 ``outputs/demo/{TIMESTAMP}``。前端的进度条只指示已完成任务的数量，而不考虑其成功或失败。任何后端任务失败都只会在终端触发警告消息。

可视化评估结果
---------------------

评估完成后，评估结果表格将打印如下：

.. code-block:: shell
    :linenos:

    dataset     version    metric    mode      qwen2-1.5b-instruct-hf    internlm2-chat-1.8b-hf
    ----------  ---------  --------  ------  ------------------------  ------------------------
    demo_gsm8k  1d7fe4     accuracy  gen                        56.25                     32.81
    demo_math   393424     accuracy  gen                        18.75                     14.06


所有运行输出将定向到 ``outputs/demo/`` 目录，结构如下：

.. code-block:: shell
    :linenos:

    outputs/default/
    ├── 20200220_120000
    ├── 20230220_183030     # 每个实验一个文件夹
    │   ├── configs         # 用于记录的已转储的配置文件。如果在同一个实验文件夹中重新运行了不同的实验，可能会保留多个配置
    │   ├── logs            # 推理和评估阶段的日志文件
    │   │   ├── eval
    │   │   └── infer
    │   ├── predictions   # 每个任务的推理结果
    │   ├── results       # 每个任务的评估结果
    │   └── summary       # 单个实验的汇总评估结果
    ├── ...
