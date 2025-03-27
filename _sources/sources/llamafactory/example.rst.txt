全流程昇腾实践
=====================

开始本篇之前，请阅读 `LLaMA-Factory QuickStart <https://zhuanlan.zhihu.com/p/695287607>`_ 了解 LLaMA-Factory 及其主要功能的用法，
并参考 :doc:`安装指南  <./install>` 及  :doc:`快速开始  <./quick_start>` 完成基本的环境准备、LLaMA-Factory 安装及简单的微调和推理功能。
本篇在此基础上，以 Qwen1.5-7B 模型为例，帮助开发者在昇腾 NPU 上使用 LLaMA-Factory 更多实用特性。

`LLaMA-Factory QuickStart <https://zhuanlan.zhihu.com/p/695287607>`_ 中详解了下列 9 种功能，本教程为在 NPU 上全流程实践示例，
有关功能及参数的详细解析请参考 `LLaMA-Factory QuickStart <https://zhuanlan.zhihu.com/p/695287607>`_ 


1. 原始模型直接推理
2. 自定义数据集构建
3. 基于 LoRA 的 sft 指令微调
4. 动态合并 LoRA 的推理
5. 批量预测和训练效果评估
6. LoRA模型合并导出
7. 一站式 webui board 的使用
8. API Server的启动与调用
9. 大模型主流评测 benchmark

前置准备
--------

安装准备
~~~~~~~~~

请确认已按照 :doc:`安装指南  <./install>` 安装 CANN 和 LLaMA-Factory 并完成安装校验。

配置文件准备
~~~~~~~~~~~~

本示例中用到的参数配置文件与快速开始 :ref:`qwen1_5_lora_sft_ds.yaml <qwen_yaml>` 中一致，可参考快速开始。


原始模型直接推理
-----------------

验证 LLaMA-Factory 在昇腾 NPU 上推理功能是否正常：

.. code-block:: shell
    :linenos:

    ASCEND_RT_VISIBLE_DEVICES=0 llamafactory-cli webchat --model_name_or_path qwen/Qwen1.5-7B \
                --adapter_name_or_path saves/Qwen1.5-7B/lora/sft \
                --template qwen \
                --finetuning_type lora

如下图所示可正常进行对话，即为可正常推理：

.. figure:: ./images/webchat.png
  :align: center

自定义数据集构建
-------------------

本篇用到的数据集为 LLaMA-Factory 自带的 identity 和 alpaca_en_demo，对 identity 数据集进行如下全局替换即可实现定制指令：

- ``{{name}}`` 替换为 ``Ascend-helper``
- ``{{author}}`` 替换为 ``Ascend``

更多自定义数据集的构建请参考 `官方数据集构造指引 <https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README_zh.md>`_ 。

.. _sft:

基于 LoRA 的 sft 指令微调
-------------------------
在 :doc:`快速开始 <./quick_start>` 中，已经尝试过使用 src/train.py 为入口的微调脚本，本篇中均使用 llamafactory-cli 命令启动微调、推理等程序。

.. code-block:: shell
    :linenos:

    ASCEND_RT_VISIBLE_DEVICES=0 llamafactory-cli train <your_path>/qwen1_5_lora_sft_ds.yaml


动态合并 LoRA 的推理
-------------------------

.. code-block:: shell
    :linenos:

    ASCEND_RT_VISIBLE_DEVICES=0 llamafactory-cli chat --model_name_or_path qwen/Qwen1.5-7B \
                --adapter_name_or_path saves/Qwen1.5-7B/lora/sft \
                --template qwen \
                --finetuning_type lora

通过询问大模型是谁检验 sft 指令微调的成果，如下图，大模型回答自己是 Ascend-helper 说明 sft 成功，如失败，可返回 :ref:`sft` 增加训练轮数重新训练。

.. figure:: ./images/sft-chat.gif
  :align: center


批量预测和训练效果评估
------------------------

使用批量预测和评估前，需先安装 jieba、rouge-chinese、nltk 三个库：

.. code-block:: shell
    :linenos:

    pip install jieba,rouge-chinese,nltk -i https://pypi.tuna.tsinghua.edu.cn/simple

然后使用以下指令对微调后的模型在 alpaca_gpt4_zh 和 identity 数据集上进行批量预测和效果评估：

.. code-block:: shell
    :linenos:

    ASCEND_RT_VISIBLE_DEVICES=0 llamafactory-cli train \
                --stage sft \
                --do_predict \
                --model_name_or_path qwen/Qwen1.5-7B \
                --adapter_name_or_path ./saves/Qwen1.5-7B/lora/sft  \
                --dataset alpaca_gpt4_zh,identity \
                --dataset_dir ./data \
                --template qwen \
                --finetuning_type lora \
                --output_dir ./saves/Qwen1.5-7B/lora/predict \
                --overwrite_cache \
                --overwrite_output_dir \
                --cutoff_len 1024 \
                --preprocessing_num_workers 16 \
                --per_device_eval_batch_size 1 \
                --max_samples 20 \
                --predict_with_generate

完成批量预测与评估后，在指定的输出路径下会看到以下文件：

.. code-block:: shell

    all_results.json
    generated_predictions.jsonl
    predict_results.json
    trainer_log.jsonl

``generated_predictions.json`` 中为所有测试样本的输入提示词 prompt、标签 label 和模型输出预测结果 predict，下面是其中一个示例：

.. code-block:: json

    {"prompt": "system\nYou are a helpful assistant.\nuser\n保持健康的三个提示。\nassistant\n", "label": "以下是保持健康的三个提示：\n\n1. 保持身体活动。每天做适当的身体运动，如散步、跑步或游泳，能促进心血管健康，增强肌肉力量，并有助于减少体重。\n\n2. 均衡饮食。每天食用新鲜的蔬菜、水果、全谷物和脂肪含量低的蛋白质食物，避免高糖、高脂肪和加工食品，以保持健康的饮食习惯。\n\n3. 睡眠充足。睡眠对人体健康至关重要，成年人每天应保证 7-8 小时的睡眠。良好的睡眠有助于减轻压力，促进身体恢复，并提高注意力和记忆力。", "predict": "保持健康的三个提示包括：1. 均衡饮食：饮食应包括各种食物，如蔬菜、水果、全麦面包、蛋白质和健康脂肪，以满足身体的营养需求。\n2. 锻炼：每周至少进行150分钟的中等强度有氧运动，如快走、跑步、游泳或骑自行车，以及至少两次力量训练，以帮助维持身体健康。\n3. 睡眠：保持规律的睡眠习惯，每晚至少睡7-8小时，以帮助身体恢复和充电。"}


``predict_results.json`` 中即为训练效果评估所得结果：

.. code-block:: yaml

    {
        "predict_bleu-4": 50.941235,
        "predict_rouge-1": 65.7085975,
        "predict_rouge-2": 52.576409999999996,
        "predict_rouge-l": 60.487535,
        "predict_runtime": 196.1634,
        "predict_samples_per_second": 0.204,
        "predict_steps_per_second": 0.204
    }


LoRA 模型合并导出
------------------

LoRA 模型合并和导出时，可通过指定 ``export_device`` 参数为 ``auto`` 来自动检测当前加速卡环境，
启用 NPU 作为导出设备：

.. 端到端导出 Qwen1.5-7B LoRA 模型比 cpu 快 37.3% 左右。

.. code-block:: shell
    :linenos:

    ASCEND_RT_VISIBLE_DEVICES=0 llamafactory-cli export \
                --model_name_or_path qwen/Qwen1.5-7B \
                --adapter_name_or_path ./saves/Qwen1.5-7B/lora/sft  \
                --template qwen \
                --finetuning_type lora \
                --export_dir ./saves/Qwen1.5-7B/lora/megred-model-path \
                --export_size 2 \
                --export_device auto \
                --export_legacy_format False

一站式 webui board 的使用
----------------------------

使用 webui 可零代码实现以上功能，启动命令如下：

.. code-block:: shell
    :linenos:

    ASCEND_RT_VISIBLE_DEVICES=0 GRADIO_SHARE=0 GRADIO_SERVER_PORT=7007 GRADIO_SERVER_NAME="0.0.0.0" llamafactory-cli webui

在 webui 实现 Qwen1.5-7B 模型的 LoRA 模型微调、动态推理和模型导出的操作示例：

.. raw:: html

   <iframe width="696" height="422" src="//player.bilibili.com/player.html?bvid=BV1BM3NeAEx1&page=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>


API Server的启动与调用
--------------------------

``API_PORT`` 为 API 服务的端口号，可替换为自定义端口。通过以下命令启动 API 服务：

.. code-block:: shell
    :linenos:

    ASCEND_RT_VISIBLE_DEVICES=0 API_PORT=7007 llamafactory-cli api \
                --model_name_or_path qwen/Qwen1.5-7B \
                --adapter_name_or_path ./saves/Qwen1.5-7B/lora/sft \
                --template qwen \
                --finetuning_type lora

终端输出如下关键信息时，即可在下游任务重通过 API 调用 Qwen1.5-7B

.. code-block:: shell
    :linenos:

    Visit http://localhost:7007/docs for API document.
    INFO:     Started server process [2261535]
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    INFO:     Uvicorn running on http://0.0.0.0:7007 (Press CTRL+C to quit)

使用 API 调用 Qwen1.5-7B 实现问答聊天的示例代码，通过 ``message`` 传入您的问题：

.. code-block:: python
    :linenos:

    import os
    from openai import OpenAI
    from transformers.utils.versions import require_version

    require_version("openai>=1.5.0", "To fix: pip install openai>=1.5.0")

    if __name__ == '__main__':
        # change to your custom port
        port = 7007
        client = OpenAI(
            api_key="0",
            base_url="http://localhost:{}/v1".format(os.environ.get("API_PORT", 7007)),
        )
        messages = []
        messages.append({"role": "user", "content": "hello, what is Ascend NPU"})
        result = client.chat.completions.create(messages=messages, model="test")
        print(result.choices[0].message)

执行成功后可在终端看到如下输出，Qwen1.5-7B 正确介绍了 Ascend NPU：

.. code-block:: shell

    ChatCompletionMessage(content='The Ascend NPU, or Neural Processing Unit, is an AI chip developed by Huawei that is designed to accelerate the performance of deep learning and artificial intelligence workloads. It is specifically designed to be energy-efficient, and is intended to be used in a wide range of devices, from smartphones to data centers. The Ascend NPU is designed to support a variety of AI workloads, including object detection, natural language processing, and speech recognition.', role='assistant', function_call=None, tool_calls=None)

进阶-大模型主流评测 benchmark
--------------------------------

通过以下指令启动对 Qwen1.5-7B 模型在 mmlu 数据集的评测：

.. code-block:: shell
    :linenos:

    llamafactory-cli eval \
        --model_name_or_path qwen/Qwen1.5-7B \
        --template fewshot \
        --task mmlu \
        --split validation \
        --lang en \
        --n_shot 5 \
        --batch_size 1

评测完成后，终端输出的评测结果如下，与 Qwen1.5-7B 官方报告对齐：

.. code-block:: shell

            Average: 61.79                                                                                                                                                              
               STEM: 54.83
    Social Sciences: 73.00
         Humanities: 55.02
              Other: 67.32
