快速开始
==================

.. note::

    阅读本篇前，请确保已按照 :doc:`安装教程 <./install>` 准备好昇腾环境及 SGLang ！
    
    本篇教程将介绍如何使用 SGLang 进行快速开发，帮助您快速上手 SGLang。

本文档帮助昇腾开发者快速使用 SGLang × 昇腾 进行 LLM 推理服务。可以访问 `这篇官方文档 <https://docs.sglang.ai/>`_ 获取更多信息。

概览
------------------------

SGLang 是一款适用于 LLM 和 VLM 的高速服务框架。通过协同设计后端运行时环境与前端语言，让用户与模型的交互更快速、更可控。

使用 SGLang 启动服务
------------------------

以下示例展示了如何使用 SGLang 启动一个简单的会话生成服务：

启动一个 server:

.. code-block:: shell
    :linenos:

    # Launch the SGLang server on NPU
    python -m sglang.launch_server --model Qwen/Qwen2.5-0.5B-Instruct \
    --device npu --port 8000 --attention-backend ascend \
    --host 0.0.0.0 --trust-remote-code

启动成功后，将看到类似如下的日志输出：

.. code-block:: shell
    :linenos:
    
    INFO:     Started server process [89394]
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
    INFO:     127.0.0.1:40106 - "GET /get_model_info HTTP/1.1" 200 OK
    Prefill batch. #new-seq: 1, #new-token: 128, #cached-token: 0, token usage: 0.00, #running-req: 0, #queue-req: 0, 
    INFO:     127.0.0.1:40108 - "POST /generate HTTP/1.1" 200 OK
    The server is fired up and ready to roll!

使用 curl 进行测试：

.. code-block:: shell
    :linenos:

    curl -s http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "qwen/qwen2.5-0.5b-instruct",
            "messages": [
            {
                "role": "user",
                "content": "What is the capital of France?"
            }
            ]
        }'

将看到类似如下返回结果：

.. code-block:: shell
    :linenos:

    {"id":"3f2f1aa779b544c19f01c08b803bf4ef","object":"chat.completion","created":1759136880,"model":"qwen/qwen2.5-0.5b-instruct","choices":[{"index":0,"message":{"role":"assistant","content":"The capital of France is Paris.","reasoning_content":null,"tool_calls":null},"logprobs":null,"finish_reason":"stop","matched_stop":151645}],"usage":{"prompt_tokens":36,"total_tokens":44,"completion_tokens":8,"prompt_tokens_details":null,"reasoning_tokens":0},"metadata":{"weight_version":"default"}}

使用 SGLang 进行推理验证
------------------------

以下代码展示了如何使用 SGLang 进行推理验证：

.. code-block:: shell
    :linenos:

    # example.py
    import torch

    import sglang as sgl

    def main():
        
        prompts = [
            "Hello, my name is",
            "The Independence Day of the United States is",
            "The capital of Germany is",
            "The full form of AI is",
        ] * 1

        llm = sgl.Engine(model_path="/Qwen2.5/Qwen2.5-0.5B-Instruct", device="npu", attention_backend="ascend")

        sampling_params = {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 100}
        
        outputs = llm.generate(prompts, sampling_params)
        for prompt, output in zip(prompts, outputs):
            print("===============================")
            print(f"Prompt: {prompt}\nGenerated text: {output['text']}")

    if __name__ == '__main__':
        main()

运行 example.py 进行测试，查看是否得到输出即可验证 SGLang 是否安装成功。