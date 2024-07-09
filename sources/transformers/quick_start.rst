快速开始
============

.. note::

    阅读本篇前，请确保已按照 :doc:`安装教程 <./install>` 准备好昇腾环境及 transformers!
    关于具体的模型获取和训练, 可以转到 :doc:`模型训练和推理 <./fine-tune>`


模型使用可分为使用AutoModelForCausalLM, pipeline两种方式, 本质上没有区别

使用AutoModelForCausalLM
-----------------------------------------------

.. code-block:: python
    :linenos:

    import torch
    import torch_npu
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "path/to/model"
    device = "npu:0" if torch.npu.isavailable() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).to(device)


使用pipeline
-------------------------

.. code-block:: python
    :linenos:

    import transformers
    import torch
    import torch_npu   
    
    model_id = "path/to/model"
    device = "npu:0" if torch.npu.isavailable() else "cpu"
    
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=device,
    )


全流程
----------

.. code-block:: python
    :linenos:

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    import torch_npu 

    #如果提前下载好模型将meta-llama/Meta-Llama-3-8B-Instruct更换为本地地址
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    device = "npu:0"  if torch.npu.isavailable() else "cpu" # 指定使用的设备为 NPU 0

    # 加载预训练的分词器
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 加载预训练的语言模型, 并指定数据类型为bfloat16, 自动选择设备映射
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).to(device) # 将模型移动到指定的设备
    
    # 定义消息列表，包含系统消息和用户消息
    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ]
    
    # 使用分词器将消息列表应用到聊天模板中，并转换为张量
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt" # 返回 PyTorch 张量
    ).to(model.device)
    

    # 定义终止标记，包括模型的结束标记 ID 和一个空标记 ID
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    # 生成响应
    outputs = model.generate(
        input_ids,
        max_new_tokens=256, # 设置生成的最大token
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6, # 设置采样温度，影响生成的多样性
        top_p=0.9,
    )

    # 获取生成的响应，排除输入的部分
    response = outputs[0][input_ids.shape[-1]:]
    print(tokenizer.decode(response, skip_special_tokens=True))

输出示例

.. code-block:: shell
    :linenos:

    Arrrr, shiver me timbers! Me name be Captain Chat, the scurviest pirate chatbot to ever sail the Seven Seas! Me be here to swab yer deck with me wit and wisdom, and to make ye laugh with me pirate puns and jokes. So hoist the sails and set course for adventure, matey!

