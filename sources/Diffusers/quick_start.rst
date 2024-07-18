快速开始
==================

本示例以文生图 Diffusers 库中文生图任务为样例，展示如何进行文生图模型 stable-diffusion-xl-base-1.0 的基于 LoRA 的微调及动态合并 LoRA 的推理。

文生图
-------------

.. _download:

模型及数据集下载
~~~~~~~~~~~~~~~~~~~~

1. 请提前下载 `stabilityai/stable-diffusion-xl-base-1.0 <https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0>`_ 模型至自定义路径

2. 请提前下载 `madebyollin/sdxl-vae-fp16-fix <https://huggingface.co/madebyollin/sdxl-vae-fp16-fix>`_ 模型至自定义路径

3. 请提前下载 `reach-vb/pokemon-blip-captions <https://huggingface.co/datasets/reach-vb/pokemon-blip-captions>`_ 数据集至自定义路径


.. _finetune:

基于 LoRA 的微调
~~~~~~~~~~~~~~~~~~~~

进入 Diffusers 项目目录，新建并执行以下脚本：

.. note::
    
    请根据 :ref:`download` 中模型及数据集的实际缓存路径指定 stable-diffusion-xl-base-1.0 模型缓存路径 ``MODEL_NAME``，sdxl-vae-fp16-fix 模型缓存路径 ``VAE_NAME`` 和。

.. code-block:: shell
    :linenos:
    :emphasize-lines: 1,2,3

    export MODEL_NAME="./models_ckpt/stable-diffusion-xl-base-1.0/"
    export VAE_NAME="./ckpt/sdxl-vae-fp16-fix"
    export TRAIN_DIR="~/diffusers/data/pokemon-blip-captions/pokemon"

    python3  ./examples/text_to_image/train_text_to_image_lora_sdxl.py \
        --pretrained_model_name_or_path=$MODEL_NAME \
        --pretrained_vae_model_name_or_path=$VAE_NAME \
        --dataset_name=$DATASET_NAME --caption_column="text" \
        --resolution=1024 \
        --random_flip \
        --train_batch_size=1 \
        --num_train_epochs=2 \
        --checkpointing_steps=500 \
        --learning_rate=1e-04 \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --mixed_precision="no" \
        --seed=42 \
        --output_dir="sd-pokemon-model-lora-sdxl" \
        --validation_prompt="cute dragon creature"

微调过程无报错，并且终端显示 ``Steps: 100%`` 的进度条说明微调成功。


动态合并 LoRA 的推理
~~~~~~~~~~~~~~~~~~~~

.. note::

    请根据 :ref:`download` 中模型实际缓存路径指定 ``model_path``
    
    根据  :ref:`finetune` 中指定的 LoRA 模型路径 ``output_dir`` 指定 ``lora_model_path``

    [可选] 修改 ``prompt`` 可使得生成图像改变

.. code-block:: python
    :linenos:
    :emphasize-lines: 9

    from diffusers import DiffusionPipeline
    import torch

    lora_model_path = "path/to/sd-pokemon-model-lora-sdxl/checkpoint-800/"
    model_path = "./models_ckpt/stable-diffusion-xl-base-1.0/"
    pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)

    # 将模型放到 NPU 上
    pipe.to("npu")

    # 加载 LoRA 权重
    pipe.load_lora_weights(lora_model_path)
    # 输入 prompt
    prompt = "Sylveon Pokemon with elegant features, magical design, \
            light purple aura, extremely detailed and intricate markings, \
            photo realistic, unreal engine, octane render"
    # 推理
    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]

    image.save("pokemon-finetuned-inference-generation.png")


微调过程无报错，并且终端显示 ``Loading pipeline components...: 100%`` 的进度条说明微调成功。
查看当前目录下保存的 ``pokemon-finetuned-inference-generation.png`` 图像，可根据 ``prompt`` 生成内容相关的图像说明推理成功。

