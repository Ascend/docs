快速开始
==================

.. note::
    阅读本篇前，请确保已按照 :doc:`安装教程 <./install>` 准备好昇腾环境及 open_clip ！

本文档帮助昇腾开发者快速使用 open_clip × 昇腾 进行训练和推理。

使用 NPU 的训练
---------------------

首先在 ``src/training/main.py`` 脚本导入 torch 后，导入 torch-npu，并将 cuda 对应的 ``GradScaler`` 替换为 npu 的：

.. code-block:: python
    :linenos:
    
    import torch
    import torch-npu
    from torch.npu.amp import GradScaler

以 ``MS_COCO_2017_URL_TEXT`` 数据集的训练为例，使用在 ``DataComp`` 数据集训练过的 ``CLIP-ViT-B-32`` 模型权重为预训练权重，使用以下脚本启动单卡/多卡 NPU 上的训练：

单卡训练
~~~~~~~~~~~~~~~

.. note::

    请根据实际情况指定数据集路径 ``train-data`` 、 ``val-data`` 、 ``imagenet-val`` 和预训练模型路径 ``pretrained``

.. code-block:: shell
    :linenos:

    python -m training.main \
        --model ViT-B-32 \
        --save-frequency 1 \
        --zeroshot-frequency 1 \
        --report-to tensorboard \
        --train-data="./data/MS_COCO_2017_URL_TEXT/traincoco.csv" \
        --val-data="./data/MS_COCO_2017_URL_TEXT/traincoco.csv" \
        --imagenet-val="./data/ImageNet-1000/val/" \
        --pretrained "./models/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K/open_clip_pytorch_model.bin" \
        --warmup 10000 \
        --batch-size=128 \
        --lr=1e-3 \
        --wd=0.1 \
        --epochs=8 \
        --workers=8 \
        --seed 0 

分布式训练
~~~~~~~~~~~~~~~

使用 torchrun 启动 NPU 分布式训练，需指定通信后端为 hccl（``--dist-backend="hccl"``）:

.. note::

    请根据实际情况指定数据集路径 ``train-data`` 、 ``val-data`` 、 ``imagenet-val`` 和预训练模型路径 ``pretrained``
    
    ``nproc_per_node`` 需指定为每个节点卡的数量，为 torchrun 所需参数，更多 torchrun 相关参数详细含义可参考 `PyTorch 官方文档 <https://pytorch.org/docs/stable/elastic/run.html>`_。


.. code-block:: shell
    :linenos:

    # train on multi-npu
    torchrun --nproc_per_node 2 -m training.main \
        --save-frequency 1 \
        --zeroshot-frequency 1 \
        --report-to tensorboard \
        --train-data="./data/MS_COCO_2017_URL_TEXT/traincoco.csv" \
        --val-data="./data/MS_COCO_2017_URL_TEXT/traincoco.csv" \
        --imagenet-val="./data/ImageNet-1000/val/" \
        --pretrained "./models/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K/open_clip_pytorch_model.bin" \
        --warmup 10000 \
        --batch-size=64 \
        --lr=1e-3 \
        --wd=0.1 \
        --epochs=1 \
        --workers=8 \
        --seed 0 \
        --model ViT-B-32 \
        --dist-backend="hccl"

使用 NPU 的推理
---------------------

一般而言，自定义脚本中使用 open_clip 在昇腾上训练，需要导入 torch-npu，并将数据和模型放到 NPU 上，如下样例所示：

.. note::

    请根据实际情况替换模型缓存路径 ``/path/to/modelsViT-B-32/`` 、 ``/path/to/models/ViT-B-32/ViT-B-32.pt`` 、 ``/path/to/your/image.jpg``

.. code-block:: python
    :linenos:
    :emphasize-lines: 2,14,15,16,18

    import torch
    import torch_npu
    from PIL import Image
    import open_clip as clip

    # 下载模型至指定缓存路径
    model = clip.openai.load_openai_model('ViT-B-32', cache_dir="/path/to/modelsViT-B-32/")

    model, _, preprocess = clip.create_model_and_transforms('ViT-B-32', pretrained='/path/to/models/ViT-B-32/ViT-B-32.pt')
    tokenizer = clip.get_tokenizer('ViT-B-32')

    # put inputs and model to npu
    image = preprocess(Image.open("/path/to/your/image.jpg")).unsqueeze(0).to("npu")
    text = tokenizer(["a diagram", "a dog", "a cat"]).to("npu")
    model = model.to("npu")

    with torch.no_grad(), torch.npu.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]

本示例所用输入图像：

.. figure:: ./images/CLIP.png
  :align: center
  :scale: 50%

对应输出以下内容，正确预测其分类为 a dog：

.. code-block:: shell

  Label probs: tensor([[0.0010, 0.9941, 0.0049]], device='npu:0')

模型评估
------------------

在 ``src/training/profiler.py`` 脚本导入 torch-npu，并将模型放到 NPU 上：

.. code-block:: python
    :linenos:
    :emphasize-lines: 4,16,17
    
    import argparse

    import torch
    import torch_npu

    import open_clip
    import pandas as pd
    from torch.utils.flop_counter import FlopCounterMode

    ... ...

    def profile_model(model_name, batch_size=1, profiler='torch'):
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        elif torch.npu.is_available():
            model = model.npu()

使用以下指令完成模型评估：

.. code-block:: shell
    :linenos:
    
    python3 -m training.profiler --model ViT-L-14 --results-file "./logs/profiler_results.csv"

评估结果保存在 ``./logs/profiler_results.csv`` 文件中：

.. code-block:: shell

    model,image_size,image_width,text_width,embed_dim,mparams,image_mparams,text_mparams,gflops,image_gflops,text_gflops
    ViT-L-14,224,1024,768,768,427.62,303.97,123.65,175.33,162.03,13.3

