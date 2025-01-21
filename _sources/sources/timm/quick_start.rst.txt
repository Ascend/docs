快速开始
==================

.. note::

    阅读本篇前，请确保已按照 :doc:`安装教程 <./install>` 准备好昇腾环境及 timm ！
    
本文档帮助昇腾开发者快速使用 timm × 昇腾 进行训练和推理。

导入 torch-npu
---------------------

首先在入口脚本（如本文档中的 ``train.py``， ``validate.py`` ， ``inference.py``）导入 torch 后，导入 torch-npu：

.. code-block:: python
    :linenos:
    :emphasize-lines: 2

    import torch
    import torch-npu


单卡/分布式训练
---------------------

以 ``ImageNet-1000`` 数据集的训练为例，使用以下脚本启动单卡/多卡 NPU 上基于 timm 的图像分类模型训练：

.. note::

    请根据您的 NPU 环境指定 NPU 卡数量 ``num_npus`` 和模型名称/路径 ``model`` ，并替换数据集路径 ``path/to/dataset/ImageNet-1000``

.. code-block:: shell
    :linenos:
    :emphasize-lines: 1,3

    num_npus=1
    ./distributed_train.sh $num_npus path/to/dataset/ImageNet-1000 \
        --device npu \
        --model seresnet34 \
        --sched cosine \
        --epochs 150 \
        --warmup-epochs 5 \
        --lr 0.4 \
        --reprob 0.5 \
        --remode pixel \
        --batch-size 256 \
        --amp -j 4


模型验证
---------------------

.. note::

    请根据实际情况替换验证集数据路径 ``path/to/data`` 、模型路径 ``path/to/model``

.. code-block:: shell
    :linenos:

    python validate.py path/to/data --device npu --model path/to/model --batch-size 64 --pretrained


正常输出验证过程日志及最终验证结果 ``result`` 说明验证成功，如下为一种示例（根据模型及数据集不同，日志会有区别）：

.. code-block:: shell

    Validating in float32. AMP not enabled.                                                                                                                                                    
    Loading pretrained weights from Hugging Face hub (timm/tiny_vit_21m_512.dist_in22k_ft_in1k)                                                                                                
    [timm/tiny_vit_21m_512.dist_in22k_ft_in1k] Safe alternative available for 'pytorch_model.bin' (as 'model.safetensors'). Loading weights using safetensors.                                 
    Model ./model_ckpts/tiny_vit_21m_512 created, param count: 21268120                                                                                                                        
    Data processing configuration for current model + dataset:                                                                                                                                 
            input_size: (3, 512, 512)                                                                                                                                                          
            interpolation: bicubic                                                                                                                                                             
            mean: (0.485, 0.456, 0.406)                                                                                                                                                        
            std: (0.229, 0.224, 0.225)                                                                                                                                                         
            crop_pct: 1.0                                                                                                                                                                      
            crop_mode: squash                                                                                                                                                                  
    Test: [   0/157]  Time: 7.083s (7.083s,    9.04/s)  Loss:  0.4765 (0.4765)  Acc@1:  93.750 ( 93.750)  Acc@5:  96.875 ( 96.875)                                                             
    Test: [  10/157]  Time: 0.400s (1.008s,   63.50/s)  Loss:  0.6594 (0.4929)  Acc@1:  78.125 ( 87.926)  Acc@5:  98.438 ( 98.011)                                                             
    Test: [  20/157]  Time: 0.399s (0.719s,   89.04/s)  Loss:  0.1891 (0.4682)  Acc@1:  96.875 ( 89.435)  Acc@5: 100.000 ( 98.289)

    ... ...
    
    * Acc@1 86.040 (13.960) Acc@5 97.750 (2.250)
    --result
    {
        "model": "./model_ckpts/tiny_vit_21m_512",
        "top1": 86.04,
        "top1_err": 13.96,
        "top5": 97.75,
        "top5_err": 2.25,
        "param_count": 21.27,
        "img_size": 512,
        "crop_pct": 1.0,
        "interpolation": "bicubic"
    }

模型推理
------------------

.. note::

    请根据实际情况替换验证集数据路径 ``path/to/data`` 和模型权重路径 ``path/to/checkpoint/model_best.pth.tar``


.. code-block:: shell
    :linenos:
    :emphasize-lines: 2

    python inference.py ../open_clip/data/ImageNet-1000/val/ \
        --device npu \
        --batch-size 64 \
        --model ./model_ckpts/tiny_vit_21m_512 \
        --label-type detail \
        --topk 5

正常输出验证过程日志及最终验证结果 ``result`` 说明验证成功，如下为一种示例（根据模型及数据集不同，日志会有区别）：

.. code-block:: shell

    Running inference in float32. AMP not enabled.
    Loading pretrained weights from Hugging Face hub (timm/tiny_vit_21m_512.dist_in22k_ft_in1k)
    [timm/tiny_vit_21m_512.dist_in22k_ft_in1k] Safe alternative available for 'pytorch_model.bin' (as 'model.safetensors'). Loading weights using safetensors.
    Model ./model_ckpts/tiny_vit_21m_512 created, param count: 21268120
    Predict: [0/157] Time 6.418 (6.418)
    Predict: [10/157] Time 0.394 (0.942)
    Predict: [20/157] Time 0.427 (0.708)

    ... ...

    "ILSVRC2012_val_00005844.JPEG":{
        "label":[
            "stinkhorn, carrion fungus: any of various ill-smelling brown-capped fungi of the order Phallales",
            "earthstar: any fungus of the family Geastraceae; in form suggesting a puffball whose outer peridium splits into the shape of a star",
            "coral fungus: any of numerous fungi of the family Clavariaceae often brightly colored that grow in often intricately branched clusters like coral",
            "mushroom: fleshy body of any of numerous edible fungi",
            "gyromitra: any fungus of the genus Gyromitra"
        ],
        "prob":[
            0.878154695,
            0.0030552391,
            0.0012754521,
            0.0010740706,
            0.000946458
        ]
    },

    ... ...