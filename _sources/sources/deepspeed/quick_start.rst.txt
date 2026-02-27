快速开始
==========

.. note:: 
    在本示例之前，请确保已经安装了 `DeepSpeed <./install.html>`_ 环境。 如果还未安装，可以执行 ``pip install deepspeed`` 完成安装。


1. 使用DeepSpeed多卡并行训练
-------------------------------
以下代码使用了cifar10数据集，使用DeepSpeed训练模型在多张NPU卡上进行模型训练（来自 `DeepSpeed Examples <https://github.com/microsoft/DeepSpeedExamples/blob/master/training/cifar/cifar10_deepspeed.py>`_），自DeepSpeed v0.12.6之后，代码无需任何修改，即可自动检测NPU并进行训练。

.. rli:: https://raw.githubusercontent.com/microsoft/DeepSpeedExamples/master/training/cifar/cifar10_deepspeed.py
    :language: python
    :linenos:

2. 训练结果查看
----------------
训练完成后，会打印模型对图像识别的结果。

.. code-block:: shell
    :linenos:

    Finished Training
    Accuracy of the network on the 10000 test images:  57 %
    Accuracy of plane :  65 %
    Accuracy of   car :  67 %
    Accuracy of  bird :  52 %
    Accuracy of   cat :  34 %
    Accuracy of  deer :  52 %
    Accuracy of   dog :  49 %
    Accuracy of  frog :  59 %
    Accuracy of horse :  66 %
    Accuracy of  ship :  66 %
    Accuracy of truck :  56 %
