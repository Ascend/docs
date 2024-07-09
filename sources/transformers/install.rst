安装指南
===========

本文将介绍如何在昇腾环境下使用transfomers, 帮助开发者完成transformers的安装

.. note:: 

    请确保环境安装了对应的固件和驱动, 详情请参考 `快速安装昇腾环境 <../ascend/quick_install.html>`_。

创建虚拟环境
--------------------

首先需要安装并激活python环境

.. code-block:: shell

    conda create -n your_env_name python=3.10
    conda activate your_env_name

安装transformers
----------------------

直接使用pip命令进行安装

.. code-block:: shell

    pip install transformers

验证安装
--------------------

.. code-block:: python 

    from transformers import pipeline
    print(pipeline('sentiment-analysis')('we love you'))

如果成功运行并输出下面或类似内容， 则安装成功

.. code-block:: shell 

    [{'label': 'POSITIVE', 'score': 0.9998704791069031}]

transformers卸载
---------------------

.. code-block:: shell 

    pip uninstall transformers


