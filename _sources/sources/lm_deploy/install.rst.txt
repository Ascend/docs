安装指南
==============

LMDeploy 是一个用于大型语言模型（LLMs）和视觉-语言模型（VLMs）压缩、部署和服务的 Python 库。其核心推理引擎包括 TurboMind 引擎和 PyTorch 引擎，
前者由 C++ 和 CUDA 开发，致力于推理性能的优化，而后者纯 Python 开发，旨在降低开发者的门槛。

本教程面向使用 lm-deploy & 昇腾的开发者，帮助完成昇腾环境下 lm-deploy 的安装。


lm_deploy 下载安装
---------------------------

使用 pip 安装（推荐）
++++++++++++++++++++++++++

推荐在一个干净的 conda 环境下（python3.8 - 3.12），安装 lmdeploy ：

.. code-block:: shell 
    :linenos:

    conda create -n lmdeploy python=3.8 -y
    conda activate lmdeploy
    pip install lmdeploy


从源码安装
+++++++++++++++++++++++++++++++

如果你使用 PyTorch 引擎进行推理，从源代码安装非常简单：

.. code-block:: shell 
    :linenos:

    git clone https://github.com/InternLM/lmdeploy.git
    cd lmdeploy
    pip install -e .

安装校验
-----------------

安装过程中未出现错误，且执行下面命令后出现 lmdeploy 版本号即为安装成功。

.. code-block:: shell
    :linenos:

    python -c "import lmdeploy; print(lmdeploy.__version__)"

    # 以下为输出示例
    # 0.6.2


