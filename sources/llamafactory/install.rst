安装指南
==============

本教程面向使用 LLAMA-Factory & 昇腾的开发者，帮助完成昇腾环境下 LLaMA-Factory 的安装。此处提供 :ref:`install` 和 :ref:`docker` 两种安装方式。

.. _install:

直接安装
------------

昇腾环境安装
~~~~~~~~~~~~~~~

请根据已有昇腾产品型号及CPU架构等按照 :doc:`快速安装昇腾环境指引 <../ascend/quick_install>` 进行昇腾环境安装，或使用已安装好昇腾环境及 LLaMA-Factory 的 docker 镜像：

- TODO

.. warning::
  LLAMA-Factory 支持的 CANN 最低版本为 8.0.rc1。

  安装 CANN 时，请同时安装 Kernel 算子包。

Python 环境创建
~~~~~~~~~~~~~~~

.. note::
  如果你已经选择使用上述 docker 镜像，可忽略此步骤，直接开始使用 LLaMA-Factory。

.. code-block:: shell
    :linenos:
  
    # 创建 python 3.10 的虚拟环境
    conda create -n <your_env_name> python=3.10
    # 激活虚拟环境
    conda activate <your_env_name>


LLaMA-Factory 安装
~~~~~~~~~~~~~~~~~~~~~~~~

使用以下指令安装带有 torch-npu 的 LLaMA-Factory：

.. code-block:: shell
    :linenos:

    pip install -e .[torch_npu,metrics]

.. _docker:

使用 Docker
------------

进入 docker-npu 目录：

.. code-block:: shell
  
  cd docker/docker-npu

该目录下为 Dockerfile 及 docker compose 配置文件：

Dockerfile:

.. literalinclude:: ./Dockerfile
    :language: docker
    :linenos:

docker-compose.yaml:

.. literalinclude:: ./docker-compose.yaml
    :language: yaml
    :linenos:

.. note::

  默认镜像为 `cosdt/cann:8.0.rc1-910b-ubuntu22.04 <https://hub.docker.com/layers/cosdt/cann/8.0.rc1-910b-ubuntu22.04/images/sha256-29ef8aacf6b2babd292f06f00b9190c212e7c79a947411e213135e4d41a178a9?context=explore>`_。
  更多选择见 `cosdt/cann <https://hub.docker.com/r/cosdt/cann/tags>`_.

使用以下指令构建及启动 docker 容器：

.. code-block:: shell
  
  docker build -f ./Dockerfile \
      --build-arg INSTALL_DEEPSPEED=false \
      --build-arg PIP_INDEX=https://pypi.org/simple \
      -t llamafactory:latest 

安装校验
----------------------

使用 ``llamafactory-cli env`` 指令对 LLaMA-Factory × 昇腾的安装进行校验，如下所示，正确显示 LLaMA-Factory、PyTorch NPU 和 CANN 版本号及 NPU 型号等信息即说明安装成功。

.. code-block:: shell
  
  - `llamafactory` version: 0.8.2.dev0
  - Platform: Linux-4.19.90-vhulk2211.3.0.h1543.eulerosv2r10.aarch64-aarch64-with-glibc2.31
  - Python version: 3.10.14
  - PyTorch version: 2.1.0 (NPU)
  - Transformers version: 4.41.2
  - Datasets version: 2.19.2
  - Accelerate version: 0.31.0
  - PEFT version: 0.11.1
  - TRL version: 0.9.4
  - NPU type: xxx
  - CANN version: 8.0.RC2.alpha001

LLaMA-Factory 卸载
----------------------

.. code-block:: shell
  
  pip uninstall llamafactory

