安装指南
==============

本教程面向使用 torchchat & 昇腾的开发者，帮助完成昇腾环境下 torchchat 的安装。

昇腾环境安装
------------

请根据已有昇腾产品型号及CPU架构等按照 :doc:`快速安装昇腾环境指引 <../ascend/quick_install>` 进行昇腾环境安装，或直接获取对应产品的昇腾环境镜像 `ascendai/cann <https://hub.docker.com/r/ascendai/cann/tags>`_ 。

.. warning::
  torchchat 依赖 CANN 最低版本为 8.0.0，安装 CANN 时，请同时安装 Kernel 算子包。

torchchat 安装
----------------------

环境创建
~~~~~~~~~~~~
torchchat 依赖于 Python 3.10+，强烈建议使用 venv 或者 conda 创建 Python 环境。

- 使用 venv 创建 Python 环境

.. code-block:: shell
    :linenos:
  
    # 创建名为 torchchat 的 python 3.10 的虚拟环境
    python -m venv .venv
    # 激活虚拟环境
    source .venv/bin/activate

- 使用 conda 创建 Python 环境

.. code-block:: shell
    :linenos:

    # 安装miniconda
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh

    # 创建名为 torchchat 的 python 3.10 的虚拟环境
    conda create -y -n torchchat python=3.10
    # 激活虚拟环境
    conda activate torchchat

使用 git 获取源码
~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell
    :linenos:
  
    git clone https://github.com/pytorch/torchchat.git
    cd torchchat

一键安装
~~~~~~~~~~~~~
torchchat 提供一键安装脚本，安装依赖包。

.. code-block:: shell
    :linenos:
  
    # 安装依赖包
    ./install/install_requirements.sh
    # 查看安装的依赖包
    pip list

以下为主要依赖包：

.. code-block:: shell
    :linenos:

    Package                  Version
    ------------------------ -------------------
    torch                     2.7.0.dev20250310+cpu
    torchao                   0.10.0+git711fa080
    torchdata                 0.11.0
    torchtune                 0.6.0
    torchvision               0.22.0.dev20250310


安装 torch_npu
~~~~~~~~~~~~~~~
torch_npu 是昇腾 AI 计算框架的核心组件，提供了对昇腾硬件的支持，
目前 torchchat 支持 2.7.0 版本的 torch_npu，安装时请确保 CANN 版本为 8.0.0 及以上。

nightly 版本的 torch_npu 需要通过源代码进行安装。

环境安装:

.. code-block:: shell

    # 获取源代码
    git clone https://github.com/Ascend/pytorch.git
    cd pytorch

    # 安装依赖包
    pip install wheel
    pip install -r requirements.txt

    # 安装 torch_npu
    python setup.py build build_py

    # 查看安装版本
    torch_version=$(python -c "import torch; print(torch.__version__)")
    torch_npu_version=$(python -c "import torch_npu; print(torch_npu.__version__)")
    echo "torch version: ${torch_version}"
    echo "torch_npu version: ${torch_npu_version}"

卸载 torch_npu
~~~~~~~~~~~~~~~~
torch_npu 依赖于 CANN 的安装路径，卸载时请确保 CANN 的安装路径正确。

.. code-block:: shell
    :linenos:

    # 卸载 torch_npu
    cd pytorch
    python setup.py clean

    # 卸载依赖包
    pip uninstall -y torch_npu