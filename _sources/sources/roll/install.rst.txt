安装指南
==============

本教程面向使用 roll & Ascend 的开发者，帮助完成昇腾环境下 roll 的安装。

昇腾环境安装
------------

请根据已有昇腾产品型号及 CPU 架构等按照 :doc:`快速安装昇腾环境指引 <../ascend/quick_install>` 进行昇腾环境安装。

.. warning::
  CANN 最低版本为 8.2.RC1，安装 CANN 时，请同时安装 Kernel 算子包以及 nnal 加速库软件包。

Python 环境创建
----------------------

.. code-block:: shell
    :linenos:

    # 创建名为 roll 的 python 3.10 的虚拟环境
    conda create -y -n roll python=3.10
    # 激活虚拟环境
    conda activate roll

Torch 安装创建
----------------------

.. code-block:: shell
    :linenos:

    # 安装 torch 的 CPU 版本
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu

    # 安装 torch_npu
    pip install torch_npu==2.5.1

vllm & vllm-ascend 安装
----------------------

.. code-block:: shell
    :linenos:

    # vllm
    git clone -b v0.8.4 --depth 1 https://github.com/vllm-project/vllm.git
    cd vllm

    VLLM_TARGET_DEVICE=empty pip install -v -e .
    cd ..


.. code-block:: shell
    :linenos:

    # vllm-ascend
    git clone -b v0.8.4rc2 --depth 1 https://github.com/vllm-project/vllm-ascend.git
    cd vllm-ascend

    export COMPILE_CUSTOM_KERNELS=1
    pip install -e .
    cd ..

如果在安装 vllm-ascend 时遇到类似以下问题：

.. code-block:: shell
    :linenos:

    RuntimeError: CMake configuration failed: Command '['/pathto/miniconda3/envs/roll/bin/python3.10', '-m', 'pybind11', '--cmake']' returned non-zero exit status 2.

可尝试在 vllm-ascend 目录下 setup.py 文件 151-158 行进行如下修改并重新进行编译：

.. code-block:: shell
    :linenos:

    try:
        # if pybind11 is installed via pip
        pybind11_cmake_path = (subprocess.check_output(
            [python_executable, "-m", "pybind11",
            "--cmakedir"]).decode().strip())
    except subprocess.CalledProcessError as e:
        # else specify pybind11 path installed from source code on CI container
        raise RuntimeError(f"CMake configuration failed: {e}")

安装 roll
----------------------  

使用以下指令安装 roll 及相关依赖：

.. code-block:: shell
    :linenos:

    git clone https://github.com/alibaba/ROLL.git
    cd ROLL

    # Install roll dependencies
    pip install -r requirements_common.txt
    pip install deepspeed==0.16.0


其他第三方库说明
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20

   * - Software
     - Description
   * - transformers
     - v4.52.4
   * - flash_attn
     - not supported
   * - transformer-engine[pytorch]
     - not supported


1. 支持通过 transformers 使能 --flash_attention_2， transformers 需等于 4.52.4版本。

2. 不支持通过 flash_attn 使能 flash attention 加速。

3. 暂不支持 transformer-engine[pytorch]。

   
.. code-block:: shell
    :linenos:

    pip install transformers==4.52.4