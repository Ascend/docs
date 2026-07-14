安装指南
==============

本教程面向使用 SGLang & 昇腾的开发者，帮助完成昇腾环境下 SGLang 的安装。截至 2025 年 9 月，该项目涉及的如下组件正在活跃开发中，建议使用最新版本，并注意版本以及设备兼容性。

昇腾环境安装
------------

请根据已有昇腾产品型号及 CPU 架构等按照 :doc:`快速安装昇腾环境指引 <../ascend/quick_install>` 进行昇腾环境安装。

.. warning::
  CANN 推荐版本为 8.2.RC1 以上，安装 CANN 时，请同时安装 Kernel 算子包以及 nnal ARM 平台加速库软件包。


SGLang 安装
----------------------

方法1：使用源码安装 SGLang
~~~~~~~~~~~~~~~~~~~~~~


Python 环境创建
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell
    :linenos:

    # Create a new conda environment, and only python 3.11 is supported
    conda create --name sglang_npu python=3.11
    # Activate the virtual environment
    conda activate sglang_npu

安装 python 依赖
^^^^^^^^^^^^^^^^^^^^^^ 

.. code-block:: shell
    :linenos:

    pip install attrs==24.2.0 numpy==1.26.4 scipy==1.13.1 decorator==5.1.1 psutil==6.0.0 pytest==8.3.2 pytest-xdist==3.6.1 pyyaml


MemFabric Adaptor 安装
^^^^^^^^^^^^^^^^^^^^^^ 

MemFabric Adaptor 是 Mooncake Transfer Engine 在昇腾 NPU 集群上实现 KV cache 传输的替代方案。


目前，MemFabric Adaptor 仅支持 aarch64 架构的设备。请根据实际架构选择安装：

.. code-block:: shell
    :linenos:

    MF_WHL_NAME="mf_adapter-1.0.0-cp311-cp311-linux_aarch64.whl"
    MEMFABRIC_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/sglang/${MF_WHL_NAME}"
    wget -O "${MF_WHL_NAME}" "${MEMFABRIC_URL}" && pip install "./${MF_WHL_NAME}"


torch-npu 安装
^^^^^^^^^^^^^^^^^^^^^^

按照 :doc:`torch-npu 安装指引 <../pytorch/install>` 本项目由于 NPUGraph 和 Triton-Ascend 的限制，目前仅支持安装 2.6.0 版本 torch 和 torch-npu，后续会推出更通用的版本方案。

.. code-block:: shell
    :linenos:

    # Install torch 2.6.0 and torchvision 0.21.0 on CPU only
    PYTORCH_VERSION=2.6.0
    TORCHVISION_VERSION=0.21.0
    pip install torch==$PYTORCH_VERSION torchvision==$TORCHVISION_VERSION --index-url https://download.pytorch.org/whl/cpu

    # Install torch_npu 2.6.0 or you can just pip install torch_npu==2.6.0
    PTA_VERSION="v7.2.0-pytorch2.6.0"
    PTA_NAME="torch_npu-2.6.0.post3-cp311-cp311-manylinux_2_28_aarch64.whl"
    PTA_URL="https://gitcode.com/ascend/pytorch/releases/download/${PTA_VERSION}/${PTA_NAME}"
    wget -O "${PTA_NAME}" "${PTA_URL}" && pip install "./${PTA_NAME}"

安装完成后，可以通过以下代码验证 torch_npu 是否安装成功：

.. code-block:: shell
    :linenos:

    import torch
    # import torch_npu # In torch 2.6.0，no need to import torch_npu explicitly

    x = torch.randn(2, 2).npu()
    y = torch.randn(2, 2).npu()
    z = x.mm(y)

    print(z)

程序能够成功打印矩阵 Z 的值即为安装成功。

vLLM 安装
^^^^^^^^^^^^^^^^^^^^^^

vLLM 目前仍是昇腾 NPU 上的一个主要前提条件。基于 torch==2.6.0 版本，vLLM 需要从源码编译安装 v0.8.5 版本。

.. code-block:: shell
    :linenos:

    VLLM_TAG=v0.8.5
    git clone --depth 1 https://github.com/vllm-project/vllm.git --branch $VLLM_TAG
    cd vllm 
    VLLM_TARGET_DEVICE="empty" pip install -v -e .
    cd ..

Triton-Ascend 安装
^^^^^^^^^^^^^^^^^^^^^^

Triton Ascend还在频繁更新。为能使用最新功能特性，建议拉取代码进行源码安装。详细安装步骤请参考 `安装指南 <https://gitcode.com/Ascend/triton-ascend/blob/master/docs/sources/getting-started/installation.md>`_。

或者选择安装 Triton Ascend nightly 包：

.. code-block:: shell
    :linenos:

    pip install -i https://test.pypi.org/simple/ "triton-ascend<3.2.0rc" --pre --no-cache-dir


安装 Deep-ep 与 sgl-kernel-npu:
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell
    :linenos:
    
    pip install wheel==0.45.1
    git clone https://github.com/sgl-project/sgl-kernel-npu.git

    # Add environment variables
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64/stub:$LD_LIBRARY_PATH
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    cd sgl-kernel-npu

    # Compile and install deep-ep, sgl-kernel-npu
    bash build.sh
    pip install output/deep_ep*.whl output/sgl_kernel_npu*.whl --no-cache-dir
    cd ..
    rm -rf sgl-kernel-npu

    # Link to the deep_ep_cpp.*.so file
    cd "$(pip show deep-ep | grep -E '^Location:' | awk '{print $2}')" && ln -s deep_ep/deep_ep_cpp*.so


源码安装 SGLang：
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell
    :linenos:

    # Use the last release branch
    git clone -b v0.5.3rc0 https://github.com/sgl-project/sglang.git
    cd sglang

    pip install --upgrade pip
    # Install SGLang with NPU support
    pip install -e python[srt_npu]
    cd ..



方法2：使用 docker 镜像安装 SGLang
~~~~~~~~~~~~~~~~~~~~~~

注意：--privileged 和 --network=host 是 RDMA 所必需的，而 RDMA 通常也是 Ascend NPU 集群的必备组件。

以下 Docker 命令基于 Atlas 800I A3 机型。若使用 Atlas 800I A2 机型，请确保仅将 davinci [0-7] 映射到容器中。

.. code-block:: shell
    :linenos:

    # Clone the SGLang repository
    git clone https://github.com/sgl-project/sglang.git
    cd sglang/docker

    # Build the docker image
    docker build -t <image_name> -f Dockerfile.npu .

    alias drun='docker run -it --rm --privileged --network=host --ipc=host --shm-size=16g \
        --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 \
        --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 \
        --device=/dev/davinci8 --device=/dev/davinci9 --device=/dev/davinci10 --device=/dev/davinci11 \
        --device=/dev/davinci12 --device=/dev/davinci13 --device=/dev/davinci14 --device=/dev/davinci15 \
        --device=/dev/davinci_manager --device=/dev/hisi_hdc \
        --volume /usr/local/sbin:/usr/local/sbin --volume /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        --volume /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
        --volume /etc/ascend_install.info:/etc/ascend_install.info \
        --volume /var/queue_schedule:/var/queue_schedule --volume ~/.cache/:/root/.cache/'

    # Run the docker container and start the SGLang server
    drun --env "HF_TOKEN=<secret>" \
        <image_name> \
        python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --attention-backend ascend --host 0.0.0.0 --port 30000

