安装指南
==================

本文面向昇腾开发者，帮助开发者完成stable-diffusion-webui在昇腾上的安装

.. note:: 

    请确保环境安装了对应的固件和驱动，详情请参考 `快速安装昇腾环境 <../ascend/quick_install.html>`_。


安装miniconda
----------------

.. code-block:: shell
    :linenos:

    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm -rf ~/miniconda3/miniconda.sh
    ~/miniconda3/bin/conda init bash
    ~/miniconda3/bin/conda init zsh

使用conda创建环境
---------------------

.. code-block:: shell
    :linenos:

    conda create -n python310 python=3.10.6
    conda activate python310

安装stable-diffusion-webui
----------------------------------

- 自动安装命令如下：

.. code-block:: shell
    :linenos:

    git clone --branch dev https://github.com/AUTOMATIC1111/stable-diffusion-webui.gitcd stable-diffusion-webui

    #此命令将在首次安装时自动在 Ascend 设备上安装 torch 和 torch_npu。
    ./webui.sh --listen --skip-torch-cuda-test --no-half


- 手动安装：

.. code-block:: shell
    :linenos:

    # install stable-diffusion-webui
    git clone --branch dev https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
    cd stable-diffusion-webui
    python -m venv venv
    source ./venv/bin/activate
    pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu 
    pip install torch_npu==2.1.0
    pip install https://github.com/openai/CLIP/archive/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1.zip --prefer-binary
    pip install https://github.com/mlfoundations/open_clip/archive/bb6e834e9c70d9c27d0dc3ecedeebeaeb1ffad6b.zip 
    pip install -U -I --no-deps xformers==0.0.23.post1 
    pip install install ngrok
    mkdir repositories 
    git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui-assets.git stable-diffusion-webui-assets
    git -C stable-diffusion-webui-assets checkout 6f7db241d2f8ba7457bac5ca9753331f0c266917 
    git clone https://github.com/Stability-AI/stablediffusion.git stable-diffusion-stability-ai 
    git -C stable-diffusion-stability-ai checkout cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf 
    git clone https://github.com/Stability-AI/generative-models.git generative-models 
    git -C generative-models checkout 45c443b316737a4ab6e40413d7794a7f5657c19f
    git clone https://github.com/crowsonkb/k-diffusion.git k-diffusion 
    git -C k-diffusion checkout ab527a9a6d347f364e3d185ba6d714e22d80cb3c
    git clone https://github.com/salesforce/BLIP.git BLIP 
    git -C BLIP checkout 48211a1594f1321b00f14c9f7a5b4813144b2fb9 
    pip install -r requirements.txt 
    pip install -r requirements_npu.txt 

