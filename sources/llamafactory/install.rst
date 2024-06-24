安装指南
==============

本教程面向使用 LLAMA-Factory & 昇腾的开发者，帮助完成昇腾环境下 LLaMA-Factory 的安装。此处提供 docker 和 pip 两种安装方式:

.. raw:: html

    <script type="text/javascript" src="../../_static/llamafactory_actions.js"></script>
    <div id="div-installation" style="">
        <div class="row">
            <div class="row-element-1" id="col-values">
                 <div class="row" id="row-install_type">
                    <div class="mobile-headings">安装方式</div>
                    <div class="values-element block-3 install-type selected" id="install_type-docker">Docker</div>
                    <div class="values-element block-3 install-type" id="install_type-pip">pip</div>
                </div>
            </div>
        </div>
    </div>


.. raw:: html

    <section id="install-llmf-pip-section">
        <h2>使用 pip</h2>
        <div class="admonition note">
                <p class="admonition-title">备注</p>
                <p>请确保已经根据快速安装昇腾环境指引<a class="reference internal" href="../ascend/quick_install.html"><span class="doc">安装</span></a>了对应的CANN-toolkit版本以及相应的固件和驱动，并应用了CANN-toolkit环境变量。</p>
        </div>
        <div class="admonition warning">
            <p class="admonition-title">警告</p>
            <p>LLAMA-Factory 支持的 CANN 最低版本为 8.0.rc1。安装 CANN 时，请同时安装 Kernel 算子包。</p>
        </div>
        <h3>Python 环境创建</h3>
            <div class="code">
                <p>创建并激活 Python 环境：</p>
                <div class="highlight">
                  <pre>conda create -n <your_env_name> python=3.10</pre>
                  <pre>conda activate <your_env_name></pre>
                </div>
            </div>
        <h3>LLaMA-Factory 安装</h3>
            <div class="code">
                <p>使用以下指令安装带有 torch-npu 的 LLaMA-Factory：</p>
                <div class="highlight">
                  <pre>pip install -e .[torch_npu,metrics]</pre>
                </div>
            </div>
    </section>

    <div id="install-llmf-docker-section">
        <section>
            <h2>使用 Docker</h2>
            <div class="admonition note">
                <p class="admonition-title">备注</p>
                <p>请确保已经根据快速安装昇腾环境指引<a class="reference internal" href="../ascend/quick_install.html"><span class="doc">安装</span></a>了对应的的固件和驱动。</p>
            </div>
            <div class="admonition tip">
              <p class="admonition-title">提示</p>
              <p>默认镜像为<a class="reference" href="https://hub.docker.com/layers/cosdt/cann/8.0.rc1-910b-ubuntu22.04/images/sha256-29ef8aacf6b2babd292f06f00b9190c212e7c79a947411e213135e4d41a178a9?context=explore"><span class="doc">cosdt/cann:8.0.rc1-910b-ubuntu22.04</span></a>。更多选择见<a class="reference" href="https://hub.docker.com/r/cosdt/cann/tags"><span class="doc">cosdt/cann</span></a>。</p>
            </div>
            <div class="code">
                <p>进入存放 Dockerfile 及 docker-compose.yaml 的 docker-npu 目录：</p>
                <div class="highlight">
                  <pre>cd docker/docker-npu</pre>
                </div>
            </div>
            <div class="code">
                <p>使用以下指令构建及启动 docker 容器：</p>
                <div class="highlight">
                  <pre>docker build -f ./Dockerfile --build-arg INSTALL_DEEPSPEED=false --build-arg PIP_INDEX=https://pypi.org/simple -t llamafactory:latest </pre>
                </div>
            </div>
        </section>
    </div>



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
  :linenos:
  
  pip uninstall llamafactory

附录
-------------

.. warning::

  使用 pip 安装请忽略本附录文件。

Dockerfile 及 docker compose 配置文件：

Dockerfile:

.. literalinclude:: ./Dockerfile
    :language: docker
    :linenos:

docker-compose.yaml:

.. literalinclude:: ./docker-compose.yaml
    :language: yaml
    :linenos:

    
可通过修改或添加 docker-compose.yaml 文件中 devices 来改变指定 NPU 卡或指定多卡 NPU。如下所示为指定 0~3 四卡：

.. code-block:: yaml
    :linenos:

    devices:
      - /dev/davinci0
      - /dev/davinci1
      - /dev/davinci2
      - /dev/davinci3
      - /dev/davinci_manager
      - /dev/devmm_svm
      - /dev/hisi_hdc
