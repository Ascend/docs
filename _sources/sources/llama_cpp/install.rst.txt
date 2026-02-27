安装指南
==============

本教程面向使用 llama.cpp & 昇腾的开发者，帮助完成昇腾环境下 llama.cpp 的安装。

.. note::
  目前 llama.cpp 仅支持 Atlas 300T A2 型号设备

llama.cpp 下载安装
---------------------------

此处提供源码安装和 docker 两种安装方式，请按需选择:

.. raw:: html

    <script type="text/javascript" src="../../_static/llamacpp_actions.js"></script>
    <div id="div-installation" style="">
        <div class="row">
            <div class="row-element-1" id="col-values">
                    <div class="row" id="row-install_type">
                    <div class="mobile-headings">安装方式</div>
                    <div class="values-element block-3 install-type selected" id="install_type-sourceCode">源码安装</div>
                    <div class="values-element block-3 install-type" id="install_type-docker">Docker</div>
                </div>
            </div>
        </div>
    </div>

---------------

.. raw:: html

    <section id="install-llamacpp-sourceCode-section">
        <h2>使用源代码安装</h2>
        <div class="admonition note">
                <p class="admonition-title">备注</p>
                <p>请确保已经根据快速安装昇腾环境指引<a class="reference internal" href="../ascend/quick_install.html"><span class="doc">安装</span></a>了对应的CANN-toolkit版本以及相应的固件和驱动，并应用了CANN-toolkit环境变量。</p>
        </div>
        <div class="admonition warning">
            <p class="admonition-title">提示</p>
            <p>LLAMA-Factory 支持的 CANN 最低版本为 8.0.rc1。安装 CANN 时，请同时安装 Kernel 算子包。</p>
        </div>
      
        <h3>获取源代码</h3>
            <div class="code">
                <p>使用以下 git 指令获取源码</p>
                <div class="highlight">
                <pre>git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp</pre>
                </div>
            </div>

        <h3>构建 llama.cpp </h3>
            <div class="code">
                <div class="highlight">
                <pre>cmake -B build -DGGML_CANN=on -DCMAKE_BUILD_TYPE=release
    cmake --build build --config release</pre>
                </div>
            </div>

    </section>

    <div id="install-llamacpp-docker-section">
        <section>
            <h2>使用 Docker</h2>
            <div class="admonition note">
                <p class="admonition-title">备注</p>
                <p>请确保已经根据快速安装昇腾环境指引<a class="reference internal" href="../ascend/quick_install.html"><span class="doc">安装</span></a>了对应的的固件和驱动。</p>
            </div>
            <div class="admonition tip">
              <p class="admonition-title">提示</p>
              <p>更多 CANN 的基础镜像选择见<a class="reference" href="https://hub.docker.com/r/ascendai/cann/tags"><span class="doc">ascendai/cann</span></a>。</p>
            </div>           
                <p>构建 docker 镜像：</p>
                <div class="highlight">
                <pre>git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    docker build -t llama-cpp-cann -f .devops/llama-cli-cann.Dockerfile .</pre>
                </div>
                <p>找到所有卡的运行信息：</p>
                <div class="highlight">
                <pre>npu-smi info</pre>
                </div>
                <p>启动 docker 容器：</p>
                <div class="highlight">
                <pre>docker run --name llamacpp \
    --device /dev/davinci0  \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /PATH_TO_YOUR_MODELS/:/app/models \
    -it llama-cpp-cann -m /app/models/MODEL_PATH -ngl 32 \
    -p "Building a website can be done in 10 simple steps:"</pre>
            </div>
        </section>
    </div>

安装校验
-----------------

安装完成后，无任何报错信息，即为安装成功，下面为部分回显信息:

.. code-block:: shell 
    :linenos:

    [ 97%] Built target test-grammar-integration
    [ 97%] Built target llama-speculative
    [ 97%] Built target llama-perplexity
    [ 98%] Linking CXX executable ../../bin/llama-bench
    [ 98%] Linking CXX executable ../bin/test-json-schema-to-grammar
    [ 98%] Built target llama-bench
    [ 98%] Built target test-json-schema-to-grammar
    [100%] Linking CXX executable ../../bin/llama-server
    [100%] Built target llama-server

