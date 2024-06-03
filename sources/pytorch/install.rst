安装PyTorch和PyTorch-NPU
===========================

跟随指导，安装在NPU上运行的PyTorch版本。


1. 选择需要安装的PyTorch版本
------------------------------
根据您的需求，选择合适的PyTorch和CANN版本：

.. raw:: html

    <script type="text/javascript" src="../../_static/pytorch_actions.js"></script>
    <div id="div-installation" style="">
        <div class="row">
            <div class="row-element-1" id="col-headings">
                <div class="headings-element">PyTorch版本</div>
                <div class="headings-element">PyTorch-NPU版本</div>
                <div class="headings-element">CANN-toolkit版本</div>
                <div class="headings-element">CPU架构</div>
                <div class="headings-element">安装方式</div>
            </div>
            <div class="row-element-2" id="col-values">
                <div class="row" id="row-pytorch">
                    <div class="mobile-headings">PyTorch版本</div>
                    <div class="values-element block-3 install-pytorch selected" id="pytorch-2.2.0">2.2.0</div>
                    <div class="values-element block-3 install-pytorch" id="pytorch-2.1.0">2.1.0</div>
                    <div class="values-element block-3 install-pytorch" id="pytorch-2.0.1">2.0.1</div>
                </div>
                <div class="row" id="row-pytorch_npu">
                    <div class="mobile-headings">PyTorch-NPU版本</div>
                    <div class="values-element block-1 install-pytorch_npu selected" id="pytorch_npu-version">null</div>
                </div>
                <div class="row" id="row-cann">
                    <div class="mobile-headings">CANN-toolkit版本</div>
                    <div class="values-element block-1 install-cann selected" id="cann-version">null</div>
                </div>
                <div class="row" id="row-arch">
                    <div class="mobile-headings">CPU架构</div>
                    <div class="values-element block-2 install-arch" id="arch-x86_64">x86-64</div>
                    <div class="values-element block-2 install-arch selected" id="arch-aarch64">aarch64</div>
                </div>
                 <div class="row" id="row-install_type">
                    <div class="mobile-headings">安装方式</div>
                    <div class="values-element block-2 install-type selected" id="install_type-pip">pip</div>
                    <div class="values-element block-2 install-type" id="install_type-source">源码构建</div>
                </div>
            </div>
        </div>
    </div>


请先根据上述表格建议的CANN-toolkit版本 :doc:`安装昇腾环境<../ascend/quick_install>`。


2. 安装PyTorch
----------------
.. raw:: html

    <section id="install-pytorch-pip-section">
        <p><b>使用pip安装</b></p>
                <div class="highlight">
                    <pre id="install-pytorch-pip"></pre>
                </div>
    </section>
    <div id="install-pytorch-source-section">
        <section>
            <h3>2.1 环境依赖</h3>
                <p>1. Python 3.8及以上<br>2. 支持C++17的编译器，例如clang 或者 gcc (9.4.0及以上)</p>
        </section>
        <section>
            <h3>2.2 构建</h3>
                <div class="highlight">
                    <pre id="install-pytorch-source-build"></pre>
                </div>
        </section>
    </div>

3. 安装PyTorch-NPU
--------------------

.. raw:: html

    <section id="install-pytorch_npu-pip-section">
        <p><b>使用pip安装</b></p>
                <div class="highlight">
                    <pre id="install-pytorch_npu-pip"></pre>
                </div>
    </section>
    <div id="install-pytorch_npu-source-section">
        <section>
            <h3>3.1 环境依赖</h3>
                <p>1. Python 3.8 ~ 3.10<br>2. 支持C++17的编译器，例如clang 或者 gcc (9.4.0及以上)</p>
        </section>
        <section>
            <h3>3.2 构建</h3>
                <div class="highlight">
                    <pre id="install-pytorch_npu-source-build"></pre>
                </div>
        </section>
    </div>

4. 验证安装结果
------------------

::

    import torch
    import torch_npu

    torch.npu.set_device(0)
    a = torch.randn(2,3).to('npu')
    b = torch.randn(2,3).to('npu')
    a + b