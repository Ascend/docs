快速安装昇腾环境
================

跟随指导，在您的机器上快速安装昇腾环境。

1. 系统要求
----------------
1.1 前置检查
^^^^^^^^^^^^^
确认昇腾AI处理器已经安装妥当
::

    lspci | grep 'Processing accelerators'

确认操作系统架构及版本
::

    uname -m && cat /etc/*release


1.2 软件要求
^^^^^^^^^^^^^
======== ========================================
软件     版本
======== ========================================
操作系统  OpenEuler20.03/22.03, Ubuntu 20.04/24.04
Python   3.8, 3.9, 3.10
======== ========================================


2. 环境安装
------------------
根据您的需求，选择合适的软件包版本：

.. raw:: html

    <div id="div-installation" style="">
        <div class="row">
            <div class="row-element-1" id="col-headings">
                <div class="headings-element">操作系统</div>
                <div class="headings-element">CPU架构</div>
                <div class="headings-element">NPU型号</div>
                <div class="headings-element">版本选择</div>
            </div>
            <div class="row-element-2" id="col-values">
                <div class="row" id="row-os">
                    <div class="mobile-headings">操作系统</div>
                    <div class="values-element block-2 install-os selected" id="os-ubuntu">Ubuntu</div>
                    <div class="values-element block-2 install-os" id="os-openeuler">openEuler</div>
                </div>
                <div class="row" id="row-arch">
                    <div class="mobile-headings">CPU架构</div>
                    <div class="values-element block-2 install-arch" id="arch-x86_64">x86-64</div>
                    <div class="values-element block-2 install-arch selected" id="arch-aarch64">aarch64</div>
                </div>
                <div class="row" id="row-npu">
                    <div class="mobile-headings">NPU型号</div>
                    <div class="values-element block-1 install-npu selected" id="npu-910b">Atlas 300T A2 训练卡</div>
                </div>
                <div class="row" id="row-version">
                    <div class="mobile-headings">版本选择</div>
                    <select class="values-element block-3 install-package" id="cann-version">
                        <option value="na">Select CANN Version</option>
                    </select>
                    <div class="values-element block-3 install-package" id="driver-version">Driver</div>
                    <div class="values-element block-3 install-package" id="firmware-version">Firmware</div>
                </div>
            </div>
        </div>
        <div id="install-instructions" style="display:none;">
            <section>
                <h3>2.1 安装驱动</h3>
                    <p><b>安装依赖</b></p>
                    <div class="highlight-default notranslate" id="install-dependencies-ubuntu">
                        <div class="highlight">
                            <pre>sudo apt-get install -y gcc g++ make cmake zlib1g zlib1g-dev openssl libsqlite3-dev libssl-dev libffi-dev unzip pciutils net-tools libblas-dev gfortran libblas3</pre>
                        </div>
                    </div>
                    <div class="highlight-default notranslate" id="install-dependencies-openeuler">
                        <div class="highlight">
                            <pre>yum install -y gcc gcc-c++ make cmake unzip zlib-devel libffi-devel openssl-devel pciutils net-tools sqlite-devel lapack-devel gcc-gfortran</pre>
                        </div>
                    </div>
                    <p><b>创建驱动运行用户</b></p>
                    <div class="highlight-default notranslate">
                        <div class="highlight">
                            <pre>groupadd -g HwHiAiUser<br>useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash</pre>
                        </div>
                    </div>
                    <p><b>下载并安装</b></p>
                    <div class="highlight-default notranslate">
                        <div class="highlight">
                            <pre id="install_drvier"></pre>
                        </div>
                    </div>
                <section>
                    <h3>2.2 安装固件</h3>
                    <div class="highlight-default notranslate">
                        <div class="highlight">
                            <pre id="install_firmware"></pre>
                        </div>
                    </div>
                </section>
                <section>
                    <h3>2.3 安装CANN</h3>
                        <p><b>安装python依赖</b></p>
                        <div class="highlight-default notranslate">
                            <div class="highlight">
                                <pre>pip3 install attrs numpy decorator sympy cffi pyyaml pathlib2 psutil protobuf scipy requests absl-py wheel typing_extensions</pre>
                            </div>
                        </div>
                        <p><b>下载并安装</b></p>
                        <div class="highlight-default notranslate">
                            <div class="highlight">
                                <pre id="install_cann"></pre>
                            </div>
                        </div>
                </section>
                <section>
                    <h3>2.4 设置环境变量</h3>
                    <div class="highlight-default notranslate">
                        <div class="highlight">
                            <pre>echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> ~/.bashrc<br>source ~/.bashrc</pre>
                        </div>
                    </div>
                </section>
            </section>
        </div>
    </div>


3. 卸载
----------
**卸载驱动**
::

    sudo /usr/local/Ascend/firmware/script/uninstall.sh

**卸载固件**
::

    sudo /usr/local/Ascend/driver/script/uninstall.sh

**卸载CANN-toolkit**
::

    #root用户安装
    sudo /usr/local/Ascend/ascend-toolkit/<cann_version>/{arch}-linux/script/uninstall.sh

    #非root用户安装
    ~/Ascend/ascend-toolkit/<cann_version>/{arch}-linux/script/uninstall.sh