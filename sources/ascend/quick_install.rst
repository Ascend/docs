快速安装昇腾环境
================

跟随指导，在您的机器上快速安装昇腾环境。

1. 系统要求
----------------
1.1 前置检查
^^^^^^^^^^^^^
确认昇腾AI处理器已经安装妥当

.. code-block:: bash

    lspci | grep 'Processing accelerators'

确认操作系统架构及版本

.. code-block:: bash

    uname -m && cat /etc/*release

确认Python版本

.. code-block:: bash

    python --version


1.2 软件要求
^^^^^^^^^^^^^
======== ========================================
软件     版本
======== ========================================
操作系统  openEuler20.03/22.03, Ubuntu 20.04/22.04
Python   3.8, 3.9, 3.10
======== ========================================


2. 环境安装
------------------
根据您的需求，选择合适的软件包版本：

.. warning:: 

    以下文档需要使用非root用户进行安装安装

.. raw:: html

    <script type="text/javascript" src="../../_static/ascend_actions.js"></script>
    <div id="div-installation" style="">
        <div class="row">
            <div class="row-element-1" id="col-headings">
                <div class="headings-element">安装方式</div>
                <div class="headings-element">操作系统</div>
                <div class="headings-element" id="header-os_version">操作系统版本</div>
                <div class="headings-element">CPU架构</div>
                <div class="headings-element">NPU型号</div>
                <div class="headings-element">昇腾套件版本</div>
            </div>
            <div class="row-element-2" id="col-values">
                <div class="row" id="row-install_type">
                    <div class="mobile-headings">安装方式</div>
                    <div class="values-element block-2 install-type selected" id="install_type-direct">直接安装</div>
                    <div class="values-element block-2 install-type" id="install_type-docker">Docker</div>
                </div>
                <div class="row" id="row-os">
                    <div class="mobile-headings">操作系统</div>
                    <div class="values-element block-2 install-os selected" id="os-openeuler">openEuler</div>
                    <div class="values-element block-2 install-os" id="os-ubuntu">Ubuntu</div>
                </div>
                <div class="row" id="row-os_version">
                    <div class="mobile-headings">操作系统版本</div>
                </div>
                <div class="row" id="row-arch">
                    <div class="mobile-headings">CPU架构</div>
                    <div class="values-element block-2 install-arch" id="arch-x86_64">x86-64</div>
                    <div class="values-element block-2 install-arch selected" id="arch-aarch64">aarch64</div>
                </div>
                <div class="row" id="row-npu">
                    <div class="mobile-headings">NPU型号</div>
                    <div class="values-element block-2 install-npu selected" id="npu-910b">Atlas 800T A2 训练卡</div>
                    <div class="values-element block-2 install-npu" id="npu-310p">Atlas 300I Pro 推理卡</div>
                </div>
                <div class="row" id="row-ascend_version">
                    <div class="mobile-headings">昇腾套件版本</div>
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
                    <p><b>2.1.1 安装依赖</b></p>
                    <div class="highlight-default notranslate" id="install-dependencies-ubuntu">
                        <div class="highlight">
                            <pre>sudo apt-get install -y gcc g++ make cmake zlib1g zlib1g-dev openssl libsqlite3-dev libssl-dev libffi-dev unzip pciutils net-tools libblas-dev gfortran libblas3 python3-dev</pre>
                        </div>
                    </div>
                    <div class="highlight-default notranslate" id="install-dependencies-openeuler">
                        <div class="highlight">
                            <pre>sudo yum install -y gcc gcc-c++ make cmake unzip zlib-devel libffi-devel openssl-devel pciutils net-tools sqlite-devel lapack-devel gcc-gfortran python3-devel</pre>
                        </div>
                    </div>
                    <p><b>2.1.2 创建驱动运行用户</b></p>
                    <div class="admonition note">
                        <p class="admonition-title">备注</p>
                        <p>请使用命令 <code class="docutils literal notranslate">id HwHiAiUser</code> 查看用户是否存在，若存在请跳过此步骤</p>
                    </div>
                    <div class="highlight-default notranslate">
                        <div class="highlight">
                            <pre>sudo groupadd HwHiAiUser<br>sudo useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash<br>sudo usermod -aG HwHiAiUser $USER</pre>
                        </div>
                    </div>
                    <p><b>2.1.3 下载并安装</b></p>
                    <div class="highlight-default notranslate">
                        <div class="highlight">
                            <pre></pre>
                        </div>
                    </div>
                    <p>确认您的驱动是否安装成功，可以通过以下命令验证：<code class="docutils literal notranslate"><span class="pre">npu-smi</span> <span class="pre">info</span></code>，若出现以下回显信息，说明驱动安装成功。</p>
                    <div class="highlight-default notranslate">
                        <div class="highlight">
                            <pre>+-------------------------------------------------------------------------------------------+
    | npu-smi 23.0.2              Version: 23.0.2                                               |
    +----------------------+---------------+----------------------------------------------------+
    | NPU   Name           | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
    | Chip                 | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
    +======================+===============+====================================================+
    | 0     xxx            | OK            | 0.0         40                0    / 0             |
    | 0                    | 0000:C1:00.0  | 0           882  / 15169      0    / 32768         |
    +======================+===============+====================================================+
                            </pre>
                        </div>
                    </div>
                <section>
                    <h3>2.2 安装固件</h3>
                    <div class="highlight-default notranslate">
                        <div class="highlight">
                            <pre></pre>
                        </div>
                    </div>
                    <div class="admonition note">
                        <p class="admonition-title">备注</p>
                        <p>根据提示决定是否需要重启系统</p>
                    </div>
                    <p>安装固件后，若系统出现如下关键回显信息，表示固件安装成功。</p>
                    <div class="highlight-default notranslate">
                        <div class="highlight">
                            <pre>Firmware package installed successfully!</pre>
                        </div>
                    </div>
                </section>
                <section id="install_cann_section">
                    <h3>2.3 安装CANN</h3>
                        <p><b>2.3.1 安装python依赖</b></p>
                        <div class="highlight-default notranslate">
                            <div class="highlight">
                                <pre>pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple attrs numpy==1.24.0 decorator sympy cffi pyyaml pathlib2 psutil protobuf scipy requests absl-py wheel typing_extensions</pre>
                            </div>
                        </div>
                        <p><b>2.3.2 下载并安装</b></p>
                        <div class="highlight-default notranslate">
                            <div class="highlight">
                                <pre></pre>
                            </div>
                        </div>
                        <p>安装CANN-toolkit后，若系统出现以下关键回显信息，表示CANN-toolkit安装成功。</p>
                        <div class="highlight-default notranslate">
                            <div class="highlight">
                                <pre>Ascend-cann-toolkit install success.</pre>
                            </div>
                        </div>
                        <div id="install_kernel_section">
                            <p><b>2.3.3 安装算子包</b></p>
                            <div class="highlight-default notranslate">
                                <div class="highlight">
                                    <pre></pre>
                                </div>
                            </div>
                        </div>
                         <p>安装算子包后，若系统出现以下关键回显信息，表示算子包安装成功。</p>
                        <div class="highlight-default notranslate">
                            <div class="highlight">
                                <pre>Ascend-cann-kernels install success.</pre>
                            </div>
                        </div>
                        <p><b>2.3.4 设置环境变量</b></p>
                        <div class="highlight-default notranslate">
                            <div class="highlight">
                                <pre>source /usr/local/Ascend/ascend-toolkit/set_env.sh</pre>
                            </div>
                        </div>
                </section>
                <section id="use_docker_section">
                    <h3>2.3 运行Docker容器</h3>
                        <div>
                        下列命令将创建一个名为'cann-container'的Docker容器，并将设备和驱动挂载到容器中。<br><br>
                        </div>
                        <div class="highlight-default notranslate">
                            <div class="highlight">
                                <pre></pre>
                            </div>
                        </div>
                </section>
            </section>
        </div>
    </div>


3. 卸载
----------
**卸载驱动**

.. code-block:: bash

    sudo /usr/local/Ascend/firmware/script/uninstall.sh

**卸载固件**

.. code-block:: bash

    sudo /usr/local/Ascend/driver/script/uninstall.sh

**卸载CANN-toolkit**

.. code-block:: bash

    ~/Ascend/ascend-toolkit/<cann_version>/{arch}-linux/script/uninstall.sh