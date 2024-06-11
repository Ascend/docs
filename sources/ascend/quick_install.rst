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
操作系统  openEuler20.03/22.03, Ubuntu 20.04/24.04
Python   3.8, 3.9, 3.10
======== ========================================


2. 环境安装
------------------
根据您的需求，选择合适的软件包版本：

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
                    <div class="values-element block-2 install-npu selected" id="npu-910b">Atlas 300T A2 训练卡</div>
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
                    <p><b>安装依赖</b></p>
                    <div class="highlight-default notranslate" id="install-dependencies-ubuntu">
                        <div class="highlight">
                            <pre>sudo apt-get install -y gcc g++ make cmake zlib1g zlib1g-dev openssl libsqlite3-dev libssl-dev libffi-dev unzip pciutils net-tools libblas-dev gfortran libblas3</pre>
                        </div>
                    </div>
                    <div class="highlight-default notranslate" id="install-dependencies-openeuler">
                        <div class="highlight">
                            <pre>sudo yum install -y gcc gcc-c++ make cmake unzip zlib-devel libffi-devel openssl-devel pciutils net-tools sqlite-devel lapack-devel gcc-gfortran</pre>
                        </div>
                    </div>
                    <p><b>创建驱动运行用户</b></p>
                    <div class="highlight-default notranslate">
                        <div class="highlight">
                            <pre>sudo groupadd -g HwHiAiUser<br>sudo useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash<br>sudo usermod -aG HwHiAiUser $USER</pre>
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
                <section id="install_cann_section">
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
                        <div id="install_kernel_section">
                            <p><b>安装算子包（可选）</b></p>
                            <div class="highlight-default notranslate">
                                <div class="highlight">
                                    <pre id="install_kernel"></pre>
                                </div>
                            </div>
                        </div>
                        <p><b>设置环境变量</b></p>
                        <div class="highlight-default notranslate">
                            <div class="highlight">
                                <pre>echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> ~/.bashrc<br>source ~/.bashrc</pre>
                            </div>
                        </div>
                </section>
                <section id="use_docker_section">
                    <h3>2.3 使用Docker镜像</h3>
                        <div class="highlight-default notranslate">
                            <div class="highlight">
                                <pre id="use_docker"></pre>
                            </div>
                        </div>
                </section>
            </section>
        </div>
    </div>


3. 验证安装（可选）
---------------------
**验证驱动**

确认您的驱动是否安装成功，可以通过以下命令验证：``npu-smi info``
，若出现以下回显信息，说明驱动安装成功。
::

    +-------------------------------------------------------------------------------------------+
    | npu-smi 23.0.2              Version: 23.0.2                                               |
    +----------------------+---------------+----------------------------------------------------+
    | NPU   Name           | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
    | Chip                 | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
    +======================+===============+====================================================+
    | 0     xxx            | OK            | 0.0         40                0    / 0             |
    | 0                    | 0000:C1:00.0  | 0           882  / 15169      0    / 32768         |
    +======================+===============+====================================================+
    | 1     xxx            | OK            | 0.0         35                0    / 0             |
    | 0                    | 0000:81:00.0  | 0           1603 / 15169      0    / 32768         |
    +======================+===============+====================================================+
    | 2     xxx            | OK            | 0.0         32                0    / 0             |
    | 0                    | 0000:41:00.0  | 0           2440 / 15169      0    / 32768         |
    +======================+===============+====================================================+
    | 3     xxx            | OK            | 0.0         40                0    / 0             |
    | 0                    | 0000:01:00.0  | 0           1014 / 15071      0    / 32768         |
    +======================+===============+====================================================+
    | 4     xxx            | OK            | 0.0         39                0    / 0             |
    | 0                    | 0000:C2:00.0  | 0           457  / 15169      0    / 32768         |
    +======================+===============+====================================================+
    | 5     xxx            | OK            | 0.0         33                0    / 0             |
    | 0                    | 0000:82:00.0  | 0           523  / 15169      0    / 32768         |
    +======================+===============+====================================================+
    | 6     xxx            | OK            | 0.0         31                0    / 0             |
    | 0                    | 0000:42:00.0  | 0           2182 / 15169      0    / 32768         |
    +======================+===============+====================================================+
    | 7     xxx            | OK            | 0.0         39                0    / 0             |
    | 0                    | 0000:02:00.0  | 0           2771 / 15071      0    / 32768         |
    +======================+===============+====================================================+

**验证固件**

安装固件后，若系统出现如下关键回显信息，表示固件安装成功。
::

    Firmware package installed successfully!

**验证CANN-toolkit**

安装CANN-toolkit后，若系统出现以下关键回显信息，表示CANN-toolkit安装成功。
::

    [INFO] Ascend-cann-toolkit install success


4. 卸载
----------
**卸载驱动**
::

    sudo /usr/local/Ascend/firmware/script/uninstall.sh

**卸载固件**
::

    sudo /usr/local/Ascend/driver/script/uninstall.sh

**卸载CANN-toolkit**
::

    <path>/ascend-toolkit/<cann_version>/{arch}-linux/script/uninstall.sh