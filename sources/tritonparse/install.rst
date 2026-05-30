安装指南
===========

本文将介绍如何在昇腾环境下安装 TritonParse，帮助开发者完成 Triton 内核编译链路的可视化与分析工具部署。



昇腾环境安装
----------------

在安装 TritonParse 之前，请根据已有昇腾产品型号及CPU架构等按照 `快速安装昇腾环境指引 <../ascend/quick_install>`_ 进行昇腾环境安装。

- Python >= 3.10
- Operating System: Linux, macOS, or Windows (with WSL recommended)
- Triton > 3.3.1
- Node.js >= 22.0.0 (for website development only)

.. note::

    TritonParse 的核心功能是对 Triton 内核进行编译跟踪与结构化日志记录，因此 Triton 是必须安装的依赖项。

tritonparse安装
--------------------

请根据您的需要选择安装方式：

方式一：通过 PyPI 安装（推荐）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell
    :number-lines:

    # 安装 nightly 版本（推荐，包含最新功能）
    pip install -U --pre tritonparse

    # 或安装稳定版
    pip install tritonparse

方式二：从源码安装
^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell
    :number-lines:

    # 克隆并安装
    git clone https://github.com/meta-pytorch/tritonparse.git
    cd tritonparse
    pip install -e .

    # 或直接从 GitHub 安装，无需克隆
    pip install git+https://github.com/meta-pytorch/tritonparse.git

贡献者的额外配置：

.. code-block:: shell
    :number-lines:

    # Python 开发：安装格式化工具（black, usort, ruff）
    make install-dev

    # 网站开发：安装 Node.js 依赖（需要 Node.js >= 22.0.0）
    cd website && npm install
    npm run dev  # 启动开发服务器 http://localhost:5173

验证安装
------------

测试 TritonParse 是否正常工作：

.. code-block:: shell
    :number-lines:

    # 进入测试目录
    cd tests

    # 运行示例测试
    TORCHINDUCTOR_FX_GRAPH_CACHE=0 python test_add.py

预期输出：

.. code-block:: shell
    :number-lines:

    Triton kernel executed successfully
    Torch compiled function executed successfully
    ================================================================================
    TRITONPARSE PARSING RESULTS
    ================================================================================
    Parsed files directory: /path/to/tritonparse/tests/parsed_output
    Total files generated: 2
    ...
    Parsing completed successfully!
    ================================================================================

