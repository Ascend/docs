快速开始
===========

.. note::

    阅读本篇前，请确保已按照 :doc:`安装指南 <./install>` 准备好昇腾环境及 TritonParse！

本指南将介绍使用 TritonParse 分析 Triton 内核编译过程的完整工作流程。

概述
--------

TritonParse 的工作流程包含三个主要步骤：

#. **生成跟踪文件** — 捕获 Triton 编译事件
#. **解析跟踪文件** — 将原始日志处理为结构化格式
#. **分析结果** — 通过 Web 界面进行可视化探索

标准设置模式
-----------------

所有 TritonParse 工作流程都遵循以下模式：

初始化日志

^^^^^^^^^^^^

.. code-block:: python
    :number-lines:

    import tritonparse.structured_logging

    log_path = "./logs/"
    tritonparse.structured_logging.init(log_path, enable_trace_launch=True)

解析跟踪文件

^^^^^^^^^^^^^^^^

.. code-block:: python
    :number-lines:

    import tritonparse.parse.utils

    tritonparse.parse.utils.unified_parse(
        source=log_path,
        out="./parsed_output",
        overwrite=True
    )

也可以使用命令行：

.. code-block:: shell
    :number-lines:

    tritonparseoss parse ./logs/ --out ./parsed_output

第1步：生成 Triton 跟踪文件
-------------------------------

示例：完整的 Triton 内核
^^^^^^^^^^^^^^^^^^^^^^^^^^^

以下是一个在昇腾 NPU 上跟踪 Triton 内核的完整示例（改编自 NPU 测试用例）：

.. code-block:: python
    :number-lines:

    import torch
    import triton
    import triton.language as tl
    import tritonparse.structured_logging
    import tritonparse.parse.utils

    # 初始化日志记录
    log_path = "./logs/"
    tritonparse.structured_logging.init(log_path, enable_trace_launch=True)

    @triton.jit
    def triton_add(in_ptr0, in_ptr1, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
        """NPU版本的向量加法内核"""
        offset = tl.program_id(0) * XBLOCK
        base1 = tl.arange(0, XBLOCK_SUB)
        loops1: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB

        for loop1 in range(loops1):
            x0 = offset + (loop1 * XBLOCK_SUB) + base1
            tmp0 = tl.load(in_ptr0 + (x0), None)
            tmp1 = tl.load(in_ptr1 + (x0), None)
            tmp2 = tmp0 + tmp1
            tl.store(out_ptr0 + (x0), tmp2, None)

    def tensor_add_npu(a, b, ncore=2, xblock=1024, xblock_sub=1024):
        """NPU版本的张量加法包装函数"""
        c = torch.empty_like(a, device="npu")
        triton_add[ncore, 1, 1](a, b, c, xblock, xblock_sub)
        return c

    # 创建输入数据并运行
    if __name__ == "__main__":
        dtype = torch.float32
        shape = (2, 4096, 8)
        ncore = 2
        xblock = 32768
        xblock_sub = 1024

        a = torch.randn(shape, dtype=dtype, device="npu")
        b = torch.randn(shape, dtype=dtype, device="npu")
        c_npu = tensor_add_npu(a, b, ncore, xblock, xblock_sub)

        # 解析生成的日志
        tritonparse.parse.utils.unified_parse(source=log_path, out="./parsed_output", overwrite=True)


环境变量
^^^^^^^^^^^^

通过以下环境变量配置 TritonParse 的行为：

.. list-table::
   :header-rows: 1
   :widths: 40 40 20

   * - 变量
     - 描述
     - 示例
   * - ``TRITON_TRACE``
     - 跟踪输出目录
     - ``"./logs/"``
   * - ``TRITON_TRACE_LAUNCH``
     - 启用启动跟踪（``"1"`` 或 ``"0"``）
     - ``"1"``
   * - ``TORCHINDUCTOR_RUN_JIT_POST_COMPILE_HOOK``
     - ``torch.compile`` 内核启动跟踪 **必需**
     - ``"1"``
   * - ``TRITONPARSE_MORE_TENSOR_INFORMATION``
     - 收集张量统计信息（min/max/mean/std）
     - ``"1"``
   * - ``TRITONPARSE_SAVE_TENSOR_BLOBS``
     - 保存实际张量数据为 blob 文件
     - ``"1"``
   * - ``TRITONPARSE_DEBUG``
     - 启用调试日志
     - ``"1"``
   * - ``TRITON_TRACE_COMPRESSION``
     - 压缩格式（``"none"``、``"gzip"``、``"clp"``）
     - ``"gzip"``
   * - ``TRITONPARSE_KERNEL_ALLOWLIST``
     - 过滤特定内核（逗号分隔的模式）
     - ``"my_kernel*,important_*"``
   * - ``TORCHINDUCTOR_FX_GRAPH_CACHE``
     - 禁用 FX graph 缓存（用于测试）
     - ``"0"``

使用方式：

.. code-block:: shell
    :number-lines:

    export TRITON_TRACE="./logs/"
    export TRITON_TRACE_LAUNCH="1"
    # torch.compile 内核必需
    export TORCHINDUCTOR_RUN_JIT_POST_COMPILE_HOOK="1"
    # 可选：收集张量统计信息以生成更好的复现脚本
    export TRITONPARSE_MORE_TENSOR_INFORMATION="1"
    export TORCHINDUCTOR_FX_GRAPH_CACHE=0

    python your_script.py

预期输出：

.. code-block:: shell
    :number-lines:

    ================================================================================
    TRITONPARSE PARSING RESULTS
    ================================================================================
    Parsed files directory: /path/to/parsed_output
    Total files generated: 2

    Generated files:
    --------------------------------------------------
       1. dedicated_log_triton_trace_xxx.ndjson.gz (7.2KB)
       2. log_file_list.json (181B)
    ================================================================================
    Parsing completed successfully!
    ================================================================================

第2步：解析跟踪文件
-----------------------

Python API
^^^^^^^^^^

.. code-block:: python
    :number-lines:

    import tritonparse.parse.utils

    # 基础解析
    tritonparse.parse.utils.unified_parse(
        source="./logs/",           # 原始日志输入目录
        out="./parsed_output",      # 处理后文件输出目录
        overwrite=True              # 覆盖已有输出
    )

    # 高级选项
    tritonparse.parse.utils.unified_parse(
        source="./logs/",
        out="./parsed_output",
        overwrite=True,
        rank=0,                     # 分析特定 rank（多 NPU 场景）
        all_ranks=False,            # 或分析所有 rank
        verbose=True                # 启用详细日志
    )

命令行接口
^^^^^^^^^^

.. code-block:: shell
    :number-lines:

    # 基础用法
    tritonparseoss parse ./logs/ --out ./parsed_output

    # 也可以使用 python -m
    python -m tritonparse parse ./logs/ --out ./parsed_output

    # 带选项
    tritonparseoss parse ./logs/ --out ./parsed_output --overwrite --verbose

    # 多 NPU：解析特定 rank
    tritonparseoss parse ./logs/ --out ./parsed_output --rank 0

    # 多 NPU：解析所有 rank
    tritonparseoss parse ./logs/ --out ./parsed_output --all-ranks

第3步：通过 Web 界面分析
-----------------------------

方式 A：在线界面（推荐）
^^^^^^^^^^^^^^^^^^^^^^^^^^

#. **访问在线工具**：`https://meta-pytorch.org/tritonparse/ <https://meta-pytorch.org/tritonparse/>`_

#. **加载跟踪文件**：

   - 点击 "Browse Files" 或拖拽上传
   - 选择 ``parsed_output`` 目录中的 ``.gz`` 文件
   - 或选择 ``logs`` 目录中的 ``.ndjson`` 文件

#. **探索可视化**：

   - **内核概览选项卡**：内核元数据、调用栈、IR 链接
   - **IR 代码视图选项卡**：并排 IR 查看与行映射

方式 B：本地开发界面
^^^^^^^^^^^^^^^^^^^^^^

适用于贡献者或自定义部署：

.. code-block:: shell
    :number-lines:

    cd website
    npm install
    npm run dev

访问地址：``http://localhost:5173``

支持的文件格式
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - 格式
     - 描述
     - 源码映射
     - 推荐
   * - ``.gz``
     - 压缩的解析后跟踪文件
     - 支持
     - 推荐
   * - ``.ndjson``
     - 原始跟踪日志
     - 不支持
     - 仅基础使用

.. note::

    ``.ndjson`` 文件不包含 IR 阶段之间的源码映射和启动差异信息。请始终使用 ``.gz`` 文件以获得完整功能。

理解分析结果
-----------------

内核概览
^^^^^^^^

概览页面显示：

- **内核信息**：名称、哈希、Grid/Block 大小
- **编译元数据**：设备、编译时间、内存使用
- **调用栈**：触发编译的 Python 源代码
- **IR 导航**：链接到不同的 IR 表示
- **启动差异**：同一内核在不同启动之间变化的启动参数

IR 代码视图
^^^^^^^^^^^^

IR 代码视图提供：

- **并排 IR 查看**：比较不同编译阶段
- **同步高亮**：点击一行查看其他 IR 中的对应行
- **源码映射**：跟踪编译管道中的变换

文件差异视图
^^^^^^^^^^^^

并排比较来自 **两个不同跟踪文件** 的内核：

- **跨跟踪比较**：验证优化、跟踪内核演进、调试差异
- **灵活模式**：单 IR 聚焦或所有 IR 同时比较
- **可自定义差异**：忽略空白、词级/行级、上下文控制
- **URL 可分享**：``?view=file_diff&json_url=trace1.gz&json_b_url=trace2.gz``

IR 阶段说明
^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - 阶段
     - 描述
   * - **TTIR**
     - Triton IR - 语言级操作
   * - **TTAdapter**
     - Triton Adapter IR - 适配层操作
   * - **MLIRBC**
     - MLIR Bytecode - 中间表示字节码
   * - **BCMLIR**
     - BC MLIR - 后端编译中间表示
   * - **NPUBIN**
     - NPU Binary - NPU 二进制产物

启动分析
-----------

TritonParse 可以分析内核启动参数，识别同一内核在不同启动之间的差异和共性。这对于理解动态形状或其他因素如何影响内核执行非常有用。

工作原理
^^^^^^^^^^^^

#. **启用启动跟踪**：在跟踪生成步骤中通过传递 ``enable_trace_launch=True`` 给 ``tritonparse.structured_logging.init()`` 来启用。
#. **解析**：在解析步骤（``tritonparse.parse.utils.unified_parse``）中，TritonParse 会自动为每个内核分组所有启动。
#. **启动差异事件**：为每个内核生成一个类型为 ``launch_diff`` 的事件，包含：

   - ``total_launches``：内核启动的总次数
   - ``diffs``：显示哪些启动参数（如 ``grid_x``、``grid_y``）在不同启动之间发生了变化及其不同值
   - ``sames``：显示在所有启动中保持不变的启动参数
   - ``launch_index_map``：从启动索引到跟踪文件中原始行号的映射

示例 ``launch_diff`` 事件：

.. code-block:: json
    :number-lines:

    {
      "event_type": "launch_diff",
      "hash": "...",
      "name": "triton_kernel_name",
      "total_launches": 10,
      "launch_index_map": { "0": 15, "1": 25 },
      "diffs": {
        "grid_x": [1024, 2048]
      },
      "sames": {
        "grid_y": 1,
        "grid_z": 1,
        "stream": 7
      }
    }

复现脚本 - 生成独立内核脚本
-------------------------------

TritonParse 可以自动生成独立的 Python 脚本来复现特定的内核执行。适用于调试、分享测试用例和隔离性能问题。

快速开始
^^^^^^^^

**命令行：**

.. code-block:: shell
    :number-lines:

    # 为第一个启动事件生成复现脚本
    tritonparseoss reproduce ./parsed_output/trace.ndjson --line 1 --out-dir repro_output

    # 使用压缩文件
    tritonparseoss reproduce ./parsed_output/trace.ndjson.gz --line 5 --out-dir my_repro

    # 使用自定义模板
    tritonparseoss reproduce trace.ndjson --line 1 --template /path/to/my_template.py

**Python API：**

.. code-block:: python
    :number-lines:

    from tritonparse.reproducer.orchestrator import reproduce

    result = reproduce(
        input_path="./parsed_output/trace.ndjson",
        line_index=0,                    # 启动事件索引（从 0 开始）
        out_dir="repro_output",
        template="example"               # 内置模板
    )

    print(f"Script: {result['repro_script']}")
    print(f"Context: {result['repro_context']}")

生成的文件
^^^^^^^^^^

::

    repro_output/<kernel_name>/
    ├── repro_<timestamp>.py              # 独立可执行脚本
    ├── repro_context_<timestamp>.json    # 内核元数据和参数
    └── <hash>.bin                        # 张量 blob（如果在跟踪时启用）

参数说明
^^^^^^^^

.. list-table::
   :header-rows: 1

   * - 参数
     - 描述
     - 默认值
   * - ``input``
     - NDJSON 跟踪文件路径（``.ndjson`` 或 ``.ndjson.gz``）
     - 必填
   * - ``--line``
     - 启动事件的行索引（从 0 开始）
     - ``0``
   * - ``--out-dir``
     - 输出目录
     - ``repro_output/<kernel>/``
   * - ``--template``
     - 模板名称或路径
     - ``example``

张量数据策略
^^^^^^^^^^^^

复现脚本使用以下策略之一重建张量：

Blob 文件（最高保真度）
""""""""""""""""""""""""

.. code-block:: python
    :number-lines:

    # 在跟踪时启用
    tritonparse.structured_logging.init(
        "./logs/",
        enable_trace_launch=True,
        enable_tensor_blob_storage=True  # 保存实际张量数据
    )

统计重建（良好近似）
""""""""""""""""""""

- 使用保存的 mean、std、min、max 生成类似数据
- 匹配原始的 shape、dtype、device

随机数据（回退方案）
""""""""""""""""""""

- 仅匹配 shape 和 dtype 的随机生成

初始化方法对比
-----------------

TritonParse 支持多种初始化方法：

方式一：直接初始化（推荐）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
    :number-lines:

    tritonparse.structured_logging.init(
        trace_folder="./logs/",
        enable_trace_launch=True,
        enable_more_tensor_information=True,  # 收集张量统计
    )

完整参数：

.. list-table::
   :header-rows: 1

   * - 参数
     - 类型
     - 默认值
     - 描述
   * - ``trace_folder``
     - ``str``
     - ``None``
     - 跟踪文件目录
   * - ``enable_trace_launch``
     - ``bool``
     - ``False``
     - 启用所有启动的事件跟踪
   * - ``enable_trace_launch_within_profiling``
     - ``bool``
     - ``False``
     - 仅在 ``torch.profiler`` RECORD 阶段启用启动跟踪
   * - ``enable_more_tensor_information``
     - ``bool``
     - ``False``
     - 收集张量统计信息（min/max/mean/std）
   * - ``enable_tensor_blob_storage``
     - ``bool``
     - ``False``
     - 保存实际张量数据为 blob 文件
   * - ``compression``
     - ``str``
     - ``None``
     - 压缩格式（``"none"``、``"gzip"``、``"clp"``）

方式二：环境变量
^^^^^^^^^^^^^^^^

.. note::

    即使使用环境变量，也必须在代码中调用 ``tritonparse.structured_logging.init()`` 来激活跟踪。

.. code-block:: python
    :number-lines:

    import os
    os.environ["TRITON_TRACE"] = "./logs/"
    os.environ["TRITON_TRACE_LAUNCH"] = "1"

    tritonparse.structured_logging.init()  # 必需

或在 shell 中：

.. code-block:: shell
    :number-lines:

    export TRITON_TRACE="./logs/"
    export TRITON_TRACE_LAUNCH="1"
    python my_script.py

方式三：上下文管理器（TritonParseManager）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

适用于简化工作流和自动清理：

.. code-block:: python
    :number-lines:

    from tritonparse.context_manager import TritonParseManager

    with TritonParseManager(
        enable_trace_launch=True,
        out="./parsed_output",
        overwrite=True,
    ) as manager:
        # 在此运行内核代码
        result = my_kernel(input_tensor)

    # 上下文退出时自动解析日志
    print(f"Parsed output: {manager.output_link}")

方法对比：

.. list-table::
   :header-rows: 1

   * - 方法
     - 适用场景
     - 优点
     - 缺点
   * - ``init(path, ...)``
     - 代码中直接控制
     - 显式，所有选项可用
     - 需手动解析
   * - 环境变量
     - 基于环境的配置
     - 灵活，CI/CD 友好
     - 需要环境配置
   * - ``TritonParseManager``
     - 快速实验
     - 自动清理，简单
     - 控制力较弱
