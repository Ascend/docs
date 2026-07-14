快速开始
=======

本篇指南介绍了如何在 Ascend NPU 环境下快速安装和使用 Ray。借助于 Ray 的分布式能力，您可以更高效地管理集群资源和分布式计算任务。

安装说明
--------

在 Ascend 上安装 Ray 非常简单，不需要复杂的系统依赖配置。我们推荐以下安装方式：

.. code-block:: bash

    pip install ray>=2.46.0

启动与作业运行
--------------

Ray 提供了一套简单易用的命令行工具，方便您在单机或多机环境下快速拉起守护进程并提交任务。

1. 启动 Ray 节点
^^^^^^^^^^^^^^^^

- **单机启动（默认作为主节点）**：

  .. code-block:: bash

      ray start --head

- **多机启动（工作节点加入主节点）**：

  在子节点上执行以下命令（替换 ``IP`` 为主节点的真实 IP）：

  .. code-block:: bash

      ray start --address='<head_node_ip>:6379'

2. 提交与管理作业 (Job)
^^^^^^^^^^^^^^^^^^^^^^^

推荐使用 Ray Job 的方式提交任务。 **这种方式提交的脚本会默认在集群后台运行** ，即便您的终端意外关闭或 SSH 断开，任务也不会中断，极大地提升了稳定性。

- **提交任务**：

  .. code-block:: bash

      ray job submit --working-dir . -- python your_script.py

  *(提交后，系统会返回一个* ``job_id`` *，例如：* ``raysubmit_xxxx`` *)*

- **查看任务列表**：

  .. code-block:: bash

      ray job list

- **停止运行中的任务**：

  .. code-block:: bash

      ray job stop <job_id>

.. note::
    **💡 框架集成运行方式补充说明：**

    如果您使用的是已经深度集成 Ray 的上层框架（例如某些大模型微调框架会自己在代码中拉起 Ray 集群），您完全可以继续使用框架原有的命令（例如直接运行 ``python scripts.py`` 或 ``bash run.sh`` 等）。

    **区别在于** ：通过这种原生方式直接运行的进程，其日志 **大部分会实时输出在当前终端，且进程不处于后台** 。如果您关闭终端进程就会被 kill。您可以根据自己的使用习惯（后台挂机 vs 实时监控）来选择启动方式。

    **结合** ``ray job submit`` **实现后台运行** ：

    如果您既想使用其他框架的原生启动脚本，又希望让其在后台稳定运行，可以将其与 ``ray job submit`` 结合使用。只要集群提前通过 ``ray start`` 拉起，您就可以如下提交框架脚本：

    .. code-block:: bash

        ray job submit --working-dir . -- bash run_framework.sh

Ray 集群环境变量配置
--------------------

Ray 的环境变量配置有一个 **核心原则** ： **所有影响集群行为的环境变量必须在 Ray 节点启动（ray start）之前声明，或者在提交 Job 时通过专用参数传入。**

如果在 ``ray start`` 之后才执行 ``export`` ，这些环境变量对已经运行的 Ray 进程（包括由 Ray 调度的 Worker 进程）是 **无效的** 。

以下以 ``RAY_DEDUP_LOGS`` （日志去重功能）为例，展示正确的环境变量配置方式：

**示例场景：禁用日志去重**

Ray 默认会折叠大量重复的日志。在调试 NPU 问题时，为了防止关键的硬件报错信息被“去重”隐藏，我们通常需要禁用此功能。

**正确配置方式 1：集群启动前全局设定（推荐）**

在执行 ``ray start`` 之前设置环境变量，该变量会对该节点上后续启动的所有 Ray 进程生效。

.. code-block:: bash

    export RAY_DEDUP_LOGS=0
    ray start --head

**正确配置方式 2：使用** ``ray job submit`` **针对特定作业生效**

如果您不想重启集群，也可以在提交任务时通过 ``--runtime-env-json`` 将环境变量注入到该任务的运行环境中。

.. code-block:: bash

    # 举例：仅对本次提交的脚本禁用日志去重
    ray job submit --runtime-env-json '{"env_vars": {"RAY_DEDUP_LOGS": "0"}}' -- python your_script.py

ray start 常用参数说明
----------------------

虽然绝大部分用户只需使用基础的 ``ray start --head`` 就能跑通全流程，但如果您在复杂组网或者多用户共享服务器的情况下使用 Ray，可以了解并在第二步的启动流程中加入这些进阶参数：

- ``--port``
  指定 Ray 头部节点通信的端口（默认值为 ``6379``）。如果在同一台宿主机上有多个人需要各自启动 Ray 集群，可以通过指定不同的端口以避免冲突。
  *示例：* ``ray start --head --port=6380``

- ``--temp-dir``
  自定义 Ray 的临时文件目录。Ray 默认会将大量的运行日志和临时文件写在 ``/tmp/ray`` 目录下。长期运行的重负载任务如果塞满宿主机 ``/tmp`` 目录会导致机器无法工作。建议将其指向空间更大的数据盘。
  *示例：* ``ray start --head --temp-dir=/data/ray_tmp``

- ``--dashboard-host`` 与 ``--dashboard-port``
  Ray 提供了功能强大的可视化监控大盘。默认情况下，Dashboard 只绑定 ``127.0.0.1``，此时外部浏览器无法访问。如果需要通过局域网直接查看 UI，可以将 host 设为全局并自定义网页端口。
  *示例：* ``ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265``

- ``--num-cpus`` 与 ``--num-gpus``
  显式声明当前节点提供给 Ray 集群的 CPU 核心数和加速卡数。通常 Ray 能够自动检测到机器的总资源，但当您只想划分机器的“一半”资源给 Ray 调度时，可以直接强行指定。
