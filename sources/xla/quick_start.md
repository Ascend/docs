# xla-npu

在单卡昇腾 NPU 上从源码构建 [XLA-NPU](https://gitcode.com/cann/xla-npu)，并跑通官方 examples 里的基础算子测试（ops，CPU 与 NPU 结果对比），完成一次端到端验证：源码编译 → PJRT 插件加载 → JAX 后端发现 → NPU 上的算子执行与数值比对。

> XLA-NPU 是华为面向昇腾 NPU 的 XLA 后端实现：通过接入 OpenXLA/XLA、对接 JAX 框架，提供推理场景的图编译加速。本文档覆盖其 `examples/ops` 算子测试样例（Add / Tanh / Multiply / Dot / Divide 等），MMOE / Qwen2 模型推理样例见上游 [examples/README.md](https://gitcode.com/cann/xla-npu/blob/master/examples/README.md)。

## 前置条件

### 硬件

Atlas 900 A2 单卡（Ascend NPU），并按需完成物理机或容器内的设备挂载。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- Python 3.11 环境
- 可用的 CANN ≥ 8.5.0（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 基本编译工具：`git`、`wget`、`curl`、`tar`、`xz`、`g++`、`cmake`（`build.sh` 内部会自行下载 OpenJDK 21 与 Bazel 7.7.0）

### 本文档示例使用的版本

**配套机器**：

- **机器类型**：Atlas 900 A2 单卡（Ascend 910B）
- **操作系统**：Ubuntu 22.04

**配套镜像**：

swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-910b-ubuntu22.04-py3.11

**软件版本**：

| 组件 | 版本 |
| --- | --- |
| Python | 3.11 |
| CANN | 9.0.0 |
| JAX / jaxlib | 0.8.1（`build.sh` 自动安装） |

### 检查前置是否满足

检查 Python 版本：

```shell #test id="check-py"
python --version
```

输出结果如下：

```shell #test-result id="check-py" fuzzy='xxx'
Python 3.11.xxx
```

确认能看到 NPU 设备：

```shell
npu-smi info
```

输出如下类似结果（型号与数量因机器而异）：

```
+------------------------------------------------------------------------------------------------+
| npu-smi 25.5.2                   Version: 25.5.2                                               |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B4               | OK            | 89.9        39                0    / 0             |
| 0                         | 0000:41:00.0  | 0           0    / 0          2922 / 32768         |
+===========================+===============+====================================================+
| No running processes found in NPU 0                                                            |
+===========================+===============+====================================================+
```

> 如果 `npu-smi` 不存在，请回到 [Ascend 官方快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 补装驱动；CANN 版本不满足时升级到 8.5.0 及以上。

## 获取代码

<!-- 工作流注入的 UPSTREAM_REF（最新 commit 的 sha）通过这个隐藏的 #test-setup 捕获并注入到下方 clone 命令中 -->

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

克隆 xla-npu 仓库并 checkout 到工作流注入的最新 commit：

```shell #test id="clone-repo" load="upstream_ref>>ref"
git clone https://gitcode.com/cann/xla-npu.git
cd xla-npu
git checkout <ref>
echo "HEAD $(git log -1 --format=%h)"
```

输出结果如下：

```shell #test-result id="clone-repo" fuzzy='xxx' fuzzy='...'
...HEAD xxx
```

\<ref> 为流水线注入的上游最新 commit（xla-npu 无 release / tag，看护锚定默认分支 HEAD）。

## 构建安装

按上游 [INSTALL_GUIDE.md](https://gitcode.com/cann/xla-npu/blob/master/INSTALL_GUIDE.md) 的方式，用 `build.sh` 完成一键安装：下载 JDK / Bazel、克隆并冻结 XLA 依赖、bazel 编译 PJRT 插件、安装 JAX，最后生成 `build/xla_npu_env` 环境变量文件。

```shell #test-setup
cd xla-npu
chmod +x build/build.sh
# 钳制编译并行度：CI runner 核数高，bazel --jobs=$(nproc) 会触发内存 OOM。
# 本机内存紧张可进一步降到 4。
export PARALLEL_JOBS=16
./build/build.sh install --npu-auto-fuse-backend=1
```

> - `build.sh` 内部会从华为镜像源下载 OpenJDK 21 与 Bazel 7.7.0，接着 `git clone` 并 checkout 到固定 commit 的 OpenXLA/XLA，再 `bazel build //xla/xla_npu/pjrt/c:pjrt_c_api_npu_plugin.so`。首次构建需要下载依赖，耗时较长，请耐心等待；网络抖动导致的下载失败可重复执行（脚本带断点续传，已完成的步骤会跳过）。
> - `--npu-auto-fuse-backend=1` 使用 afir 融合后端（默认值，开源样例 `examples/ops` 在此后端下可跑通）；`examples/qwen` 的 Qwen2 样例在 afir 后端下不支持，跑模型样例请改用 `--npu-auto-fuse-backend=3`（ge 后端，需额外 `pip install decorator==5.2.1 sympy==1.14.0 attrs==25.4.0`）。

验证编译产物与 JAX 安装：

```shell #test id="verify-install"
cd xla-npu
test -f build/code/xla/bazel-bin/xla/xla_npu/pjrt/c/pjrt_c_api_npu_plugin.so && echo "plugin ok"
source build/xla_npu_env
python -c "import jax; print('jax', jax.__version__)"
```

输出结果如下：

```shell #test-result id="verify-install" fuzzy='xxx'
plugin ok
jax xxx
```

## 在 JAX 中使用 NPU 后端

`source build/xla_npu_env` 后，设置 PJRT 插件路径与 `JAX_PLATFORMS=npu`，即可让 JAX 在 NPU 上执行图编译。下面用一个最小脚本验证：JAX 默认后端是 `npu` 且能在设备上 `jit` 跑一次 gelu：

```shell #test id="jax-npu-smoke"
source xla-npu/build/xla_npu_env
python - <<'PY'
import os
import jax
import jax.numpy as jnp

root = os.environ['XLA_NPU_ROOT_DIR']
lib = os.path.join(root, 'build/code/xla/bazel-bin/xla/xla_npu/pjrt/c/pjrt_c_api_npu_plugin.so')
os.environ['PJRT_NAMES_AND_LIBRARY_PATHS'] = f'npu:{lib}'
os.environ['JAX_PLATFORMS'] = 'npu'
import jax.lib.xla_client  # 触发后端发现，无需调用

assert jax.default_backend() == 'npu', f'backend={jax.default_backend()}'
x = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
y = jax.jit(jax.nn.gelu)(x)
result = y.block_until_ready()
print(f'backend=npu devices={len(jax.devices())} shape={list(result.shape)}')
PY
```

输出结果如下：

```shell #test-result id="jax-npu-smoke" fuzzy='xxx' fuzzy='...'
...backend=npu devices=xxx shape=[2, 4]...
```

其中 `backend=npu` 是 JAX 探测到 xla-npu PJRT 插件后给出的标识；`devices` 数量等于机器可用 NPU 卡数。如果这一步 `default_backend()` 不是 `npu`，多半是 `PJRT_NAMES_AND_LIBRARY_PATHS` 没指对插件路径——回到「验证编译产物」一节检查 `.so` 是否存在。

## 运行算子测试样例

运行官方 `examples/ops` 样例：JAX 先在同一份输入上分别跑 CPU 与 NPU，再对比两者结果（最大绝对误差 / 最大相对误差 / 逐算子通过状态），全部算子一致时打印 `TEST PASSED`：

```shell #test id="run-ops"
source xla-npu/build/xla_npu_env
cd xla-npu/examples/ops
python main.py --num-runs 1 > /tmp/ops.log 2>&1
grep -q "TEST PASSED" /tmp/ops.log && echo "ops ok" || { tail -n 60 /tmp/ops.log; exit 1; }
```

输出结果如下：

```shell #test-result id="run-ops"
ops ok
```

> - 输出重定向到 `/tmp/ops.log` 以便断言成功标志 `TEST PASSED`；失败时自动打印日志尾部便于定位。
> - `--num-runs 1` 表示每个算子各跑 1 次；`--skip-cpu` / `--skip-npu` 可只跑单边，`--cleanup` 会清理临时输入输出文件。
> - ops 样例会跳过 NPU 后端尚未实现对应 opcode 的算子（Dot / Gather / Compare EQ / Select / And，见 `examples/ops/config.py` 的 `_UNSUPPORTED_OPS`），其余算子（Add / Tanh / Multiply / Divide / Subtract / Negate / Exp / Maximum / Concatenate / ReduceMax / ReduceSum / Rsqrt / Sin / Cos / Transpose / Slice）逐项完成 CPU↔NPU 比对。
> - MMOE / Qwen2 模型推理样例（`examples/mmoe` / `examples/qwen`）需要 `pip install flax==0.12.4` 并按各自 README 初始化参数，本 Quick Start 只跑最轻量的 ops 样例。
