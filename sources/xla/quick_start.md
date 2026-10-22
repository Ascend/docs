# xla-npu

在单卡昇腾 NPU 上从源码构建 [XLA-NPU](https://gitcode.com/cann/xla-npu)，并跑通官方 `examples/ops` 算子测试（CPU 与 NPU 结果对比）。

## 前置条件

- Python 3.11
- CANN ≥ 8.5.0（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 基本工具：`git`、`wget`、`curl`、`tar`、`xz`、`cmake`、`g++`

完整要求见官方安装指南 [INSTALL_GUIDE.md](https://gitcode.com/cann/xla-npu/blob/master/INSTALL_GUIDE.md)。

## 安装

按官方 [INSTALL_GUIDE.md](https://gitcode.com/cann/xla-npu/blob/master/INSTALL_GUIDE.md) 的步骤，clone 仓库后用 `build.sh` 一键安装：

<!-- 工作流注入上游最新 commit 的 sha（xla-npu 无 release / tag，看护锚定默认分支 HEAD） -->

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

```shell #test id="clone-repo" load="upstream_ref>>ref"
git clone https://gitcode.com/cann/xla-npu.git
cd xla-npu
git checkout <ref>
echo "HEAD $(git log -1 --format=%h)"
```

```shell #test-result id="clone-repo" fuzzy='xxx' fuzzy='...'
...HEAD xxx
```

```shell #test-setup
cd xla-npu
chmod +x build/build.sh
# 钳制编译并行度：CI runner 核数高，bazel --jobs=$(nproc) 会触发内存 OOM。
export PARALLEL_JOBS=16
./build/build.sh install --npu-auto-fuse-backend=1
```

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

## 运行样例

按官方 [examples/README.md](https://gitcode.com/cann/xla-npu/blob/master/examples/README.md)，`source build/xla_npu_env` 后运行 ops 算子测试：JAX 先在同一份输入上分别跑 CPU 与 NPU，再对比两者结果，全部算子一致时打印 `TEST PASSED`：

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

> MMOE / Qwen2 模型推理样例（`examples/mmoe` / `examples/qwen`）需要 `pip install flax==0.12.4` 并按各自 README 初始化参数，本 Quick Start 只跑最轻量的 ops 样例，详见 [examples/README.md](https://gitcode.com/cann/xla-npu/blob/master/examples/README.md)。
