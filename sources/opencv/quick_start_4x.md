# OpenCV 4.x 零补丁构建

[OpenCV](https://github.com/opencv/opencv) 是开源计算机视觉库。本文走的是**零补丁**那条路线：在单卡昇腾 NPU 上把 **OpenCV 4.14.0**（4.x 的最后一版）源码编译成带 CANN 后端的 `cv2`（`WITH_CANN=ON`），**一行源码不打补丁**，然后跑通三件事：

- **CPU 核心 API**：读图 / 写图 / 颜色空间 / 缩放 / 绘制 / 视频读写；
- **NPU DNN 推理**：`cv2.dnn` + `DNN_BACKEND_CANN` + `DNN_TARGET_NPU`，MobileNetV2 在 910B 上跑出与 CPU 参考一致的 top-1；
- **零补丁证据**：构建全程 `git status --porcelain` 为空，且推理时能抓到三行 CANN 后端日志，证明这次真的上了 NPU。

**什么时候用本文**：你要的是「OpenCV 在昇腾上能跑」，不想维护任何本地 patch，也能接受 4.x（4.14.0 是 4.x 的末代，5.0 之后不再有 4.x 更新）。要用 OpenCV 5.x 的新 DNN 图引擎（`onnx_importer2`），或要用 `cv2.cann` 那套 contrib 经典算子 API，本文都不覆盖。

## 这条路线的 NPU 覆盖到哪（重要边界）

**零补丁 ⇒ NPU 能力只有 DNN 推理这一条。** `imgproc` / `imgcodecs` / `videoio` 在本文这条路线下**全部走 CPU**。

OpenCV 4.x 主仓里的 NPU 加速入口只有一处：

| 入口 | 在哪 | 本文的零补丁路线覆盖到吗 |
| --- | --- | --- |
| DNN CANN 后端 | 主仓 `modules/dnn`（`WITH_CANN=ON`） | ✅ **覆盖** —— 本文验证的那条（跑没跑在 NPU 上由 CANN 后端日志坐实，见「怎么证明这次真的跑在 NPU 上」） |

所以本文那几件 CPU 事情，NPU 归属是这样：

| 本文验证的 CPU 场景 | 能上 NPU 吗 |
| --- | --- |
| 读图 / 写图（`imread` / `imwrite`） | ❌ `imgcodecs` 没有 NPU 解码路径 |
| 颜色空间（`cvtColor`） | ❌ `imgproc` 在这条路线下没有 NPU 路径 |
| 缩放（`resize`） | ❌ 同上 |
| 绘制（`rectangle` / `putText`） | ❌ 同上 |
| 视频读写（`VideoWriter` / `VideoCapture`） | ❌ 视频编解码仍是 CPU |

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，并按需完成物理机或容器内的设备挂载（`/dev/davinci*` 等）。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境（3.12）
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)），`npu-smi info` 能列出设备
- 可访问 GitHub（clone 源码 + 下载模型）
- 编译依赖：`git` / `build-essential` / `cmake` / `pkg-config` / `zlib1g-dev` / `libjpeg-dev` / `libpng-dev` / `libtiff-dev`（见「前置安装」里的 apt 那一步）

```{admonition} Note
:class: note
`opencv-python`（pip wheel）**不带** CANN 后端，必须从源码构建，`cv2.dnn.DNN_BACKEND_CANN` 才会链到真后端。
```

### 本文档示例使用的版本

**配套机器**：

- **机器类型**：Atlas 900 A2 PODc（Ascend 910B4）
- **操作系统**：Ubuntu 22.04.5 arm64

**配套镜像**：

swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.2.0-910b-ubuntu22.04-py3.12

**软件版本**：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.2.0（`/usr/local/Ascend/cann-9.2.0`） |
| CMake / GCC | 3.22.1 / 11.4.0 |
| OpenCV | **4.14.0**（源码，`WITH_CANN=ON`，`CMAKE_BUILD_TYPE=Release`）—— 4.x 的末代 |

CANN 9.1.0 的库目录布局与 9.2.0 略有差异（见「桥接 CANN 的库目录」），本文的链接体积实测在 9.1.0 与 9.2.0 上各做了一次 —— aarch64 的 ±128 MiB 分支窗口是架构约束，与 CANN 版本无关。

```{admonition} Note
:class: note
**本文的验证状态（2026-11-04）**：配套的看护脚本（`tests/opencv/test_quick_start_ascend.py`）在一台 910B4 机器上把本文**从头到尾跑了一遍**（22 个步骤：apt-get 依赖 → clone 4.14.0 → 确认零补丁 → CANN 库桥接 → cmake configure → Release 编译安装 → 校验产物与 `.text` → 6 个 CPU quickstart → CANN/NPU 推理），**全部通过**，整轮 1522.6 s。零补丁是真的（`git status --porcelain` 输出 0 行）、推理是真的上卡（三行 CANN 后端标记齐全 + NPU 与 CPU top-1 / 分数一致）。
```

## 前置安装

确认能看到 NPU 设备：

```shell
npu-smi info
```

```{admonition} Note
:class: note
如果 `npu-smi` 不存在，请回到 [Ascend 官方快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 补装驱动。
```

检查 Python / CMake：

```shell #test id="check-toolchain"
python --version
cmake --version | head -n 1
```

输出结果如下：

```shell #test-result id="check-toolchain" fuzzy='xxx'
Python 3.12.xxx
cmake version xxx
```

加载 CANN 环境变量：

```shell #test-setup
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
```

确认 toolkit 根目录可访问：

```shell #test id="check-cann-path"
ls -d /usr/local/Ascend/ascend-toolkit/latest
```

输出结果如下：

```shell #test-result id="check-cann-path"
/usr/local/Ascend/ascend-toolkit/latest
```

装编译依赖（国内网络把 apt 源换成阿里云镜像）：

```shell #test-setup
sed -i 's|http://ports.ubuntu.com/ubuntu-ports/|https://mirrors.aliyun.com/ubuntu-ports/|g' /etc/apt/sources.list
apt-get update -qq && apt-get install -y -qq git build-essential cmake pkg-config zlib1g-dev libjpeg-dev libpng-dev libtiff-dev
```

## 为什么这条路能零补丁

OpenCV 5.x 上跑通 CANN 后端需要改源码（`cv::MatShape` 适配、窄 op 头等），本文不采用那套方案，改用「**钉 4.14.0 + Release 构建 + CANN 库指向 `<CANN>/lib64`**」把补丁数降到 0：

| OpenCV 5.x 上要打的补丁 | 根因 | 本文怎么绕开 |
| --- | --- | --- |
| `cv::MatShape` 适配 | OpenCV **5.0.0** 把 dnn 层的 shape 换成 `cv::MatShape`，而上游 CANN 后端 `modules/dnn/src/op_cann.hpp` 从 4.11 到 5.0.0 **一个字节都没改**（4.11.0 / 4.12.0 / 4.13.0 / 4.14.0 / 5.0.0 全是 4446 字节，构造函数仍是 `const std::vector<int>&`）→ 5.0 编译不过 | 钉 **4.14.0**：`gemm_layer.cpp` / `matmul_layer.cpp` 在 4.11→4.14 与 CANN 后端同源（仍是 `std::vector<int>`），原样编译通过。**实测**：`op_cann.hpp` 在 4.11.0 与 4.14.0 之间 **md5 逐字节相同**，`op_cann.cpp` 只差 3 处（全是错误文案与注释，零逻辑变化） |
| 窄 op 头 | `all_ops.h` 一次拉进 ~1500 个 op，每个引用 `op_cann.hpp` 的 TU 都展开一遍；Debug(-O0) 下 dnn `.text` 冲到 516MB，超 aarch64 `R_AARCH64_CALL26` 的 ±128MB 分支范围 → 链接必炸 | 改 **Release(-O3)**：实测**不打**窄 op 头的 4.14.0 Release 构建，dnn `.text` 只有 **53.3 MiB**（55,920,432 B），`libopencv_dnn.so` 一次链接通过（实测环境：Atlas 900 A2 / 910B4、Ubuntu 22.04 arm64、gcc 11.4、CANN 9.1.0、OpenCV 4.14.0）。⚠️ 这条**只在 4.x 成立**：5.0.0 的引用 TU 从 28 个涨到 39 个、dnn 源码量 +50.6%，同样 Release + `all_ops.h` 实测仍然链接失败（`relocation truncated to fit: R_AARCH64_JUMP26`，受害者是 protobuf 的 `AddDescriptorsRunner`）—— Release 不是通用解法，见「已知边界」 |

为什么 4.x 天然比 5.x 省补丁：`all_ops.h` 是 **CANN 的头**，与 OpenCV 版本无关 ⇒ 每个引用 TU 展开的体积恒定，`.text` 随 TU 数量线性增长。实测引用 `op_cann.hpp` 的 TU 数：4.11.0 = 27，**4.14.0 = 28**（多的那个是 `layers/reduce_layer.cpp` —— 4.14 给 reduce 层新增了 CANN 后端支持），5.0.0 = 39。dnn 源码量则是 4.11.0 = 2,999,863 B → **4.14.0 = 3,068,750 B（+2.3%）** → 5.0.0 = 4,518,814 B（+50.6%）。4.14 站在「TU 只多 1 个、自身代码只多 2.3%」的位置，离 128 MiB 窗口还有 ~74 MiB 余量；5.0 站在「TU 多 39%、自身代码多 50%」的位置，就撞墙了。

### 另一个坑：CANN 9.2 的裁剪版 `libgraph.so`

这一条跟 OpenCV 版本无关，**上游所有版本（4.11.0 / 4.14.0 / 5.0.0）都一样**（`net_cann.cpp` 在这三个 tag 里那几行一字不差），但它会以「补丁还不够用」的样子出现，所以单开一节。

`modules/dnn/src/net_cann.cpp` 里有一句：

```cpp
std::map<ge::AscendString, ge::AscendString> options = { ... };
```

`std::map` 需要 `ge::AscendString` 的 `operator<`。CANN 9.2.0 的头文件 `graph/ascend_string.h` 声明了它，但**能不能链上，取决于你链的是哪份 `libgraph.so`**：

| CANN 9.2.0 里的 `libgraph.so` | 大小 | `ge::AscendString` 的 4 个**序关系**运算符（`<` `>` `<=` `>=`） |
| --- | --- | --- |
| `<CANN>/lib64/libgraph.so`（= `aarch64-linux/lib64`） | 2.3 MB | **有**（`_ZNK2ge12AscendStringltERKS0_` 等 4 个） |
| `<CANN>/aarch64-linux/devlib/linux/aarch64/libgraph.so` | 198 KB | **没有**（裁剪版只留 `Find` / `Hash` / `GetLength` / `GetString` / `==` / `!=`） |

指错目录的典型症状（**链接期**报错，不是编译期）：

```
undefined reference to `ge::AscendString::operator<(ge::AscendString const&) const'
```

判定方法（一眼看出是完整版还是裁剪版）：

```shell
nm -D --defined-only <CANN>/lib64/libgraph.so | grep -c '_ZNK2ge12AscendStringlt'
```

**1** = 完整版（有 `operator<`）；**0** = 裁剪版。

（顺带一个版本无关的数：完整版 `libgraph.so` 里 `ge::AscendString` 的 const 成员共 12 个，裁剪版 8 个 —— 差的正好是 4 个序关系运算符。）

**怎么绕**：把给 OpenCV 的 CANN 软链一律指向 `<CANN>/lib64`（运行时的 `LD_LIBRARY_PATH` 用的也是它，`set_env.sh` 里写死 `${version_dirpath}/lib64`），**不要**指向 `devlib/...`。见下一节。

## 构建 OpenCV（4.14.0，零补丁）

### clone OpenCV（版本钉死 4.14.0）

本文没有 `<ref>` 占位符 —— 版本是钉死的，钉 4.14.0 是「零补丁」的前提（原因见上面的补丁表）。

```shell #test-setup
git clone --depth 1 --branch 4.14.0 https://github.com/opencv/opencv.git
```

### 确认源码是干净的（零补丁的证据）

打完补丁的树一定会有 diff；没打补丁的树 `git status` 是空的。同时确认 CANN 后端的构造函数签名仍是 4.x 原样（`std::vector<int>`，不是 5.0 的 `cv::MatShape`）。

```shell #test id="check-source-pristine"
cd opencv
git status --porcelain | wc -l
grep -n "const std::vector<int>& shape, const std::string& name" modules/dnn/src/op_cann.hpp
```

输出结果如下：

```shell #test-result id="check-source-pristine" fuzzy='xxx'
0
xxx:        CannConstOp(const uint8_t* data, const int dtype, const std::vector<int>& shape, const std::string& name);
```

```{admonition} Note
:class: note
第一行 `0` 是 `git status --porcelain` 的行数 —— 0 行 = 源码树与上游 4.14.0 逐字节相同（零补丁）。第二行的 `xxx` 是 `grep` 命中的行号。
```

### 桥接 CANN 的库目录

OpenCV 的 `OpenCVFindCANN.cmake` 在 `${CANN_INSTALL_DIR}/{acllib,lib64,compiler/lib64}/` 下找 ACL 库：

| 找什么 | 期望位置 |
| --- | --- |
| `libascendcl.so` | `acllib/lib64/` |
| `libacl_op_compiler.so`、`libacl_dvpp_mpi.so` | `lib64/` |
| `libgraph.so`、`libge_compiler.so`、`libgraph_base.so` | `compiler/lib64/` |

这三个位置在 CANN 里都不叫这三个名字，所以要补软链。**统一指向「完整版 CANN 库目录」**：

- CANN 9.1.x：`<CANN>/lib64` 不存在，完整库目录是 `<CANN>/aarch64-linux/lib64`；
- CANN 9.2.x：`<CANN>/lib64` 已经是指向 `aarch64-linux/lib64` 的软链（同一批库）。

注意 **不要**指向 `<CANN>/aarch64-linux/devlib/linux/aarch64/` —— 那是裁剪过的设备侧运行时（`libgraph.so` 只有 198 KB，少了 `ge::AscendString` 的比较运算符，见上一节）。

补 symlink（`acllib` / `compiler` 要先是目录，所以先 `mkdir`）：

```shell #test-setup
CANN_DIR=/usr/local/Ascend/ascend-toolkit/latest
# CANN 9.1 没有 <CANN>/lib64，补一条指向 aarch64-linux/lib64 的软链；
# CANN 9.2 已经有（本身就是 aarch64-linux/lib64 的软链），不动它。
[ -e "$CANN_DIR/lib64" ] || ln -sfn "$CANN_DIR/aarch64-linux/lib64" "$CANN_DIR/lib64"
mkdir -p "$CANN_DIR/acllib" "$CANN_DIR/compiler"
ln -sfn "$CANN_DIR/lib64" "$CANN_DIR/acllib/lib64"
ln -sfn "$CANN_DIR/lib64" "$CANN_DIR/compiler/lib64"
```

确认三个软链都指向**完整版** lib64（最后那个 `nm` 计数是 1 = 完整版，0 = 裁剪版）：

```shell #test id="check-cann-lib-layout"
CANN_DIR=/usr/local/Ascend/ascend-toolkit/latest
ls -l "$CANN_DIR/acllib/lib64/libascendcl.so" "$CANN_DIR/lib64/libacl_op_compiler.so" "$CANN_DIR/compiler/lib64/libgraph.so"
nm -D --defined-only "$CANN_DIR/compiler/lib64/libgraph.so" | grep -c '_ZNK2ge12AscendStringlt'
```

输出结果如下：

```shell #test-result id="check-cann-lib-layout" fuzzy='xxx'
xxx
xxx
xxx
1
```

```{admonition} Note
:class: note
前三行的 `xxx` 是三个软链的 `ls -l` 输出（目标路径随 CANN 版本不同：9.1 指 `aarch64-linux/lib64`，9.2 指 `lib64`）。最后一行 `1` 是 `ge::AscendString::operator<` 的符号数 —— **0 就说明链到了裁剪版**。
```

```{admonition} Note
:class: note
`ln -sfn` 到已存在的软链是「改指向」而不是「往目录里塞链接」，所以重复执行是安全的（`lib64` 那条还额外用 `[ -e ... ]` 挡掉了「指向自己」的 `are the same file` 报错）。

`libgraph_base.so` 是 CANN ≥ 6.3.RC1 之后 `OpenCVFindCANN.cmake` 才会去找的库（找不到只 WARN 不报错），但它是 `ge::AscendString` 那组符号的另一个来源，务必确认它在 `compiler/lib64/` 里。

**不想往 CANN 安装目录里写东西**（只读挂载 / 无 root）时，等价做法是直接把路径喂给 cmake：`-Dfound_lib_ascendcl=... -Dfound_lib_acl_op_compiler=... -Dfound_libacldvppmpi=... -Dfound_lib_graph=... -Dfound_lib_ge_compiler=... -Dfound_libgraph_base=...`（`find_library` 命中已存在的 cache 变量就不再搜目录）。本文的构建走的是软链那条路线。
```

### 配置并编译

CANN 后端开关 `WITH_CANN=ON`（不是 `BUILD_CANN`）；`-DOPENCV_DOWNLOAD_MIRROR_ID=gitcode` 让 cmake configure 阶段走 gitcode 镜像拉 ADE / IPPICV / TBB。

**两个取舍点**：

- `CMAKE_BUILD_TYPE=Release` 而不是 `Debug` —— 这是绕开窄 op 头补丁的关键（Debug 的 `-O0` 会把 `all_ops.h` 里每个 op 的内联函数都落盘，`.text` 撑爆 aarch64 的分支范围）。**注意这条只对 4.x 成立**：5.0.0 实测 Release + `all_ops.h` 一样链接失败（`R_AARCH64_JUMP26` 越界）；
- `-DBUILD_opencv_apps=OFF` —— OpenCV 自带的 `apps/model-diagnostics` 是个**可执行文件**，它要直接链 CANN 的 `libge_compiler.so` / `libgraph.so` / `libgraph_base.so`，而这三个库自身还有一堆没被链进来的依赖（`ge::*`、`ascend_private::protobuf::*`、`absl::*`…）。共享库链接允许留未定义符号（所以 `libopencv_dnn.so` 没事），可执行文件链接不允许 → **链接期报几百条 `undefined reference` 并让整个 `all` 目标失败**（CANN 9.2.0 实测）。关掉 apps 就绕开了。

安装前缀用 `/usr/local/opencv-cann-414`：装到 `/usr/local/opencv-cann` 也行，只要把下面两处（前缀 / `PYTHONPATH`）一起换掉即可。

```shell #test id="cmake-configure"
cd opencv
mkdir -p build && cd build
set -o pipefail
PY_INC=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
PY_LIB=$(python3 -c 'import sysconfig,os,sys; print(os.path.join(sysconfig.get_config_var("LIBDIR"), "libpython%d.%d.so" % sys.version_info[:2]))')
test -f "$PY_INC/Python.h" || { echo "Python.h not found under $PY_INC"; exit 1; }
test -f "$PY_LIB" || { echo "libpython not found at $PY_LIB"; exit 1; }
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local/opencv-cann-414 \
      -DWITH_CANN=ON \
      -DCANN_INSTALL_DIR=/usr/local/Ascend/ascend-toolkit/latest \
      -DBUILD_opencv_world=OFF \
      -DBUILD_opencv_apps=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_TESTS=OFF \
      -DBUILD_PERF_TESTS=OFF \
      -DBUILD_LIST=core,imgproc,imgcodecs,videoio,dnn,python3 \
      -DSOC_VERSION=ascend910b1 \
      -DBUILD_opencv_python3=ON \
      -DBUILD_opencv_python_bindings_generator=ON \
      -DPYTHON_INCLUDE_DIR="$PY_INC" \
      -DPYTHON_LIBRARY="$PY_LIB" \
      -DOPENCV_ENABLE_NONFREE=OFF \
      -DOPENCV_DOWNLOAD_MIRROR_ID=gitcode \
      .. | tee /tmp/opencv-configure.log
```

```{admonition} Note
:class: note
Python 路径用 `sysconfig` 现场取，不写死版本号目录：CI 镜像是 `/usr/local/python3.12.13`，验证机是 `/usr/local/python3.12.15`，两条路径都由 `sysconfig` 给出。

写死的后果不是 configure 期报错，而是**编译期**才炸：`PYTHON_INCLUDE_DIR` 指向不存在的目录时 cmake 不会校验，`make` 编 `modules/python/src2/cv2.cpp` 时才报 `fatal error: Python.h: No such file or directory`（npu-4 实测）。前面两条 `test -f` 把这个失败提前到 configure 之前并打印出实际路径。
```

输出结果如下：

```shell #test-result id="cmake-configure" fuzzy='xxx' fuzzy='...'
-- The CXX compiler identification is GNU xxx
...
--   CANN:xxx YES
...
-- Configuring done
-- Generating done
-- Build files have been written to: xxx/build
```

确认 cmake 找到了 CANN 库（configure 输出里的 `CANN: lib... is found at ...`）：

```shell #test id="check-cmake-libs"
grep "CANN: libascendcl.so is found" /tmp/opencv-configure.log
grep "CANN: libgraph.so is found" /tmp/opencv-configure.log
```

输出结果如下：

```shell #test-result id="check-cmake-libs" fuzzy='xxx'
-- CANN: libascendcl.so is found at xxx
-- CANN: libgraph.so is found at xxx
```

```{admonition} Note
:class: note
这段必须排在 configure 块**之后**：doc-test 的执行顺序 = 文档顺序，而 `/tmp/opencv-configure.log` 是上面那个 configure 块用 `tee` 写出来的。放在前面会得到 `grep: /tmp/opencv-configure.log: No such file or directory`。
```

编译 + 安装一体（`--target install`）。**并行度别开高**：`all_ops.h` 那几个 TU 每个要吃掉 3GB 上下内存，容器内存配额（cgroup limit）只有 32GB 时，`-j` 超过 6 会被 OOM killer 直接干掉 `cc1plus`（表现为编译日志里出现 `c++: fatal error: Killed signal terminated program cc1plus`）。所以并行度做成可覆盖的环境变量：默认 `--parallel 2`（保守），CI 机器内存更富裕时可以调高。

```shell #test-setup
cd opencv/build
cmake --build . --target install --parallel "${OPENCV_BUILD_JOBS:-2}"
```

```{admonition} Note
:class: note
只想「一次跑完」的话也可以把 `OPENCV_BUILD_JOBS` 设成 `$(nproc)`；但先看一眼容器的内存配额（`cat /sys/fs/cgroup/memory/memory.limit_in_bytes`，v2 是 `/sys/fs/cgroup/memory.max`），`nproc × 3GB` 超过配额就必然 OOM。
```

### 校验 build 产物

```{admonition} Note
:class: note
`opencv_version` 那个小工具属于 `apps/`（上面 `-DBUILD_opencv_apps=OFF` 关掉了），所以这里直接看装出来的库。
```

```shell #test id="verify-build"
ls -l /usr/local/opencv-cann-414/lib/libopencv_core.so.4.14.0 /usr/local/opencv-cann-414/lib/libopencv_dnn.so.4.14.0
```

输出结果如下：

```shell #test-result id="verify-build" fuzzy='xxx'
xxx
xxx
```

顺便看一眼 dnn 库的实际体积 —— 这是「Release 不用窄 op 头补丁」那一条的证据（Debug -O0 下同样的库 `.text` 是 516MB）：

> 实测（4.14.0 + CANN 9.1.0 + gcc 11.4 + Release -O3）：`libopencv_dnn.so.4.14.0` 的 `.text` = **55,920,432 B（≈53.3 MiB）**，`Total` = 77,111,426 B，库文件 98,067,792 B。这个数比 aarch64 的 ±128 MiB 分支窗口小一个量级，所以**不需要**窄 op 头补丁。同一份源码在 Debug(-O0) 下 `.text` 会涨到 516MB，直接撑爆窗口。
>
> 对照：4.11.0 同参数是 `.text` = 54,529,328 B（52.0 MiB）—— 4.14.0 多出的 ~1.3 MiB 就是第 28 个 CANN TU（`reduce_layer.cpp`）展开 `all_ops.h` 的代价。
>
> 换 CANN 9.2.0 复测（npu-4 / 910B4 / gcc 11.4 / Release -O3，同一份 4.14.0 源码、零补丁）：`.text` = **56,123,568 B（≈53.5 MiB）**、`Total` = 77,331,307 B、库文件 98,304,552 B —— 与 9.1.0 相差 ~0.2 MiB（CANN 头文件版本差异），离 128 MiB 窗口的余量不变。
>
> 2026-11-04 在 npu-4 上跑完整份文档的 doc-test（22 个步骤全绿，含零补丁构建 + NPU 推理）再测一遍：`.text` = **56,123,568 B —— 与上一次逐字节相同**，`Total` = 77,336,380 B、库文件 98,308,672 B（后两个比上一轮大 ~4~5 KB，是链接期符号表 / 节名的抖动，不影响 `.text`）。也就是说「4.14.0 + Release 不用窄 op 头」这条结论在**两次独立构建**上都成立。

```shell #test id="dnn-text-size"
size -A /usr/local/opencv-cann-414/lib/libopencv_dnn.so.4.14.0 | awk '$1==".text" || $1=="Total"'
```

输出结果如下：

```shell #test-result id="dnn-text-size" fuzzy='xxx'
.text xxx
Total xxx
```

```{admonition} Note
:class: note
`xxx` 是字节数。这条断言**必须只输出两行**：`awk` 把 `.text` 与 `Total` 钉死，中间不留任何别的段。

原因是 doc-test 契约里 placeholder 是**通配符**而不是行省略 —— expected 里如果写一行独占的 `...`，正则会强要「一个换行 + 至少一个字符」，而过滤后的输出里 `.text` 与 `Total` 是**紧挨着**的 ⇒ 永远 MISMATCH。（`grep -E '^\.text|^Total'` 还有一个副作用：它会连带捞进 `.text.hot` / `.text._Z...` 之类的段，行数随构建配置浮动。）
```

再加一条上界，把「Release 不是 Debug」这件事也钉住：

```shell #test id="dnn-text-size-bound"
TEXT=$(size -A /usr/local/opencv-cann-414/lib/libopencv_dnn.so.4.14.0 | awk '$1==".text"{print $2}')
echo "text bytes: $TEXT"
test "$TEXT" -gt 0 && test "$TEXT" -lt 67108864
```

输出结果如下：

```shell #test-result id="dnn-text-size-bound" fuzzy='xxx'
text bytes: xxx
```

```{admonition} Note
:class: note
上界取 64 MiB：Release 实测 56,123,568 B（CANN 9.2.0）/ 55,920,432 B（CANN 9.1.0），留 ~8 MiB 余量；而 Debug(-O0) 是 516 MB，一秒就被这条断言拦下。
```

## OpenCV Python quickstart

把 Python 指向源码版 `cv2`（带 CANN 后端）：

```shell #test-setup
export PYTHONPATH=/usr/local/opencv-cann-414/lib/python3.12/site-packages:${PYTHONPATH:-}
```

```{admonition} Note
:class: note
注意是 **prepend**（保留镜像 `set_env.sh` 注入的 CANN `PYTHONPATH` 条目）。GE 图编译依赖其中的 `tbe` Python 模块；整体覆盖 `PYTHONPATH` 会让 DNN CANN 后端在图编译阶段报 `There is no valid so about OpsKernelInfoStore or GraphOptimizer`（ret=0xFFFFFFFF）。
```

下面用 clone 下来的 OpenCV 仓库自带测试图 `opencv/samples/data/baboon.jpg`（512×512 狒狒）作样例输入：

![baboon 输入图](images/baboon-input.jpg)

### 读图 / 写图

```python #test id="imread-imwrite"
import os
import cv2

img = cv2.imread('opencv/samples/data/baboon.jpg')
ok = cv2.imwrite('/tmp/opencv_quickstart.png', img)
print('imwrite ok:', ok)
print('shape:', img.shape, 'dtype:', img.dtype)
print('written:', os.path.exists('/tmp/opencv_quickstart.png'))
```

输出结果如下：

```shell #test-result id="imread-imwrite"
imwrite ok: True
shape: (512, 512, 3) dtype: uint8
written: True
```

### 颜色空间转换

```python #test id="cvtcolor"
import cv2

bgr = cv2.imread('opencv/samples/data/baboon.jpg')
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
print('gray shape:', gray.shape, 'mean:', float(gray.mean()))
cv2.imwrite('/tmp/quickstart-gray.png', gray)
```

输出结果如下：

```shell #test-result id="cvtcolor" fuzzy='xxx'
gray shape: (512, 512) mean: xxx
```

```{admonition} Note
:class: note
`xxx` 是灰度均值（baboon 整体偏中亮，约 129.7/255）。灰度图长这样：
```

![灰度转换结果](images/quickstart-gray.png)

### 缩放

```python #test id="resize"
import cv2

img = cv2.imread('opencv/samples/data/baboon.jpg')
resized = cv2.resize(img, (50, 200), interpolation=cv2.INTER_AREA)
print('resized shape:', resized.shape)
cv2.imwrite('/tmp/quickstart-resize.png', resized)
```

输出结果如下：

```shell #test-result id="resize"
resized shape: (200, 50, 3)
```

OpenCV shape 是 `(rows, cols, channels)`，`resize(img, (W, H), ...)` 得 `(H, W, C)`；缩到 50×200（宽×高）后：

![缩放结果](images/quickstart-resize.png)

### 绘制

```python #test id="draw"
import os
import cv2

img = cv2.imread('opencv/samples/data/baboon.jpg')
cv2.rectangle(img, (10, 10), (500, 500), (0, 255, 0), thickness=2)
cv2.putText(img, 'Hello OpenCV', (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
print('rect+text drawn, image mean:', float(img.mean()))
cv2.imwrite('/tmp/quickstart-draw.png', img)
print('written:', os.path.exists('/tmp/quickstart-draw.png'))
```

输出结果如下：

```shell #test-result id="draw" fuzzy='xxx'
rect+text drawn, image mean: xxx
written: True
```

绿框 + 文字绘制结果：

![绘制结果](images/quickstart-draw.png)

```{admonition} Note
:class: note
视频读写（`cv2.VideoWriter` / `cv2.VideoCapture`）需要 `videoio` 模块 —— 本文这版 `BUILD_LIST` **带了** `videoio`，所以可以直接用，见下一节。
```

### 视频读写（videoio）

`BUILD_LIST` 里有 `videoio` 时，`cv2.VideoWriter` / `cv2.VideoCapture` 可用。下面写 10 帧再读回来：

```python #test id="videoio"
import cv2
import numpy as np

w = cv2.VideoWriter('/tmp/quickstart-video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10.0, (320, 240))
print('writer opened:', w.isOpened())
for i in range(10):
    w.write(np.full((240, 320, 3), i * 20, dtype=np.uint8))
w.release()

cap = cv2.VideoCapture('/tmp/quickstart-video.avi')
n = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    n += 1
cap.release()
print('readback frames:', n)
```

输出结果如下：

```shell #test-result id="videoio"
writer opened: True
readback frames: 10
```

## NPU 上跑一次 DNN 推理

### 后端注册表：CANN/NPU 真的编进来了吗

```python #test id="dnn-backend-targets"
import cv2

print('cv2 version:', cv2.__version__)
# 「枚举存在」不能证明后端编进去了：cv2.dnn.DNN_BACKEND_CANN 是无条件暴露的
# （opencv2/dnn/dnn.hpp 的 enum Backend 里没有 #ifdef 守卫），不带 CANN 的
# 构建照样打印得出来。
# 真检查看后端注册表：modules/dnn/src/registry.cpp 里只有 #ifdef HAVE_CANN
# 才会登记 (DNN_BACKEND_CANN, DNN_TARGET_NPU)，于是
# getAvailableTargets(CANN) 在没编 CANN 时返回空列表。
print('OPENCV backend targets:', cv2.dnn.getAvailableTargets(cv2.dnn.DNN_BACKEND_OPENCV))
cann_targets = cv2.dnn.getAvailableTargets(cv2.dnn.DNN_BACKEND_CANN)
print('CANN backend targets:', cann_targets)
if cv2.dnn.DNN_TARGET_NPU not in cann_targets:
    raise SystemExit('CANN/NPU backend is NOT compiled into this cv2: %r' % (cann_targets,))
print('CANN/NPU backend available: True')
```

输出结果如下：

```shell #test-result id="dnn-backend-targets" fuzzy='xxx'
cv2 version: xxx
OPENCV backend targets: [0]
CANN backend targets: [9]
CANN/NPU backend available: True
```

```{admonition} Note
:class: note
`xxx` 是 `cv2.__version__`（4.14.0）。

`getAvailableTargets` 返回的是**后端注册表里实际登记的 target**：`DNN_BACKEND_OPENCV` → `[0]`（`DNN_TARGET_CPU`）、`DNN_BACKEND_CANN` → `[9]`（`DNN_TARGET_NPU`）。两个数值取自 `opencv2/dnn/dnn.hpp` 的 `enum Target`（`DNN_TARGET_CPU = 0` 起数，`DNN_TARGET_NPU` 是第 10 个）。**空列表 = 这个后端根本没编进来** —— `modules/dnn/src/registry.cpp` 里 `(DNN_BACKEND_CANN, DNN_TARGET_NPU)` 的登记被 `#ifdef HAVE_CANN` 包着，不带 CANN 的构建 `getAvailableTargets(CANN)` 返回 `[]`，脚本随即退出 1。
```

### MobileNetV2 ONNX 推理（NPU vs CPU 对照）

跑 MobileNetV2（onnx/models 的 `mobilenetv2-12`，PyTorch 导出，13,964,571 bytes）：

> **模型来源**：本文档仓库里带了一份同字节副本（`tests/opencv/fixtures/mobilenetv2-12.onnx`，看护直接用它，避免 GitHub raw 在国内网络的 `RemoteDisconnected` 抖动）；自己机器上跟着跑时没有这份副本，自动回落到从 onnx/models 下载（重试 5 次）。

> **为什么不用 `OPENCV_FORCE_DNN_ENGINE=1`**：那是 OpenCV 5.0 的坑 —— 5.0 默认 `ENGINE_AUTO` 会选新图引擎（`onnx_importer2`），新引擎尚未支持非 CPU 后端。**4.x 的默认引擎就是 classic 引擎**（4.14.0 实测：不设任何环境变量、不调 `setPreferableBackend`，直接 `forward()` 出来的 top-1 也是 372 / 0.9148），`setPreferableBackend` / `setPreferableTarget` 直接生效，所以本文不需要这个环境变量。

```shell #test id="dnn-cann-infer"
set -e   # python 段失败必须让整个块失败：块里最后一条命令是 cat，它的退出码会盖掉 python 的
python << 'PY'
import ctypes
import os
import shutil
import sys
import time
import urllib.request
import numpy as np
import cv2

# CANN 后端的关键进度日志全是 CV_LOG_INFO，而 Release 构建（带 NDEBUG）的
# 默认日志级别是 WARNING —— modules/core/src/logger.cpp 的
#   m_defaultUnconfiguredGlobalLevel = m_isDebugBuild ? LOG_LEVEL_INFO : LOG_LEVEL_WARNING
# 而 m_isDebugBuild 在 NDEBUG 下是 false。所以不显式抬到 INFO，
# 下面 #test-result 断言的那三行 CANN 标记一行都不会出现。
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_INFO)

MODEL_URL = ('https://github.com/onnx/models/raw/main/'
             'validated/vision/classification/mobilenet/'
             'model/mobilenetv2-12.onnx')
MODEL_PATH = '/tmp/mobilenetv2-12.onnx'
MODEL_FIXTURES = (
    'tests/opencv/fixtures/mobilenetv2-12.onnx',  # 本文档仓库的副本（看护用）
    'fixtures/mobilenetv2-12.onnx',                        # 裸 checkout 时的兼容路径
)

if not os.path.exists(MODEL_PATH):
    fixture = next((p for p in MODEL_FIXTURES if os.path.exists(p)), None)
    if fixture:
        # 仓库 checkout（CI）：直接用入仓副本，零网络依赖
        shutil.copy(fixture, MODEL_PATH)
    else:
        # 读者自跑：从 onnx/models 下载；github raw 偶发 RemoteDisconnected
        # （国内网络抖动），重试 5 次，半截文件要丢掉避免下次误判已下载
        for attempt in range(5):
            try:
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
                break
            except OSError:
                if os.path.exists(MODEL_PATH):
                    os.unlink(MODEL_PATH)
                if attempt == 4:
                    raise
                time.sleep(10)

# 两个 net 实例：NPU 那份走 CANN 后端，CPU 那份用默认 classic 引擎。
# 必须分开建 —— modules/dnn/src/net_cann.cpp 的 initBackend() 开头就是
#   if (netWasConverted && netWasAllocated) return;
# 同一个 net 上第二次 forward 直接复用已编译的 OM，不会再走一遍后端初始化。
net_npu = cv2.dnn.readNetFromONNX(MODEL_PATH)
net_npu.setPreferableBackend(cv2.dnn.DNN_BACKEND_CANN)
net_npu.setPreferableTarget(cv2.dnn.DNN_TARGET_NPU)

net_cpu = cv2.dnn.readNetFromONNX(MODEL_PATH)  # 不设 backend/target => classic 引擎 + CPU

image = cv2.imread('opencv/samples/data/baboon.jpg')
x = cv2.resize(image, (224, 224)).astype(np.float32) / 255.0  # BGR, [0,1]
x = x[..., ::-1]                                              # BGR -> RGB
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)      # ImageNet mean/std
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
x = (x - mean) / std
blob = np.ascontiguousarray(x.transpose(2, 0, 1))[np.newaxis]  # 1xCxHxW float32

net_npu.setInput(blob)
net_cpu.setInput(blob)

# 捕获 OpenCV 的 INFO 日志。它们写进 C++ 的 std::cout（libstdc++ 默认让
# std::cout 与 C 的 stdout 同步，OpenCV 全树没有 sync_with_stdio(false)），
# 行首还带时间戳 + 绝对编译路径 + 函数名，没法直接拿来断言 —— 所以这里把
# fd 1 临时改到文件，跑完 forward 再刷一次缓冲取回来，最后只把「归一化后
# 的标记行」写进结果文件。
# 四次 forward（NPU 冷启 / NPU 计时 / CPU 热身 / CPU 计时）都在这个重定向
# 里跑：OpenCV 的 INFO 日志跟着一起进文件，不会污染块的 stdout ——
# stdout 要和 #test-result 逐字对齐。
LOG_PATH = '/tmp/cv_cann_log.txt'
saved_fd = os.dup(1)
log_fd = os.open(LOG_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
os.dup2(log_fd, 1)
try:
    out = net_npu.forward()          # 冷启：含 GE 图编译（0.5~1 分钟）
    t0 = time.perf_counter()
    out = net_npu.forward()          # 复用已编译的 OM —— 这才是纯推理耗时
    npu_ms = (time.perf_counter() - t0) * 1000.0
    out_cpu = net_cpu.forward()      # CPU 对照：先热身一次，让引擎把懒初始化做完
    t0 = time.perf_counter()
    out_cpu = net_cpu.forward()
    cpu_ms = (time.perf_counter() - t0) * 1000.0
finally:
    # fflush(NULL) 把 C++ std::cout 的缓冲刷进上面的文件。不刷就只能指望
    # GE 图编译 fork() 出来的子进程 exit(0) 时顺手 flush —— 而那只带得出
    # fork 之前的行，fork 之后父进程侧写的 "Compile success" 会整个丢掉。
    ctypes.CDLL(None).fflush(None)
    os.dup2(saved_fd, 1)
    os.close(log_fd)
    os.close(saved_fd)

cann_log = open(LOG_PATH, 'r', errors='replace').read()

# 这三行是「这次真的上了 NPU」的硬证据 —— top-1=372 证明不了：同一份构建
# 走默认 CPU 引擎也是 372（0.9148 vs NPU 的 0.9143）。行号见
# modules/dnn/src/net_cann.cpp。
CANN_MARKERS = (
    'DNN/CANN: converting ge::Graph to OM buffer',  # :236 逐层转完，开始编 OM
    'Compile success, model size = ',               # :348 GE 图编译成功
    'DNN/CANN: OM buffer size = ',                  # :238 OM 模型装进 CannNet
)


def _fail(msg):
    """把失败原因同时写进 stderr 和结果文件，然后退出 1。

    写结果文件是因为块的最后一条命令是 cat：调用方万一没开 set -e，
    cat 出来的内容也得跟 #test-result 对不上，不能出现「上一轮的陈旧
    文件刚好蒙对」的假通过。
    """
    sys.stderr.write(msg + '\n')
    fd = os.open('/tmp/dnn_result.txt', os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.write(fd, (msg + '\n').encode())
    os.close(fd)
    sys.exit(1)


missing = [m for m in CANN_MARKERS if m not in cann_log]
if missing:
    # 少了标记 = 这次压根没走 CANN 后端。日志尾巴打出来方便定位。
    sys.stderr.write('---- captured OpenCV log (tail 40) ----\n')
    sys.stderr.write('\n'.join(cann_log.splitlines()[-40:]) + '\n')
    _fail('CANN backend markers missing: %r' % (missing,))


# mobilenetv2-12 输出是 logits（不含 softmax），自己归一化
def _top1(out_blob):
    logits = out_blob.reshape(-1)
    e = np.exp(logits - logits.max())
    prob = e / e.sum()
    return int(np.argmax(prob)), float(np.max(prob))


idx_npu, score_npu = _top1(out)
idx_cpu, score_cpu = _top1(out_cpu)

# NPU 与 CPU 必须给出同一个答案 —— 这是「加速器算对了」唯一的硬判据。
# 光看 top-1 相同不够（分类一致不代表数值一致），所以分数也要在容差内。
if idx_npu != idx_cpu:
    _fail('NPU vs CPU top-1 mismatch: npu=#%d cpu=#%d' % (idx_npu, idx_cpu))
if abs(score_npu - score_cpu) > 0.01:
    _fail('NPU vs CPU score mismatch: npu=%.6f cpu=%.6f (tol 0.01)'
          % (score_npu, score_cpu))
if out.shape != out_cpu.shape or out.dtype != out_cpu.dtype:
    _fail('NPU vs CPU output mismatch: %s/%s vs %s/%s'
          % (out.shape, out.dtype, out_cpu.shape, out_cpu.dtype))

img = image.copy()
cv2.rectangle(img, (10, 10), (500, 500), (0, 255, 0), thickness=2)
label = 'top-1: #%d patas monkey  %.3f' % (idx_npu, score_npu)
cv2.putText(img, label, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.imwrite('/tmp/dnn-result.png', img)

# 结果不经 python 的缓冲 stdout 输出，而是用原始 syscall 写文件：
# GE 图编译会 fork 子进程，子进程 exit(0) 会把继承的 stdout 缓冲冲到
# C++ 日志行的中间（实测 "model bytes" 粘在半截 INFO 行上）。
lines = ['cann marker: %s' % m for m in CANN_MARKERS]
lines += [
    'model bytes: %d' % os.path.getsize(MODEL_PATH),
    'output shape: %s' % (out.shape,),
    'output dtype: %s' % out.dtype,
    'top class index: %d' % idx_npu,
    'top class score: %s' % score_npu,
    'cpu top class index: %d' % idx_cpu,
    'cpu top class score: %s' % score_cpu,
    'npu forward ms: %d' % round(npu_ms),
    'cpu forward ms: %d' % round(cpu_ms),
    'npu/cpu time ratio: %0.2f' % (cpu_ms / npu_ms if npu_ms > 0 else 0.0),
]
result = '\n'.join(lines) + '\n'
fd = os.open('/tmp/dnn_result.txt', os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
os.write(fd, result.encode())
os.close(fd)
sys.stdout.flush()
os._exit(0)
PY
echo
cat /tmp/dnn_result.txt
```

输出结果如下：

```shell #test-result id="dnn-cann-infer" fuzzy='xxx' fuzzy='...'
...
cann marker: DNN/CANN: converting ge::Graph to OM buffer
cann marker: Compile success, model size = xxx
cann marker: DNN/CANN: OM buffer size = xxx
model bytes: xxx
output shape: (1, 1000)
output dtype: float32
top class index: 372
top class score: 0.xxx
cpu top class index: 372
cpu top class score: 0.xxx
npu forward ms: xxx
cpu forward ms: xxx
npu/cpu time ratio: xxx
...
```

预期：`output shape (1, 1000)`、`float32`；baboon.jpg 的 top-1 是 **ImageNet-1k 第 372 类（patas / hussar monkey，赤猴）**，score ≈ 0.91（4.14.0 + CANN 9.1.0 实测 **0.9143**；换 CANN 9.2.0 在 910B4 上复测是 **0.91427**，与前者一致到小数点后 4 位；同一份构建走默认 CPU 引擎是 **0.9148**（9.2.0 复测 0.91480），NPU 与 CPU 的分差 ≈ 0.0005，远小于脚本的 0.01 容差；baboon 本尊 373 排第二 ≈ 0.03）。前三行 `cann marker:` 是「这次真的上了 NPU」的硬证据（见下一节）。

脚本同时建了**两个 net 实例**喂同一份输入：一个挂 CANN 后端 + NPU target，一个不挂后端（4.x 的默认 = classic 引擎 + CPU target）。然后硬断言两者 **top-1 相同、top-1 分数差 ≤ 0.01、输出 shape/dtype 相同** —— 加速器算错了就直接退出 1，不会把「跑完了」当成「跑对了」。`npu forward ms` / `cpu forward ms` 量的是**热启** forward（不含 GE 图编译），`npu/cpu time ratio` = CPU 耗时 ÷ NPU 耗时；**这两个数只作参考，脚本不对它们设阈值**（CI runner 的 CPU 型号和核数不固定，写死阈值必然漂）。标注结果如下：

![NPU 推理标注结果](images/dnn-result.png)

> 首次 forward 含 GE 图编译（`converting ge::Graph to OM buffer`），约 0.5~1 分钟出 `Compile success, model size = ...`；进程内二次 forward 约 1ms —— 脚本的 `npu forward ms` 量的就是这一次（第二次 forward 命中 `net_cann.cpp:107-108` 的 `if (netWasConverted && netWasAllocated) return;`，直接复用已编译的 OM，不再重跑后端初始化）。
>
> 实测（4.14.0 + CANN 9.2.0 / 910B4 / npu-4）：GE 图编译 **36.2 s**，产出的 OM 模型 **9,763,442 B**；二次 forward **1 ms**，同一份输入的 CPU 对照 **36 ms**（比值 ≈ 48×）。这两个数**只作参考**，脚本不对它们设阈值（CI runner 的 CPU 型号和核数不固定）。
>
> 2026-11-04 在 npu-4 上跑完整份文档的 doc-test 又测了一遍（这次是从头 clone + 编译 + 安装，不是复用上一轮的产物）：GE 图编译 **36.458 s**、OM 模型 **9,762,146 B**、二次 forward **1 ms**、CPU 对照 **33 ms**（比值 **45.11×**），top-1 = **372**、NPU 分数 **0.9142661690711975**、CPU 分数 **0.9148015975952148**（分差 0.0005，远小于脚本的 0.01 容差）—— 与上一轮的 0.91427 / 0.91480 一致到小数点后 4 位。整轮 doc-test 耗时 1522.6 s（25.4 min，其中编译占绝大多数）。

### 怎么证明这次真的跑在 NPU 上

**top-1 = 372 本身证明不了是 NPU 算的** —— 同一份构建走默认 CPU 引擎也是 372（score 0.9148，NPU 是 0.9143）。所以脚本把 OpenCV 自己打的 CANN 后端日志抓下来，**少一行就直接退出 1**，并在结果里输出三行归一化后的 `cann marker:`（原文全部来自 `modules/dnn/src/net_cann.cpp`）：

| 标记（脚本抓的原文） | 源码位置 | 证明了什么 |
| --- | --- | --- |
| `DNN/CANN: converting ge::Graph to OM buffer` | `net_cann.cpp:236` | 逐层转换出来的图交给了 GE，开始编 OM |
| `Compile success, model size = ...` | `net_cann.cpp:348` | GE 图编译成功、产出了 OM 模型（**fork 之后的父进程侧**） |
| `DNN/CANN: OM buffer size = ...` | `net_cann.cpp:238` | OM 模型被装进 `CannNet` 准备执行 |

这张表证明的是「**后端确实被调用了**」。但「被调用了」和「算对了」是两件事 —— 所以脚本还并行跑了一个**不挂后端的 CPU 对照**（classic 引擎 + CPU target），硬断言：top-1 相同、top-1 分数差 ≤ 0.01、输出 shape/dtype 相同。三者任一不满足就退出 1。

再往前一步，「这个 cv2 到底有没有把 CANN 后端编进去」本身也是源码级可判定的，上一节的脚本已经查了：`cv2.dnn.getAvailableTargets(cv2.dnn.DNN_BACKEND_CANN)` 返回 `DNN_TARGET_NPU` 才说明 `modules/dnn/src/registry.cpp:157-160` 里那条 `#ifdef HAVE_CANN` 的注册真的生效了。**光看 `cv2.dnn.DNN_BACKEND_CANN` 这个枚举存在是证明不了任何事的** —— `opencv2/dnn/dnn.hpp` 的 `enum Backend` 没有 `#ifdef HAVE_CANN` 守卫，不带 CANN 的构建照样打印得出来。

配套的守护脚本（`tests/opencv/test_quick_start_ascend.py`）把 **`Going back to default backend on CPU target`** 加进了 `ERROR_MARKERS`：那是 `initBackend()` 在**某一层不支持 CANN** 时打的 WARNING（`CV_LOG_ONCE_WARNING`），打完之后 `if (!newWasSupported) return;` —— **静默回落到 CPU 引擎**。这一行一旦出现，就说明这次推理压根没上 NPU，守护直接判红。

抓这三行日志要过三个坑，脚本里都显式处理了：

1. **Release 构建默认看不到 INFO**。`modules/core/src/logger.cpp` 的
   `m_defaultUnconfiguredGlobalLevel = m_isDebugBuild ? LOG_LEVEL_INFO : LOG_LEVEL_WARNING`，
   而 `m_isDebugBuild` 取决于 `NDEBUG` —— Release 带 `NDEBUG` ⇒ 默认级别是 **WARNING**，
   上面三行全是 `CV_LOG_INFO` ⇒ 不抬级别一行都看不到。脚本里显式
   `cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_INFO)`。
2. **日志行长得没法直接断言**。OpenCV 的日志行是
   `[ INFO:<线程>@<时间戳>] <绝对编译路径>:<行号> <函数名> <消息>`（`logger.cpp` 的
   `writeLogMessageEx` + `writeLogMessage`）—— 时间戳和构建路径每次都不一样。
   所以脚本把 `fd 1` 临时改到 `/tmp/cv_cann_log.txt`，跑完再从文件里挑出
   **消息正文**做归一化，而不是拿整行去比对。
3. **日志行不一定落盘**。OpenCV 的 INFO 写进 C++ 的 `std::cout`；libstdc++ 默认让
   `std::cout` 与 C 的 `stdout` 同步（OpenCV 全树没有 `sync_with_stdio(false)`），
   所以 `ctypes.CDLL(None).fflush(None)` 能把它确定性刷进那个文件。不刷就只能靠 GE 图编译
   `fork()` 出来的子进程 `exit(0)` 时顺手 flush —— 而那只带得出 **fork 之前** 的行，
   `fork` 之后父进程侧写的 `Compile success` 会整个丢掉（这行恰恰是最硬的证据）。

## 已知边界

- **没有 OpenCV 5.x**：4.14.0 没有 5.0 的新 DNN 图引擎（`onnx_importer2`），也没有 5.0 的其它新特性。反过来 4.14.0 也有 5.0 没有的东西：**reduce 层的 CANN 后端支持**（`layers/reduce_layer.cpp` 从 4.14 起才 include `op_cann.hpp`）。
- **没有 `cv2.cann`**：本文只编 OpenCV 主仓，不编 `opencv_contrib`，所以 `cv2.cann.cvtColor` / `cv2.cann.resize` 这套 contrib 经典算子加速 API 不可用。另外 contrib 的 `modules/cannops/src/cann_call.cpp` 在 4.11.0 与 4.14.0 之间**逐字节相同**（43 个文件、823,803 B），`OperatorRunner::run()` 把 `AscendStream::Null()`（`stream == nullptr`）直接喂给 `aclopCompileAndExecute`，CANN 9.1.0 断言 `stream != nullptr` ⇒ 实测 53 个 cannops 用例挂 48 个（`E29999: Assert ((stream) != nullptr)` + `Invalid_Argument_Null_Pointer(EH0008)`）。**上游还有逻辑反转 bug**：`if (rawStream == nullptr) aclrtSynchronizeStream(rawStream);` —— NULL 时反而去同步 NULL 流。要跑 `cv2.cann` 就得编 contrib + 打那组 NULL 流兜底补丁，那就不是本文这条零补丁路线了。
- **没有 `opencv_test_cannops` 单测**：那个 gtest 二进制来自 `opencv_contrib` 的 cannops 模块，本文不编 contrib，所以没有那 53 个用例的证据；NPU 正确性由上面的 **MobileNetV2 的 NPU-vs-CPU 数值对照**（top-1 相同 + 分数差 ≤ 0.01 + shape/dtype 相同）+ 三行 `cann marker:`（CANN 后端日志标记）来兜。
- **编译并行度受内存配额约束**：`all_ops.h` 那几个 TU 每个吃 3GB 上下，cgroup 32GB 的容器里 `--parallel` 别超过 6，否则 OOM killer 会杀掉 `cc1plus`。
- **「Release 就不用窄 op 头补丁」不是通用解法**：本条只在 **4.x** 上实测成立（4.14.0 的 Release 构建 dnn `.text` 53.3 MiB → 链接通过；4.11.0 是 52.0 MiB）。同样条件下 **5.0.0 实测仍然失败**：Release + `all_ops.h` 链接报 `relocation truncated to fit: R_AARCH64_JUMP26`，受害者是 protobuf 的 `AddDescriptorsRunner::AddDescriptorsRunner`（**报错对象随链接顺序随机变**，5.0.0 的 Debug 构建里同一个坑出现过 `Arena::AllocateAlignedWithCleanup` 版本 —— 所以「链接炸了但报错看着跟 CANN 无关」是常态，别顺着报错对象查根因）。也就是说：**分支范围是 aarch64 的硬约束，缩 `.text` 只是缓解，不是保证**；OpenCV 5.x 要过这一关只有窄 op 头一条路。
- **没有 `opencv_version` 等命令行工具**：它们属于 `apps/`，而 `apps` 被 `-DBUILD_opencv_apps=OFF` 关掉了（原因见「配置并编译」一节）。
- **CANN 的「裁剪版运行时」是隐形坑**：`<CANN>/aarch64-linux/devlib/linux/aarch64/` 里有一套同名但被裁剪过的 `.so`（`libgraph.so` 只有 198 KB，少 4 个 `ge::AscendString` 序关系运算符）。symlink 指错目录不会在编译期报错，只会在链接期炸成 `undefined reference to ge::AscendString::operator<`；本文「桥接 CANN 的库目录」一节里的 `nm` 检查就是为了挡住它。
- **版本钉死 4.14.0，不追 4.x 后续**：`4.14.0` 是 4.x 的末代（`git ls-remote --tags` 确认 4.x 之后就是 5.0.0），所以这个钉不会过期。但如果将来 5.x 的图引擎补上非 CPU 后端支持，5.x 那条路线会重新变得值得走。

## 外部链接

- GitHub：[opencv/opencv](https://github.com/opencv/opencv)
- 文档中心：[OpenCV 官方文档](https://docs.opencv.org/4.x/)
- 模型：[onnx/models - mobilenetv2-12](https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet/model)
