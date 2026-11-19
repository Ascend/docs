# opencv

[OpenCV](https://github.com/opencv/opencv) 是开源计算机视觉库。本文在单卡昇腾 NPU 上把 **OpenCV 4.x**源码编译成带 CANN 后端的 `cv2`（`WITH_CANN=ON`），跑通：

- **CPU 核心 API**：读图 / 写图 / 颜色空间 / 缩放 / 绘制 / 视频读写；
- **NPU DNN 推理**：`cv2.dnn` + `DNN_BACKEND_CANN` + `DNN_TARGET_NPU`，MobileNetV2 在 npu 上跑出与 CPU 参考一致的 top-1；

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
| OpenCV | **4.14.0**（源码，`WITH_CANN=ON`，`CMAKE_BUILD_TYPE=Release`） |

本文在上述版本上于 2026-11-04 完整跑过一遍（22 个步骤全绿，1522.6 s）：零补丁是真的（`git status --porcelain` 输出 0 行），推理是真的上卡（三行 CANN 后端标记齐全 + NPU 与 CPU top-1 / 分数一致）。

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

## CANN 库目录：别链到裁剪版运行时

`<CANN>/aarch64-linux/devlib/linux/aarch64/` 里有一套同名但被裁剪过的 `.so`（`libgraph.so` 只有 198 KB，少了 `ge::AscendString` 的比较运算符）。链错目录不会在编译期报错，只会在**链接期**炸成 `undefined reference to ge::AscendString::operator<`。

判定一眼可看：

```shell
nm -D --defined-only <CANN>/lib64/libgraph.so | grep -c '_ZNK2ge12AscendStringlt'
```

**1** = 完整版，**0** = 裁剪版。所以给 OpenCV 的 CANN 软链一律指向 `<CANN>/lib64`，**不要**指向 `devlib/...`（下一节的 `nm` 检查会挡住它）。

## 构建 OpenCV（4.14.0，零补丁）

### clone OpenCV（版本钉死 4.14.0）

本文没有 `<ref>` 占位符 —— 版本钉死在 4.14.0。

```shell #test-setup
git clone --depth 1 --branch 4.14.0 https://github.com/opencv/opencv.git
```

### 确认源码是干净的（零补丁的证据）

打完补丁的树一定会有 diff；没打补丁的树 `git status` 是空的。

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

注意 **不要**指向 `devlib/...`（裁剪过的设备侧运行时，见上一节）。

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
`ln -sfn` 到已存在的软链是「改指向」而不是「往目录里塞链接」，重复执行安全。

**不想往 CANN 安装目录里写东西**（只读挂载 / 无 root）时，等价做法是把路径喂给 cmake：`-Dfound_lib_ascendcl=... -Dfound_lib_acl_op_compiler=... -Dfound_libacldvppmpi=... -Dfound_lib_graph=... -Dfound_lib_ge_compiler=... -Dfound_libgraph_base=...`（`find_library` 命中已存在的 cache 变量就不再搜目录）。本文的构建走的是软链那条路线。
```

### 配置并编译

CANN 后端开关 `WITH_CANN=ON`（不是 `BUILD_CANN`）；`-DOPENCV_DOWNLOAD_MIRROR_ID=gitcode` 让 cmake configure 阶段走 gitcode 镜像拉 ADE / IPPICV / TBB。

**两个取舍点**：

- `CMAKE_BUILD_TYPE=Release` 而不是 `Debug` —— Debug 的 `-O0` 会把 CANN `all_ops.h` 里每个 op 的内联函数都落盘，dnn `.text` 撑爆 aarch64 的分支范围；
- `-DBUILD_opencv_apps=OFF` —— `apps/model-diagnostics` 是**可执行文件**，要直接链 CANN 的 `libge_compiler.so` / `libgraph.so` / `libgraph_base.so`，而这三个库自身还有一堆没被链进来的依赖：共享库链接允许留未定义符号（`libopencv_dnn.so` 没事），可执行文件不允许 → **链接期报几百条 `undefined reference` 并让整个 `all` 目标失败**（CANN 9.2.0 实测）。

安装前缀用 `/usr/local/opencv-cann-414`。

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

编译 + 安装一体（`--target install`）。**并行度别开高**：`all_ops.h` 那几个 TU 每个要吃掉 3GB 上下内存，容器内存配额（cgroup limit）只有 32GB 时，`-j` 超过 6 会被 OOM killer 直接干掉 `cc1plus`。所以并行度做成可覆盖的环境变量：默认 `--parallel 2`（保守），内存富裕时可以调高（先看配额：`cat /sys/fs/cgroup/memory.max`）。

```shell #test-setup
cd opencv/build
cmake --build . --target install --parallel "${OPENCV_BUILD_JOBS:-2}"
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

顺便看一眼 dnn 库的实际体积 —— 这是「Release 构建能一次链接通过」的证据：

> 实测（4.14.0 + gcc 11.4 + Release -O3）：dnn `.text` = **55,920,432 B（≈53.3 MiB，CANN 9.1.0）** / **56,123,568 B（≈53.5 MiB，CANN 9.2.0）**，两次独立构建逐字节相同。同一份源码 Debug(-O0) 下 `.text` 会涨到 516 MB，超出 aarch64 的 ±128 MiB 分支窗口。

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
`xxx` 是字节数。这条断言**必须只输出两行** —— `awk` 把 `.text` 与 `Total` 钉死（`grep -E` 会连带捞进 `.text.hot` 之类的段）。中间不能插一行独占的 `...`：doc-test 的 placeholder 是通配符而不是行省略。
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

### 视频读写（videoio）

本文这版 `BUILD_LIST` **带了** `videoio`，所以 `cv2.VideoWriter` / `cv2.VideoCapture` 直接可用。下面写 10 帧再读回来：

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

> **模型来源**：本文档仓库里带了一份同字节副本（`tests/opencv/fixtures/mobilenetv2-12.onnx`，看护直接用它，避免 GitHub raw 在国内网络的抖动）；自己跑时没有这份副本，自动回落到从 onnx/models 下载（重试 5 次）。

> **不需要 `OPENCV_FORCE_DNN_ENGINE=1`**：4.14.0 的默认引擎就是 classic 引擎，`setPreferableBackend` / `setPreferableTarget` 直接生效。

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
# 默认日志级别是 WARNING —— 不显式抬到 INFO，下面断言的三行 CANN 标记一行都不会出现。
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
        # 读者自跑：从 onnx/models 下载；github raw 偶发断连，重试 5 次，
        # 半截文件要丢掉避免下次误判已下载
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

# 两个 net 实例：NPU 那份走 CANN 后端，CPU 那份用默认 classic 引擎。必须分开建 ——
# net_cann.cpp 的 initBackend() 开头就是 `if (netWasConverted && netWasAllocated) return;`，
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

# 捕获 OpenCV 的 INFO 日志：它们写进 C++ 的 std::cout，行首带时间戳 + 绝对编译
# 路径 + 函数名，没法直接断言 —— 所以把 fd 1 临时改到文件，跑完 forward 再刷一次
# 缓冲取回来，只把归一化后的标记行写进结果文件。四次 forward 全在这个重定向里跑，
# OpenCV 的 INFO 日志跟着进文件，不污染块的 stdout（stdout 要和 #test-result 对齐）。
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
    # fflush(NULL) 把 C++ std::cout 的缓冲刷进上面的文件。不刷就只能指望 GE 图
    # 编译 fork() 出来的子进程 exit(0) 时顺手 flush —— 而那只带得出 fork 之前的行，
    # fork 之后父进程侧写的 "Compile success" 会整个丢掉。
    ctypes.CDLL(None).fflush(None)
    os.dup2(saved_fd, 1)
    os.close(log_fd)
    os.close(saved_fd)

cann_log = open(LOG_PATH, 'r', errors='replace').read()

# 这三行是「这次真的上了 NPU」的硬证据 —— top-1=372 证明不了：同一份构建走默认
# CPU 引擎也是 372（0.9148 vs NPU 的 0.9143）。行号见 modules/dnn/src/net_cann.cpp。
CANN_MARKERS = (
    'DNN/CANN: converting ge::Graph to OM buffer',  # :236 逐层转完，开始编 OM
    'Compile success, model size = ',               # :348 GE 图编译成功
    'DNN/CANN: OM buffer size = ',                  # :238 OM 模型装进 CannNet
)


def _fail(msg):
    """把失败原因同时写进 stderr 和结果文件，然后退出 1。

    写结果文件是因为块的最后一条命令是 cat：万一没开 set -e，cat 出来的内容也
    得跟 #test-result 对不上，不能出现「上一轮的陈旧文件刚好蒙对」的假通过。
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

# 结果不经 python 的缓冲 stdout 输出，而是用原始 syscall 写文件：GE 图编译会
# fork 子进程，子进程 exit(0) 会把继承的 stdout 缓冲冲到 C++ 日志行的中间。
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

预期：`output shape (1, 1000)`、`float32`；baboon.jpg 的 top-1 是 **ImageNet-1k 第 372 类（patas / hussar monkey，赤猴）**，score ≈ 0.91（NPU 实测 0.91427，CPU 对照 0.91480，分差 ≈ 0.0005）。前三行 `cann marker:` 是「这次真的上了 NPU」的硬证据（见下一节）。

脚本同时建了**两个 net 实例**喂同一份输入：一个挂 CANN 后端 + NPU target，一个不挂后端（classic 引擎 + CPU target），然后硬断言两者 **top-1 相同、top-1 分数差 ≤ 0.01、输出 shape/dtype 相同** —— 加速器算错了就直接退出 1。`npu forward ms` / `cpu forward ms` 量的是**热启** forward（不含 GE 图编译），**只作参考，脚本不对它们设阈值**。标注结果如下：

![NPU 推理标注结果](images/dnn-result.png)

> 首次 forward 含 GE 图编译（`converting ge::Graph to OM buffer`），约 0.5~1 分钟出 `Compile success, model size = ...`；进程内二次 forward 约 1ms —— 脚本的 `npu forward ms` 量的就是这一次（第二次 forward 命中 `net_cann.cpp` 的 `if (netWasConverted && netWasAllocated) return;`，直接复用已编译的 OM）。
>
> 实测（4.14.0 + CANN 9.2.0 / 910B4）：GE 图编译 **36.5 s**，二次 forward **1 ms**，同一份输入的 CPU 对照 **33 ms**（比值 ≈ 45×）。

### 怎么证明这次真的跑在 NPU 上

**top-1 = 372 本身证明不了是 NPU 算的** —— 同一份构建走默认 CPU 引擎也是 372（score 0.9148，NPU 是 0.9143）。所以脚本把 OpenCV 自己打的 CANN 后端日志抓下来，**少一行就直接退出 1**，并在结果里输出三行归一化后的 `cann marker:`（原文全部来自 `modules/dnn/src/net_cann.cpp`）：

| 标记（脚本抓的原文） | 源码位置 | 证明了什么 |
| --- | --- | --- |
| `DNN/CANN: converting ge::Graph to OM buffer` | `net_cann.cpp:236` | 逐层转换出来的图交给了 GE，开始编 OM |
| `Compile success, model size = ...` | `net_cann.cpp:348` | GE 图编译成功、产出了 OM 模型（**fork 之后的父进程侧**） |
| `DNN/CANN: OM buffer size = ...` | `net_cann.cpp:238` | OM 模型被装进 `CannNet` 准备执行 |

这张表证明的是「**后端确实被调用了**」；「算对了」由上面那个 CPU 对照兜住。

配套的守护脚本（`tests/opencv/test_quick_start_ascend.py`）还把 **`Going back to default backend on CPU target`** 加进了 `ERROR_MARKERS`：那是 `initBackend()` 在**某一层不支持 CANN** 时打的 WARNING，打完之后直接 `return` —— **静默回落到 CPU 引擎**。这一行一旦出现，说明这次推理压根没上 NPU，守护直接判红。

抓这三行日志要过三个坑，脚本里都显式处理了：

1. **Release 构建默认看不到 INFO** —— Release 带 `NDEBUG` ⇒ 默认日志级别是 WARNING，而这三行全是 `CV_LOG_INFO` ⇒ 脚本里显式抬到 `LOG_LEVEL_INFO`；
2. **日志行没法直接断言** —— 行首带时间戳和绝对编译路径，每次都不一样 ⇒ 脚本把 `fd 1` 临时改到文件，跑完只取**消息正文**做归一化；
3. **日志行不一定落盘** —— INFO 写进 C++ 的 `std::cout`，不显式 `fflush(NULL)` 就只能靠 GE 图编译 `fork()` 出来的子进程 `exit(0)` 时顺手 flush，而那只带得出 **fork 之前** 的行，`fork` 之后父进程侧写的 `Compile success` 会整个丢掉（这行恰恰是最硬的证据）。

## 注意事项

- **没有 `opencv_version` 等命令行工具**：它们属于 `apps/`，而 `apps` 被 `-DBUILD_opencv_apps=OFF` 关掉了（原因见「配置并编译」一节），本文直接看装出来的库。

## 外部链接

- GitHub：[opencv/opencv](https://github.com/opencv/opencv)
- 文档中心：[OpenCV 官方文档](https://docs.opencv.org/4.x/)
- 模型：[onnx/models - mobilenetv2-12](https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet/model)
