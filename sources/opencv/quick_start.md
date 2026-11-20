# opencv

[OpenCV](https://github.com/opencv/opencv) 是开源计算机视觉库。本文在单卡昇腾 NPU 上把 **OpenCV 4.14** 从源码编译成带 CANN 后端的 `cv2`（编译时打开 `WITH_CANN=ON`），跑通：

- **CPU 常用接口**：读图、写图、颜色空间转换、缩放、画图、读写视频。
- **NPU 上跑模型**：用 `cv2.dnn` 加载 MobileNetV2，指定 CANN 后端 + NPU 设备，分类结果和 CPU 跑出来的一致。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，并按需完成物理机或容器内的设备挂载（`/dev/davinci*` 等）。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)），`npu-smi info` 能看到 NPU 设备
- 编译依赖：`git` / `build-essential` / `cmake` / `pkg-config` / `zlib1g-dev` / `libjpeg-dev` / `libpng-dev` / `libtiff-dev`（见「前置安装」里的 apt 那一步）

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
| OpenCV | 4.14.0（源码，`WITH_CANN=ON`，`CMAKE_BUILD_TYPE=Release`） |

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

把 CANN 的环境变量加载进来：

```shell #test-setup
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
```

确认 CANN 装好了：

```shell #test id="check-cann-path"
ls -d /usr/local/Ascend/ascend-toolkit/latest
```

输出结果如下：

```shell #test-result id="check-cann-path"
/usr/local/Ascend/ascend-toolkit/latest
```

装编译依赖（git / cmake / 编译器 / 图像格式的开发库）：

```shell #test-setup
sed -i 's|http://ports.ubuntu.com/ubuntu-ports/|https://mirrors.aliyun.com/ubuntu-ports/|g' /etc/apt/sources.list
apt-get update -qq && apt-get install -y -qq git build-essential cmake pkg-config zlib1g-dev libjpeg-dev libpng-dev libtiff-dev
```

## CANN 库目录：别链到裁剪版运行时

CANN 里有两套名字相同的动态库：`<CANN>/lib64` 里是完整的，`<CANN>/aarch64-linux/devlib/linux/aarch64/` 里是裁剪过的（`libgraph.so` 只有 198 KB，少了几个比较运算符）。链错了编译时不会报错，等到链接才失败：`undefined reference to ge::AscendString::operator<`。

一条命令就能看出拿到的是哪一套：

```shell
nm -D --defined-only <CANN>/lib64/libgraph.so | grep -c '_ZNK2ge12AscendStringlt'
```

**1** = 完整版，**0** = 裁剪版。所以给 OpenCV 的 CANN 库一律指向 `<CANN>/lib64`，**不要**指向 `devlib/...`（下一节的检查会挡住它）。

## 构建 OpenCV

### clone OpenCV

把源码下载到本地（只取 4.14.0 这一版）：

```shell #test-setup
git clone --depth 1 --branch 4.14.0 https://github.com/opencv/opencv.git
```

### 检查源码

确认拿到的版本是 4.14.0，而且源码一行都没动过：

```shell #test id="check-opencv-source"
cd opencv
git describe --tags
git status --porcelain | wc -l
```

输出结果如下：

```shell #test-result id="check-opencv-source"
4.14.0
0
```

第一行是版本号，第二行是被改动过的文件数 —— 0 说明源码和上游完全一致。

### 桥接 CANN 的库目录

OpenCV 编译时按固定路径找 CANN 的库（`acllib/lib64`、`lib64`、`compiler/lib64`），CANN 里的目录名对不上，所以先建几个软链（symlink，指向别处的目录）指过去：

```shell #test-setup
CANN_DIR=/usr/local/Ascend/ascend-toolkit/latest
[ -e "$CANN_DIR/lib64" ] || ln -sfn "$CANN_DIR/aarch64-linux/lib64" "$CANN_DIR/lib64"
mkdir -p "$CANN_DIR/acllib" "$CANN_DIR/compiler"
ln -sfn "$CANN_DIR/lib64" "$CANN_DIR/acllib/lib64"
ln -sfn "$CANN_DIR/lib64" "$CANN_DIR/compiler/lib64"
```

确认三个软链都指向**完整版** lib64：

```shell #test id="check-cann-lib-layout"
CANN_DIR=/usr/local/Ascend/ascend-toolkit/latest
test -f "$CANN_DIR/acllib/lib64/libascendcl.so" && echo "acllib/lib64/libascendcl.so ok"
test -f "$CANN_DIR/lib64/libacl_op_compiler.so" && echo "lib64/libacl_op_compiler.so ok"
test -f "$CANN_DIR/compiler/lib64/libgraph.so" && echo "compiler/lib64/libgraph.so ok"
nm -D --defined-only "$CANN_DIR/compiler/lib64/libgraph.so" | grep -c '_ZNK2ge12AscendStringlt'
```

输出结果如下：

```shell #test-result id="check-cann-lib-layout"
acllib/lib64/libascendcl.so ok
lib64/libacl_op_compiler.so ok
compiler/lib64/libgraph.so ok
1
```

### 配置并编译

`WITH_CANN=ON` 打开 CANN 后端；`-DOPENCV_DOWNLOAD_MIRROR_ID=gitcode` 让 cmake 去 gitcode 镜像下载编译时需要的第三方库（ADE / IPPICV / TBB）。装到 `/usr/local/opencv-cann-414`：

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

```shell #test-result id="check-cmake-libs"
-- CANN: libascendcl.so is found at ...
-- CANN: libgraph.so is found at ...
```

编译并安装一步完成（`--target install`）。**并行度别开高**：CANN 的 `all_ops.h` 会让每个编译单元吃掉 3GB 上下内存，容器内存是 32GB 时 `-j` 超过 6 会被 OOM killer 干掉 `cc1plus`。所以并行度做成了可覆盖的环境变量，默认 2，内存够可以调高（先看配额：`cat /sys/fs/cgroup/memory.max`）。

```shell #test-setup
cd opencv/build
cmake --build . --target install --parallel "${OPENCV_BUILD_JOBS:-2}"
```

### 检查安装结果

确认装出来的 `cv2` 能正常导入、版本是 4.14.0：

```shell #test id="verify-build"
PYTHONPATH=/usr/local/opencv-cann-414/lib/python3.12/site-packages:${PYTHONPATH:-} python -c "import cv2; print('cv2', cv2.__version__)"
```

输出结果如下：

```shell #test-result id="verify-build"
cv2 4.14.0
```

确认 dnn 库有多大（Release 构建能链接通过的关键）：

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
`xxx` 是字节数
```

补充一条上界，确认用的确实是 Release 而不是 Debug：

```shell #test id="dnn-text-size-bound"
TEXT=$(size -A /usr/local/opencv-cann-414/lib/libopencv_dnn.so.4.14.0 | awk '$1==".text"{print $2}')
echo "text bytes: $TEXT"
test "$TEXT" -gt 0 && test "$TEXT" -lt 67108864
```

输出结果如下：

```shell #test-result id="dnn-text-size-bound" fuzzy='xxx'
text bytes: xxx
```

## OpenCV Python quickstart

刚编出来的 `cv2` 装在 `/usr/local/opencv-cann-414` 下，把它加进 `PYTHONPATH`：

```shell #test-setup
export PYTHONPATH=/usr/local/opencv-cann-414/lib/python3.12/site-packages:${PYTHONPATH:-}
```


下面用 clone 下来的 OpenCV 仓库自带测试图 `opencv/samples/data/baboon.jpg`（512×512 狒狒）作样例输入：

![baboon 输入图](images/baboon-input.jpg)

### 读图 / 写图

读一张图，再写一张出来(下面的命令用 Python 执行)：

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

把彩色图转成灰度图(下面的命令用 Python 执行)：

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

把图缩到 50×200(下面的命令用 Python 执行)：

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

在图上画个矩形、写行字(下面的命令用 Python 执行)：

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

写 10 帧再读回来(下面的命令用 Python 执行)：

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

### 先确认 CANN 后端真的编进来了

`cv2.dnn` 里既有默认的 CPU 后端，也有 CANN 后端。要用 NPU，得先把 CANN 后端编进来，查一下后端列表(下面的命令用 Python 执行)：

```python #test id="dnn-backend-targets"
import cv2

print('cv2 version:', cv2.__version__)
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
`xxx` 是 `cv2.__version__`。

第一行的 `[0]` 是 CPU 后端可用设备，第二行的 `[9]` 是 CANN 后端可用设备，也就是 NPU。**如果是空列表，说明这个 `cv2` 没带 CANN 后端**，脚本会直接报错退出。
```

### MobileNetV2 ONNX 推理（NPU vs CPU 对照）

跑 MobileNetV2（onnx/models 的 `mobilenetv2-12`，PyTorch 导出，13,964,571 字节）：

> **模型从哪来**：本文档仓库里带了一份同样的副本（`tests/opencv/fixtures/mobilenetv2-12.onnx`，看护直接用它，不用联网）；自己跑时没有这份副本，就自动从 onnx/models 下载（重试 5 次）。

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

预期：`output shape (1, 1000)`、`float32`；baboon.jpg 的分类结果是 ImageNet-1k 第 372 类（patas / hussar monkey，赤猴），置信度约 0.91（NPU 0.91427，CPU 0.91480）。开头三行 `cann marker:` 是 OpenCV 自己打的日志，说明这次真的走了 NPU。

脚本建了两个 net 实例喂同一张图：一个指定 CANN 后端 + NPU，一个用默认 CPU，然后断言两者 **top-1 相同、分数差 ≤ 0.01、输出 shape/dtype 相同** —— 算错了就直接报错退出。`npu forward ms` / `cpu forward ms` 是第二次 forward 的耗时（不含 GE 图编译），只作参考，脚本不设阈值。标注结果如下：

![NPU 推理标注结果](images/dnn-result.png)

> 第一次 forward 包含 GE 图编译（`converting ge::Graph to OM buffer`），要 0.5~1 分钟才出 `Compile success, model size = ...`；同一个进程里第二次 forward 只要 1 ms —— 脚本里的 `npu forward ms` 量的就是第二次。
>
> 实测（4.14.0 + CANN 9.2.0 / 910B4）：GE 图编译 **36.5 s**，二次 forward **1 ms**，同一张图 CPU 跑 **33 ms**（约 45 倍）。

## 外部链接

- GitHub：[opencv/opencv](https://github.com/opencv/opencv)
- 文档中心：[OpenCV 官方文档](https://docs.opencv.org/4.x/)
- 模型：[onnx/models - mobilenetv2-12](https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet/model)
