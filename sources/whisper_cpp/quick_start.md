# whisper.cpp

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**）。本文示例为单卡。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit + 驱动固件已安装并可 `source set_env.sh` |
| 编译工具 | cmake、g++（C++17）、make、git |
| whisper.cpp | 本文从 GitHub 源码编译，见下文 |

## 1. 加载 CANN 环境

`cmake` 配置阶段会调用 `npu-smi` 探测 SoC 型号；常见容器里该命令在 `/usr/local/sbin`。

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
```

## 2. 检查环境是否就绪

### 2.1 确认 NPU 在线

```shell
npu-smi info
```

:::{note}
如果 `npu-smi` 找不到，回到 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 检查驱动与设备挂载。
:::

### 2.2 确认 CANN 与编译工具

下面确认 CANN 已加载，并且 `npu-smi` 与 `cmake` 都在 `PATH` 里。

```shell #test id="check-tools"
test -n "$ASCEND_HOME_PATH"
command -v npu-smi
cmake --version
```

输出结果如下：

```shell #test-result id="check-tools"
...
cmake version ...
```

## 3. 获取源码并编译

克隆 [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp)，开启 CANN 后端后应生成 `whisper-cli`。将 `<ref>` 换成目标分支、tag 或 commit（上游默认分支为 `master`）。

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

```shell #test id="compile" load="upstream_ref>>ref"
if [ ! -d whisper.cpp/.git ]; then
  git clone https://github.com/ggml-org/whisper.cpp.git
fi
cd whisper.cpp
git checkout <ref>
cmake -B build -DGGML_CANN=on -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
ls build/bin/whisper-cli
```

输出结果如下：

```shell #test-result id="compile"
...
build/bin/whisper-cli
...
```

## 4. 准备模型

推理输入是 ggml 格式权重。下面从 Hugging Face 下载英文小模型 `tiny.en`（约 75 MB）到仓库的 `models/` 目录，并用文件头校验：正确 ggml 文件的前 4 字节为 `lmgg`。

<!--
```shell #test-setup
set -euo pipefail
ci='/root/.cache/huggingface/whisper_cpp'
cached="$ci/ggml-tiny.en.bin"
sum='921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f'
mkdir -p whisper.cpp/models
if [ -f "$cached" ]; then
  if echo "$sum  $cached" | sha256sum -c >/dev/null 2>&1; then
    cp -a "$cached" whisper.cpp/models/ggml-tiny.en.bin
  else
    rm -f "$cached"
  fi
fi
```
-->

```shell #test id="download-model"
cd whisper.cpp
mkdir -p models
if [ ! -f models/ggml-tiny.en.bin ]; then
  curl -fL --retry 3 --retry-delay 5 --connect-timeout 30 \
    -o models/ggml-tiny.en.bin \
    https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin
fi
head -c 4 models/ggml-tiny.en.bin
```

<!--
```shell #test-setup
set -euo pipefail
ci='/root/.cache/huggingface/whisper_cpp'
src='whisper.cpp/models/ggml-tiny.en.bin'
sum='921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f'
echo "$sum  $src" | sha256sum -c
if [ ! -f "$ci/ggml-tiny.en.bin" ]; then
  mkdir -p "$ci"
  cp -a "$src" "$ci/ggml-tiny.en.bin.part"
  mv "$ci/ggml-tiny.en.bin.part" "$ci/ggml-tiny.en.bin"
fi
```
-->

输出结果如下：

```shell #test-result id="download-model"
lmgg
```

## 5. 转写

仓内自带样例 `samples/jfk.wav`（16-bit WAV）。日志里的 `whisper_backend_init_gpu: using CANN0 backend` 表示模型已在 NPU 上初始化。

| 参数 | 含义 |
| --- | --- |
| `-m` | ggml 模型路径 |
| `-f` | 输入音频 |
| `-t` | 推理线程数。首次验证用 `4` |
| `--device` | 使用的 NPU 编号 |
| `ASCEND_RT_VISIBLE_DEVICES` | 限制进程可见的 NPU 编号 |

### 5.1 单卡转写

```shell #test id="transcribe"
cd whisper.cpp && ASCEND_RT_VISIBLE_DEVICES=0 ./build/bin/whisper-cli \
    -m models/ggml-tiny.en.bin \
    -f samples/jfk.wav \
    -t 4 --device 0 2>&1
```

输出结果如下：

```shell #test-result id="transcribe"
...
whisper_backend_init_gpu: using CANN0 backend
...ask not what your country...
```

### 5.2 HTTP 推理服务（可选）

`whisper-server` 把转写包成 HTTP 服务，启动后不会自己退出。

```shell
cd whisper.cpp && ASCEND_RT_VISIBLE_DEVICES=0 ./build/bin/whisper-server \
    -m models/ggml-tiny.en.bin \
    -t 4 --device 0 \
    --host 127.0.0.1 --port 8080
```

服务就绪后打开 `http://127.0.0.1:8080`，或向 `POST /inference` 上传音频。