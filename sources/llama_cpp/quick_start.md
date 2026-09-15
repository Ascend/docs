# llama.cpp

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**）。本文覆盖单卡与多卡。上游已验证的设备列表见 [CANN 后端 — Devices](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/CANN.md)。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit + 驱动固件已安装并可 `source set_env.sh` |
| 编译工具 | cmake ≥ 3.14、g++（C++17）、make、git |
| 下载工具 | curl |
| llama.cpp | 本文从 GitHub 源码编译，见下文 |

## 1. 加载 CANN 环境

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

<!--
```shell #test-result id="check-tools"
...
cmake version ...
```
-->

## 3. 获取源码并编译

克隆 [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)，开启 CANN 后端后应生成 `llama-completion`（一次性文本生成）。同一次编译还会生成 `llama-cli` 和 `llama-server`，后面的可选步骤会用到。将 `<ref>` 换成目标分支、tag 或 commit（上游默认分支为 `master`）。

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

```shell #test id="compile" load="upstream_ref>>ref"
if [ ! -d llama.cpp/.git ]; then
  git clone https://github.com/ggml-org/llama.cpp.git
fi
cd llama.cpp
git checkout <ref>
cmake -B build -DGGML_CANN=on -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
ls build/bin/llama-completion
```

<!--
```shell #test-result id="compile"
...
build/bin/llama-completion
...
```
-->

## 4. 准备 GGUF 模型

推理输入是 GGUF 文件；910B 上 CANN 后端支持 FP16、BF16、Q8_0、Q4_0，见 [CANN 后端 — DataType](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/CANN.md)。下面从 Hugging Face 下载约 400 MB 的 Q4_0 示例，文件头应为 `GGUF`。

<!--
```shell #test-setup
set -euo pipefail
ci='/root/.cache/huggingface/llama.cpp'
cached="$ci/qwen2.5-0.5b-instruct-q4_0.gguf"
sum='7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed'
if [ -f "$cached" ]; then
  if echo "$sum  $cached" | sha256sum -c >/dev/null 2>&1; then
    cp -a "$cached" qwen2.5-0.5b-instruct-q4_0.gguf
  else
    rm -f "$cached"
  fi
fi
```
-->

```shell #test id="download-model"
if [ ! -f qwen2.5-0.5b-instruct-q4_0.gguf ]; then
  curl -fL --retry 3 --retry-delay 5 --connect-timeout 30 \
    -o qwen2.5-0.5b-instruct-q4_0.gguf \
    https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf
fi
head -c 4 qwen2.5-0.5b-instruct-q4_0.gguf
```

<!--
```shell #test-setup
set -euo pipefail
ci='/root/.cache/huggingface/llama.cpp'
src='qwen2.5-0.5b-instruct-q4_0.gguf'
sum='7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed'
echo "$sum  $src" | sha256sum -c
if [ ! -f "$ci/qwen2.5-0.5b-instruct-q4_0.gguf" ]; then
  mkdir -p "$ci"
  cp -a "$src" "$ci/qwen2.5-0.5b-instruct-q4_0.gguf.part"
  mv "$ci/qwen2.5-0.5b-instruct-q4_0.gguf.part" "$ci/qwen2.5-0.5b-instruct-q4_0.gguf"
fi
```
```shell #test-result id="download-model"
GGUF
```
-->

## 5. 推理

单卡用 `-sm none -mg <设备号>` 指定全部算子到特定卡上；多卡用 `-sm layer` 按层拆分。

| 参数 | 含义 |
| --- | --- |
| `-m` | GGUF 模型路径 |
| `-p` | 提示词 |
| `-n` | 最大生成 token 数。 |
| `--seed` | 随机种子。 |
| `-ngl` | offload 到 NPU 的层数。`99` 表示尽量全部，否则层会留在 CPU |
| `-no-cnv` | 关闭自动对话。Instruct 类 GGUF 不加此项会等待键盘输入 |
| `-sm none -mg 0` | 单卡布局：全部算子跑在 0 号卡 |
| `-sm layer` | 多卡布局：按层拆到可见的多张卡 |
| `ASCEND_RT_VISIBLE_DEVICES` | 限制进程可见的 NPU 编号 |

### 5.1 单卡推理

<!--
```shell #test id="infer-device"
cd llama.cpp && ASCEND_RT_VISIBLE_DEVICES=0 ./build/bin/llama-completion \
    -m ../qwen2.5-0.5b-instruct-q4_0.gguf \
    -p "Building a website can be done in 10 simple steps:" \
    -n 64 -no-cnv -ngl 99 -sm none -mg 0 --seed 42 -v 2>&1
```
```shell #test-result id="infer-device"
...using device CANN0...
...CANN0 model buffer size = ...
```
-->

```shell #test id="infer"
cd llama.cpp && ASCEND_RT_VISIBLE_DEVICES=0 ./build/bin/llama-completion \
    -m ../qwen2.5-0.5b-instruct-q4_0.gguf \
    -p "Building a website can be done in 10 simple steps:" \
    -n 64 -no-cnv -ngl 99 -sm none -mg 0 --seed 42 2>&1
```

完整输出较长，其中应包含：

```shell #test-result id="infer"
...
Building a website can be done in 10 simple steps: building a website is a great way to increase your online presence, and this post will help you get started. These steps are all the tools you will need to build your website, and you can learn them one at a time, and you can build a website in the 10 steps as the next step.
...
```

### 5.2 多卡推理

有两张及以上 NPU 时，可用 `-sm layer` 把层拆到多卡。下面示例暴露 0、1 号卡：

<!--
```shell #test id="infer-multi-device"
cd llama.cpp && ASCEND_RT_VISIBLE_DEVICES=0,1 ./build/bin/llama-completion \
    -m ../qwen2.5-0.5b-instruct-q4_0.gguf \
    -p "Building a website can be done in 10 simple steps:" \
    -n 64 -no-cnv -ngl 99 -sm layer --seed 42 -v 2>&1
```
```shell #test-result id="infer-multi-device"
...using device CANN0...
...using device CANN1...
...CANN0 model buffer size = ...
...CANN1 model buffer size = ...
```
-->

```shell #test id="infer-multi"
cd llama.cpp && ASCEND_RT_VISIBLE_DEVICES=0,1 ./build/bin/llama-completion \
    -m ../qwen2.5-0.5b-instruct-q4_0.gguf \
    -p "Building a website can be done in 10 simple steps:" \
    -n 64 -no-cnv -ngl 99 -sm layer --seed 42 2>&1
```

完整输出较长，其中应包含：

```shell #test-result id="infer-multi"
...
Building a website can be done in 10 simple steps: building a website is a great way to increase your online presence, and this post will help you get started. These steps are all the tools you will need to build your website, and you can learn them one at a time, and you can build a website in the 10 steps as the next step.
You
...
```

### 5.3 交互式对话

`llama-cli` 面向多轮对话，启动后会等待键盘输入。想手动体验模型时可以使用：

```shell
cd llama.cpp && ASCEND_RT_VISIBLE_DEVICES=0 ./build/bin/llama-cli \
    -m ../qwen2.5-0.5b-instruct-q4_0.gguf \
    -ngl 99 -sm none -mg 0
```

### 5.4 浏览器对话

`llama-server` 会启动本地 HTTP 服务，并自带 Web 页面。启动后用浏览器打开 http://127.0.0.1:8080 即可对话。

```shell
cd llama.cpp && ASCEND_RT_VISIBLE_DEVICES=0 ./build/bin/llama-server \
    -m ../qwen2.5-0.5b-instruct-q4_0.gguf \
    -ngl 99 -sm none -mg 0 --host 127.0.0.1 --port 8080
```

## 6. 更多文档

- 上游仓库与总说明：[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- CANN 后端的设备、精度与环境变量：[docs/backend/CANN.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/CANN.md)
- 命令行对话：[llama-cli](https://github.com/ggml-org/llama.cpp/blob/master/tools/cli/README.md)
- HTTP 服务与自带 Web 页面：[llama-server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
