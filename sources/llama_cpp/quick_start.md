# 快速开始：在昇腾 NPU 上用 llama.cpp 做推理

阅读本文前，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备好 CANN 与驱动。下面从源码编译 llama.cpp 的 CANN 后端，下载一份 GGUF，完成单卡和双卡文本生成。

[llama.cpp](https://github.com/ggml-org/llama.cpp) 是面向 GGUF 的轻量推理引擎。昇腾侧通过 `-DGGML_CANN=on` 把计算调度到 NPU，设备在日志里显示为 `CANN0`。

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**）。本文覆盖单卡与双卡。上游已验证的设备列表见 [CANN 后端 — Devices](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/CANN.md)。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit + 驱动固件已安装并可 `source set_env.sh` |
| 编译工具 | cmake ≥ 3.14、g++（C++17）、make、git |
| 下载工具 | curl |
| llama.cpp | 本文从 GitHub 源码编译，见下文 |

## 1. 加载 CANN 环境

`cmake` 配置阶段会调用 `npu-smi` 探测 SoC 型号；常见容器布局里该命令在 `/usr/local/sbin`。

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
```

## 2. 检查环境是否就绪

### 2.1 确认 NPU 在线

```shell
npu-smi info
```

退出码为 0 即表示设备可见；表格中的功耗与 HBM 占用每次不同。

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

克隆 [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)，开启 CANN 后端后应生成 `llama-completion`（一次性文本生成）。将 `<ref>` 换成目标分支、tag 或 commit（上游默认分支为 `master`）。

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

输出结果如下：

```shell #test-result id="compile"
...
build/bin/llama-completion
...
```

首次全量编译可能耗时较久（数十分钟量级，视机器而定）。

## 4. 准备模型（GGUF）

推理输入是 GGUF 文件；910B 上 CANN 后端支持 FP16、BF16、Q8_0、Q4_0（见 [CANN 后端 — DataType](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/CANN.md)）。下面从 Hugging Face 下载约 400 MB 的 Q4_0 示例，并用文件头前 4 字节 `GGUF` 校验。

```shell #test id="download-model"
if [ ! -f qwen2.5-0.5b-instruct-q4_0.gguf ]; then
  curl -fL --retry 3 --retry-delay 5 --connect-timeout 30 \
    -o qwen2.5-0.5b-instruct-q4_0.gguf \
    https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf
fi
head -c 4 qwen2.5-0.5b-instruct-q4_0.gguf
```

输出结果如下：

```shell #test-result id="download-model"
GGUF
```

## 5. 推理

单卡用 `-sm none -mg <设备号>` 把全部算子钉在指定卡上；多卡用 `-sm layer` 按层拆分。

| 参数 | 含义 |
| --- | --- |
| `-m` | GGUF 模型路径 |
| `-p` | 提示词 |
| `-n` | 最大生成 token 数。首次验证用 `64`，正式跑再改大 |
| `-ngl` | offload 到 NPU 的层数。`99` 表示尽量全部，否则层会留在 CPU |
| `-no-cnv` | 关闭自动对话。Instruct 类 GGUF 不加此项会等待键盘输入 |
| `-v` | 提高日志详细度，确认权重是否加载到 NPU |
| `-sm none -mg 0` | 单卡布局：全部算子跑在 0 号卡 |
| `-sm layer` | 多卡布局：按层拆到可见的多张卡 |
| `ASCEND_RT_VISIBLE_DEVICES` | 限制进程可见的 NPU 编号 |

### 5.1 单卡推理

模型路径 `../qwen2.5-0.5b-instruct-q4_0.gguf` 相对仓库根目录。命令末尾的 `2>&1` 把日志和生成文本打到同一路，便于核对设备。

```shell #test id="infer"
cd llama.cpp && ASCEND_RT_VISIBLE_DEVICES=0 ./build/bin/llama-completion \
    -m ../qwen2.5-0.5b-instruct-q4_0.gguf \
    -p "Building a website can be done in 10 simple steps:" \
    -n 64 -no-cnv -ngl 99 -sm none -mg 0 -v 2>&1
```

输出结果如下：

```shell #test-result id="infer"
...using device CANN0...
...CANN0 model buffer size = ...
```

### 5.2 多卡推理

有两张及以上 NPU 时，可用 `-sm layer` 把层拆到多卡。下面示例暴露 0、1 号卡：

```shell #test id="infer-multi"
cd llama.cpp && ASCEND_RT_VISIBLE_DEVICES=0,1 ./build/bin/llama-completion \
    -m ../qwen2.5-0.5b-instruct-q4_0.gguf \
    -p "Building a website can be done in 10 simple steps:" \
    -n 64 -no-cnv -ngl 99 -sm layer -v 2>&1
```

输出结果如下：

```shell #test-result id="infer-multi"
...using device CANN0...
...using device CANN1...
...CANN0 model buffer size = ...
...CANN1 model buffer size = ...
```

### 5.3 交互式对话（可选）

`llama-cli` 面向多轮对话，启动后会等待键盘输入。想手动体验模型时可以使用：

```shell
cd llama.cpp && ASCEND_RT_VISIBLE_DEVICES=0 ./build/bin/llama-cli \
    -m ../qwen2.5-0.5b-instruct-q4_0.gguf \
    -ngl 99 -sm none -mg 0
```

## 6. 下一步

| 目标 | 参考 |
| --- | --- |
| CANN 环境变量、性能调优、完整模型支持表 | 上游 [docs/backend/CANN.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/CANN.md) |
| HTTP / OpenAI 兼容 API 服务 | 编译产物 `llama-server` |
| Hugging Face → GGUF 转换与量化 | 上游 `convert_hf_to_gguf.py`、`llama-quantize` |

## 故障排查

| 现象 | 可能原因 | 建议 |
| --- | --- | --- |
| `cmake` 报 SoC 探测失败 | 未 `source set_env.sh`，或 `npu-smi` 不在 `PATH` | 重做第 1–2 节 |
| 编译很久无报错 | 可能在拉 Hugging Face Web UI 资源 | 加 `-DLLAMA_USE_PREBUILT_UI=OFF` 后重新配置 |
| `head -c 4` 不是 `GGUF` | 下载到了 HTML 错误页 | 检查 URL、网络与磁盘空间 |
| 推理退出 0 但无 `CANN0` | NPU 未挂载或层未 offload 到 CANN | 检查 `ASCEND_RT_VISIBLE_DEVICES`、`-ngl`、`-v` 日志 |
| 双卡推理只有 `CANN0`、没有 `CANN1` | 只暴露了一张卡，或未加 `-sm layer` | 检查可见设备是否为 `0,1`，以及 `-sm` |
| `llama-completion` 卡住不输出 | 未加 `-no-cnv`，进入对话等待 | 加上 `-no-cnv` 或使用 `-p` 一次性生成 |
