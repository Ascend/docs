## 快速开始

在昇腾 NPU 上从源码构建并部署 Text Generation Inference（TGI）推理服务，
对 Qwen3-0.6B 完成端到端文本生成：单卡基线 + 双卡 HCCL 张量并行（`--num-shard 2`）。

本文档覆盖上游官方 release 中**尚未包含**的 Ascend NPU 支持：构建、安装、
启动与推理验证使用
[cosdt/text-generation-inference](https://github.com/cosdt/text-generation-inference)
（TGI 官方仓库的 fork，Ascend 适配以 release 形式发布在该 fork 上；release
不含预编译产物，因此按本文档从源码构建）。

### 前置条件

#### 硬件

Atlas 900 A2 PODc（Ascend 910B × 2），并按需完成物理机或容器内的设备挂载。

#### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- Rust 工具链（由本文档「安装 Rust 工具链」小节通过 rustup 安装）

#### 本文档示例使用的环境

**配套机器**：

- **机器类型**：Atlas 900 A2 PODc（Ascend 910B4，32 GB × 2）
- **操作系统**：Ubuntu 22.04

**配套镜像**：

swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12

### 1. 检查环境

#### 确认 NPU 设备

```shell #test-setup
npu-smi info
```

输出类似：

```text
+------------------------------------------------------------------------------------------------+
| npu-smi 25.5.2                   Version: 25.5.2                                               |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B4               | OK            | 89.9        39                0    / 0             |
| 0                         | 0000:41:00.0  | 0           0    / 0          2922 / 32768         |
| 1     910B4               | OK            | 89.9        39                0    / 0             |
| 0                         | 0000:41:00.0  | 0           0    / 0          2922 / 32768         |
+===========================+===============+====================================================+
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+---------------------------+---------------+----------------------------------------------------+
| No running processes found in NPU 0                                                            |
+---------------------------+---------------+----------------------------------------------------+
```

> 如果 `npu-smi` 不存在，请回到 [Ascend 官方快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 补装驱动。
> 本文档的双卡验证需要**至少两张卡**可见。

### 2. 准备环境

#### 系统依赖

TGI 是 Rust + Python 双栈项目：Rust 侧编译需要 C/C++ 工具链与 protobuf 编译器，
PyO3 嵌入 Python 需要开发头文件：

```shell #test-setup
apt-get update -qq
apt-get install -y -qq build-essential protobuf-compiler pkg-config libssl-dev curl python3-dev
```

确认编译必需的工具已可用：

```shell #test id="check-system-deps"
protoc --version
```

输出结果如下：

```shell #test-result id="check-system-deps"
libprotoc ...
```

#### 获取源码

从 [Releases 页面](https://github.com/cosdt/text-generation-inference/releases)下载最新 release tag 对应的源码包并解压到 `tgi/`（`<ref>` 为要安装的 tag）：

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

```shell #test-setup load="upstream_ref>>ref"
mkdir -p tgi && curl -fsSL "https://github.com/cosdt/text-generation-inference/archive/refs/tags/<ref>.tar.gz" | tar -xzf - -C tgi --strip-components=1
```

解压完成后确认关键文件齐全（`server/requirements_ascend.txt` 是 fork 的
NPU 依赖锁定文件，上游仓库没有它，可据此确认拿到的是昇腾适配源码）：

```shell #test id="check-source"
TGI_DIR=$PWD/tgi
[ -d "$TGI_DIR" ] || TGI_DIR=$PWD
for f in Cargo.toml start-tgi.sh server/requirements_ascend.txt; do [ -f "$TGI_DIR/$f" ] && echo "$f ok"; done
```

输出结果如下：

```shell #test-result id="check-source"
Cargo.toml ok
start-tgi.sh ok
server/requirements_ascend.txt ok
```

> `<ref>` 为要安装的 release tag。直接把 `<ref>` 换成
> [Releases 页面](https://github.com/cosdt/text-generation-inference/releases)上最新的 tag。

#### Python 依赖

从华为昇腾源安装官方当前推荐的 `torch_npu`（`torch` 作为其依赖一并安装，
与 CANN 的配套关系见 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)），
`modelscope` 用于第 3 节下载模型：

```shell #test-setup
python -m pip install --index-url https://repo.huaweicloud.com/ascend/repos/pypi/simple torch_npu
python -m pip install modelscope
```

#### 检查 NPU 设备运行时可用

```shell #test id="check-npu-runtime"
python -c "import torch, torch_npu; print(f'torch={torch.__version__}'); print(f'torch_npu={torch_npu.__version__}'); print('is_available:', torch.npu.is_available()); print('count:', torch.npu.device_count())"
```

输出结果如下：

```shell #test-result id="check-npu-runtime"
torch=...
torch_npu=...
is_available: True
count: 2
```

> 如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。

#### 安装 Rust 工具链

```shell #test-setup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.85.1 --profile minimal
export PATH="$HOME/.cargo/bin:$PATH"
```

确认工具链可用：

```shell #test id="check-toolchain"
export PATH="$HOME/.cargo/bin:$PATH"
rustc --version
```

输出结果如下：

```shell #test-result id="check-toolchain"
rustc 1.85.1 ...
```

> TGI 仓库根目录的 `rust-toolchain.toml` 固定 1.85.1，因此这里直接安装该版本。

### 3. 下载基础模型

默认使用 **ModelScope** 下载 Qwen3-0.6B（约 1.2 GB，落到默认缓存
`~/.cache/modelscope`），下载后自动检查关键文件齐全
（`model.safetensors` 为权重文件，约 1.2 GB），缺文件直接报错退出：

```shell #test-setup store="model_path"
MODEL_DIR=$(python -c "from modelscope import snapshot_download; print(snapshot_download('Qwen/Qwen3-0.6B'))" | tail -n 1)
for f in config.json model.safetensors tokenizer.json; do
  [ -f "$MODEL_DIR/$f" ] && echo "$f ok" >&2 || { echo "missing: $f" >&2; exit 1; }
done
echo "$MODEL_DIR"
```

输出结果如下（`ok` 提示行在 stderr，最后一行 stdout 是模型目录）：

```text
config.json ok
model.safetensors ok
tokenizer.json ok
~/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master
```

### 4. 构建并安装 TGI

#### 编译 Rust 二进制

编译 launcher 与 router（`--profile release-opt` 为上游提供的发布优化 profile）。
PyO3 嵌入 Python 需要 `PYO3_PYTHON` 指向环境里的 Python，
protobuf 代码生成需要 `PROTOC`：

```shell #test-setup
cd tgi
export PYO3_PYTHON="$(command -v python)"
export PROTOC="$(command -v protoc)"
cargo build --profile release-opt -p text-generation-launcher -p text-generation-router-v3
```

#### 安装 Python server

`server/` 是 TGI 的 Python 侧模型实现层（`text_generation_server` 包），
router 运行时加载它。fork 为 Ascend 维护了依赖锁定文件
`server/requirements_ascend.txt`（验证栈对应的完整依赖列表，含 kernels、
transformers、accelerate 等），安装使用 uv（上游 TGI 的构建流程同样使用
uv）：

```shell #test-setup
cd tgi
python -m pip install -q uv
uv pip install --no-build-isolation -r server/requirements_ascend.txt
uv pip install --no-build-isolation --no-deps -e server
uv pip install --no-build-isolation "grpcio-tools==1.84.0" "mypy-protobuf==3.6.0"
make -C server gen-server-raw
```

> `-e server` 加 `--no-deps`：依赖一律以锁定文件为准，避免按 pyproject 的
> 宽松版本范围重新解析引入漂移。proto 代码生成用 `gen-server-raw`（只做
> 编译）：`gen-server` 目标会额外按过期的 `requirements_gen.txt` 安装依赖，
> 覆盖锁定文件的版本（例如把 `typing_extensions` 降级到 4.13，与
> `pydantic-core` 不兼容，server 启动即崩溃）。`grpcio-tools` 固定
> `1.84.0`，与锁定文件的 `grpcio` 版本一致。

> 锁定文件里 kernels 固定 0.5.0：它是 server 的构建插件，0.5.0 自带
> `kernels.lockfile`，构建时不会去下载 CUDA 专属内核；更新的版本缺该文件，
> 在无 CUDA 的 aarch64 昇腾机器上会构建失败。

#### 检查构建产物

Rust 侧产物为两个可执行文件，都在 `tgi/target/release-opt/` 下：
`text-generation-launcher` 与 `text-generation-router`（注意
`-p text-generation-router-v3` 是包名，产出的二进制叫
`text-generation-router`，launcher 运行时按这个名字拉起它）。
`--version` 输出编译期写入的版本号（版本号随 fork 的 workspace 版本
变化，未随 tag 固定，以实际输出为准），`import text_generation_server`
验证 Python server 已装进当前环境：

```shell #test id="check-build"
TGI_DIR=$PWD/tgi
[ -d "$TGI_DIR" ] || TGI_DIR=$PWD
"$TGI_DIR/target/release-opt/text-generation-launcher" --version
"$TGI_DIR/target/release-opt/text-generation-router" --version
python -c "import text_generation_server; print('server import ok')"
```

输出结果如下：

```shell #test-result id="check-build"
text-generation-launcher ...
text-generation-router-v3 ...
server import ok
```

### 5. 启动服务并验证推理

服务生命周期由 fork 仓库自带的 `start-tgi.sh` / `stop-tgi.sh` 脚本管理。
下面分单卡与双卡两节阐述，每步给出命令与可验证的输出结果。
以下命令均在「获取源码」解压出的 `tgi` 目录下执行。

#### 5.1 单卡

##### 启动服务

`start-tgi.sh` 负责后台启动、就绪轮询与日志落盘（默认 `/tmp/tgi.log`），
脚本返回 `[READY]` 即服务可用：

```shell #test-setup
cd tgi
./start-tgi.sh --num-shard 1 --devices 0
```

> 不传 `--model-id` 时默认使用 Qwen/Qwen3-0.6B，首次启动自动经 ModelScope
> 下载（缓存于 `~/.cache/modelscope`，之后直接命中）；也可用
> `--model-id /path/to/model` 指定本地路径。

启动成功的输出：

```text
[START] model_id=/path/to/model
[START] num_shard=1  devices=0  port=8080
[START] launcher pid 12345
[READY] TGI is serving on http://127.0.0.1:8080 after ~130s
[READY] stop with: ./stop-tgi.sh
```

**冷启动约 2-3 分钟是正常现象**，`[READY]` 出现前服务不可访问，耗时主要在：

- 模型权重加载并搬运到 NPU 显存（Qwen3-0.6B 约 1.2 GB）；
- KV cache 分配与模型预热（`Warming up model` 阶段）。

期间可另开终端观察 `tail -f /tmp/tgi.log`——依次出现 `Warming up model`、
`KV-cache blocks`、`Connected` 即服务就绪（时间戳与数值随机器而异）：

```text
...  INFO text_generation_router_v3: Warming up model
...  INFO text_generation_launcher: KV-cache blocks: ..., size: ...
...  INFO text_generation_router::server: Connected
```

随后 `start-tgi.sh` 的就绪轮询打印 `[READY] TGI is serving on
http://127.0.0.1:8080 after ~130s`——~130s 为 910B 实测冷启动时长，
不同机器略有差异，属正常范围。

##### 确认服务就绪

返回模型信息 JSON 即就绪。取其中与本文档固定相关的关键字段
（`router` 为路由二进制名、`version` 为编译期版本，
`max_input_tokens` / `max_total_tokens` 来自 `start-tgi.sh` 默认值）：

```shell #test id="check-info"
curl -4s http://127.0.0.1:8080/info | python -c 'import json,sys; d=json.load(sys.stdin); print("router:", d["router"]); print("version:", d["version"]); print("max_input_tokens:", d["max_input_tokens"]); print("max_total_tokens:", d["max_total_tokens"])'
```

输出结果如下：

```shell #test-result id="check-info"
router: text-generation-router
version: ...
max_input_tokens: 100
max_total_tokens: 128
```

> 为空则说明服务尚未就绪或已退出，用 `tail -50 /tmp/tgi.log` 查看原因。

##### 发起推理

```shell #test-setup store="reply_single"
curl -4fs http://127.0.0.1:8080/generate \
  -H 'Content-Type: application/json' \
  -d '{"inputs":"What is 1+1? Answer:","parameters":{"max_new_tokens":16,"do_sample":false}}' \
  | python -c 'import json, sys; print(json.load(sys.stdin)["generated_text"].strip().replace("\n", " "))'
```

`do_sample=false` 为贪心解码，相同输入输出确定。示例输出（该回复同时
被保存为基线，供双卡一节对比）：

```text
2  The question is: What is the sum of the numbers 1
```

基线回复的断言（非空、以 `2` 开头——`do_sample=false` 贪心解码下，
1+1 的输出确定以答案 `2` 开头）：

```shell #test id="smoke-tp1" load="reply_single>>reply_single"
REPLY="<reply_single>"
[ -n "$REPLY" ] || { echo "empty reply"; exit 1; }
[ "${REPLY:0:1}" = "2" ] || { echo "unexpected reply: $REPLY"; exit 1; }
echo "TGI-TP1-OK: $REPLY"
```

输出结果如下（`...` 为模型回复内容）：

```shell #test-result id="smoke-tp1"
TGI-TP1-OK: 2  The question is: ...
```

OpenAI 兼容接口：`curl -4 http://127.0.0.1:8080/v1/chat/completions`。

##### 停止服务

```shell #test-setup
cd tgi
./stop-tgi.sh
```

脚本输出 `[STOP] done.`（无实例在跑则为 `[STOP] nothing was running.`），
内部已核对 NPU 并等待进程退出。确认进程数应为 `0`：

```shell #test id="check-residual-tp1"
npu-smi info 2>/dev/null | grep -c text-generation || true
```

输出结果如下（`0` 表示无残留进程）：

```shell #test-result id="check-residual-tp1"
0
```

> 若输出非 `0`（残留进程会占用 NPU 与端口，导致新实例报
> `EJ0003 Failed to bind the IP port`），用 `./stop-tgi.sh --force` 清理。

#### 5.2 双卡 HCCL 张量并行

##### 启动服务

双卡 HCCL 张量并行（冷启动比单卡略久，多一步 HCCL 组网初始化；
启动前先等单卡服务完全释放端口）：

```shell #test-setup
cd tgi
for i in $(seq 1 30); do
  curl -4fs http://127.0.0.1:8080/info >/dev/null 2>&1 || break
  sleep 2
done
./start-tgi.sh --num-shard 2 --devices 0,1
```

启动成功的输出（`num_shard=2  devices=0,1`；`[READY]` 行与单卡一致，
`after ~XXXs` 通常比单卡的 ~130s 略久）：

```text
[START] model_id=/path/to/model
[START] num_shard=2  devices=0,1  port=8080
[START] launcher pid 12345
[READY] TGI is serving on http://127.0.0.1:8080 after ~XXXs
[READY] stop with: ./stop-tgi.sh
```

##### 验证（与单卡基线对比）

用两张卡（`--num-shard 2`，HCCL 张量并行）跑一次相同请求，先校验
`/info` 的关键字段，再验证回复非空、且与单卡基线**逐字一致**
（贪心解码下张量并行不改变输出）：

```shell #test id="smoke-tp2" load="reply_single>>reply_single"
curl -4fs http://127.0.0.1:8080/info | python -c 'import json,sys; d=json.load(sys.stdin); print("info:", d["router"], d["version"], d["max_input_tokens"], d["max_total_tokens"])'
REPLY=$(curl -4fs http://127.0.0.1:8080/generate \
  -H 'Content-Type: application/json' \
  -d '{"inputs":"What is 1+1? Answer:","parameters":{"max_new_tokens":16,"do_sample":false}}' \
  | python -c 'import json, sys; print(json.load(sys.stdin)["generated_text"].strip().replace("\n", " "))')
[ -n "$REPLY" ] || { echo "empty reply"; exit 1; }
[ "$REPLY" = "<reply_single>" ] || { echo "TP2 reply differs from single-card baseline"; echo "single: <reply_single>"; echo "tp2:    $REPLY"; exit 1; }
echo "TGI-TP2-OK: $REPLY"
```

输出结果如下（`...` 为模型回复内容，与单卡基线一致）：

```shell #test-result id="smoke-tp2"
info: text-generation-router ... 100 128
TGI-TP2-OK: 2  The question is: ...
```

##### 停止服务

```shell #test-setup
cd tgi
./stop-tgi.sh
```

输出 `[STOP] done.`（无实例在跑则为 `[STOP] nothing was running.`），
内部已核对 NPU 并等待进程退出。确认两张卡上均无残留进程：

```shell #test id="check-residual-tp2"
npu-smi info 2>/dev/null | grep -c text-generation || true
```

输出结果如下（`0` 表示无残留进程）：

```shell #test-result id="check-residual-tp2"
0
```

> 若输出非 `0`，用 `./stop-tgi.sh --force` 清理。
