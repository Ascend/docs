# Mooncake

本文在两张昇腾 NPU 上从源码编译 [Mooncake](https://github.com/kvcache-ai/Mooncake) Transfer Engine，并用 Ascend Direct 做一次设备间写传输。

## 前置条件

### 硬件

Atlas 800T 或 900 A2 训练系列，芯片为 Ascend 910B。本文示例为两张卡：一张跑 target，一张跑 initiator。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit 与驱动已安装，并能 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| 设备网卡配置 | `/etc/hccn.conf` 存在。驱动安装时写入；容器里把宿主机这份文件挂进来 |
| 编译工具 | cmake、g++、make、git、pkg-config |
| 依赖库 | glog、gflags、libibverbs、jsoncpp、yaml-cpp、OpenSSL、libcurl，见第 3 节 |

**配套机器**：Atlas 900 A2，双卡 910B。**配套镜像**：`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`。

## 1. 加载 CANN 环境

```shell
export PATH=/usr/local/sbin:/usr/local/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

Ascend Direct 会读 `/etc/hccn.conf` 里每张卡的设备网卡 IP。没有这份文件时，后面的传输会在 ADXL 初始化阶段失败。

```shell #test id="hccn"
ls /etc/hccn.conf
```

输出结果如下：

```shell #test-result id="hccn"
/etc/hccn.conf
```

## 2. 确认 NPU 在线

```shell
npu-smi info
```

至少两张卡。找不到 `npu-smi` 时，回到 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 检查驱动与设备挂载。

## 3. 安装编译依赖

下列包提供 Transfer Engine 链接所需的头文件与库。

```shell #test id="deps"
apt-get update
apt-get install -y --no-install-recommends \
    build-essential cmake git pkg-config \
    libgoogle-glog-dev libgflags-dev libibverbs-dev \
    libjsoncpp-dev libnuma-dev libyaml-cpp-dev \
    libssl-dev libcurl4-openssl-dev
ls /usr/include/glog/logging.h /usr/include/gflags/gflags.h
```

输出结果如下：

```shell #test-result id="deps"
...
/usr/include/gflags/gflags.h
/usr/include/glog/logging.h
```

## 4. 获取源码并编译 Ascend Direct

克隆上游仓库，检出要用的 ref，打开 `-DUSE_ASCEND_DIRECT=ON`，只编译 `transfer_engine_ascend_direct_perf`。将 `<ref>` 换成目标分支、tag 或 commit，上游仓库默认分支为 `main`。
<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

```shell #test id="compile" load="upstream_ref>>ref"
if [ ! -d Mooncake/.git ]; then
  GIT_HTTP_VERSION=HTTP/1.1 git clone --depth 1 --branch <ref> \
    https://github.com/kvcache-ai/Mooncake.git
fi
cd Mooncake
if [ ! -f extern/pybind11/CMakeLists.txt ]; then
  GIT_HTTP_VERSION=HTTP/1.1 git submodule update --init --depth 1 extern/pybind11
fi
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_ASCEND_DIRECT=ON \
    -DBUILD_EXAMPLES=ON \
    -DBUILD_UNIT_TESTS=OFF \
    -DWITH_STORE=OFF \
    -DWITH_STORE_RUST=OFF \
    -DWITH_EP=OFF \
    -DWITH_P2P_STORE=OFF \
    -DUSE_ETCD=OFF \
    -DUSE_REDIS=OFF
cmake --build build --target transfer_engine_ascend_direct_perf -j$(nproc)
ls build/mooncake-transfer-engine/example/transfer_engine_ascend_direct_perf
```

输出结果如下：

```shell #test-result id="compile"
...
build/mooncake-transfer-engine/example/transfer_engine_ascend_direct_perf
...
```

`-DWITH_STORE=OFF` 与 `-DUSE_ETCD=OFF` 关掉这次用不到的 Store 和元数据后端。完整组件请按上游社区 [Build Guide](https://kvcache-ai.github.io/Mooncake/getting_started/build.html) 打开对应选项。

## 5. 在两张 NPU 之间做一次写传输

例程是双进程：先启动 target，在 NPU 0 上注册设备内存并监听；再启动 initiator，在 NPU 1 上把一块 device buffer 写到 target。`P2PHANDSHAKE` 会给 target 选一个实际端口，initiator 的 `--segment_id` 必须填日志里那一行 `listening on <IP>:<port>`。

`block_iteration=1`、`batch_size=2`、`block_size=16384` 把传输规模压小，正式测带宽再按上游社区 [Ascend Direct Transport](https://kvcache-ai.github.io/Mooncake/design/transfer-engine/ascend_direct_transport.html) 加大。glog 默认打到 stderr，所以命令末尾有 `2>&1`。

```shell #test id="transfer"
cd Mooncake
export GLOG_logtostderr=1
build/mooncake-transfer-engine/example/transfer_engine_ascend_direct_perf \
    --mode=target \
    --device_logicid=0 \
    --local_server_name=127.0.0.1:12345 \
    --metadata_server=P2PHANDSHAKE \
    --block_iteration=1 \
    --batch_size=2 \
    --block_size=16384 \
    > /tmp/mooncake-target.log 2>&1 &
target_pid=$!
for _ in $(seq 1 60); do
    if ! kill -0 "$target_pid" 2>/dev/null; then
        echo "target exited before listen" >&2
        cat /tmp/mooncake-target.log >&2
        exit 1
    fi
    endpoint=$(grep -Eo 'listening on [^[:space:]]+:[0-9]+' /tmp/mooncake-target.log | tail -1 | awk '{print $3}')
    if [ -n "$endpoint" ]; then
        break
    fi
    sleep 1
done
if [ -z "$endpoint" ]; then
    echo "target did not print a listening endpoint" >&2
    cat /tmp/mooncake-target.log >&2
    kill "$target_pid" 2>/dev/null || true
    exit 1
fi
build/mooncake-transfer-engine/example/transfer_engine_ascend_direct_perf \
    --mode=initiator \
    --device_logicid=1 \
    --local_server_name=127.0.0.1:12346 \
    --metadata_server=P2PHANDSHAKE \
    --segment_id="$endpoint" \
    --operation=write \
    --block_iteration=1 \
    --batch_size=2 \
    --block_size=16384 \
    2>&1 | tee /tmp/mooncake-initiator.log
xfer_ec=${PIPESTATUS[0]}
kill "$target_pid" 2>/dev/null || true
wait "$target_pid" 2>/dev/null || true
if grep -qE 'getTransferStatus FAILED|Sync data transfer timeout|Failed to install Ascend transport' \
    /tmp/mooncake-initiator.log /tmp/mooncake-target.log; then
    echo "transfer reported FAILED/TIMEOUT or Ascend transport failed to install" >&2
    exit 1
fi
exit "$xfer_ec"
```

输出结果如下：

```shell #test-result id="transfer"
...Success to initialize adxl engine:...
...submit transfer suc.
...Test completed: duration ...
```

initiator 日志里的 `Success to initialize adxl engine` 表示这次走了 Ascend Direct。`Test completed:` 表示这一轮写传输跑完。上游社区例程在传输失败时仍可能打印后两句并返回 0，所以上面的命令会再扫 `getTransferStatus FAILED`、`Sync data transfer timeout` 和 `Failed to install Ascend transport`。

二进制默认 `--local_server_name` 指向实验室地址，必须改成 `127.0.0.1`。
