# Mooncake

本文在昇腾 NPU 上安装社区官方包 [mooncake-transfer-engine-npu](https://pypi.org/project/mooncake-transfer-engine-npu/)，并用 TransferEngine 在两张 NPU 之间传输一块数据。

阅读本文前，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备好 CANN 与驱动。

## 前置条件

### 硬件

Atlas 800T 或 900 A2 训练系列，芯片为 Ascend 910B。本文使用两张 NPU：接收端在 0 号卡，发送端在 1 号卡。

### 软件


| 类别       | 要求                                                                                                                                                  |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| CANN     | toolkit 与驱动已安装，并能 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`。版本按 [昇腾软件配套清单](https://www.hiascend.com/developer/download/compatibility) 选择 |
| Python   | 落在社区 Quick Start 给出的范围内；NPU 包要求 `>=3.9`，社区文档写 3.10 或更新                                                                                              |
| PyTorch  | 安装与当前 CANN 配套的 `torch_npu`。版本按上面的配套清单选择，命令里不写死版本号                                                                                                   |
| 运行时库     | Ubuntu 上需要 `libcurl4`、`libibverbs1`、`rdma-core`、`librdmacm1`、`libnuma1`、`liburing2`                                                                 |
| Mooncake | 安装社区当前发布的 `mooncake-transfer-engine-npu`，见 [Quick Start](https://kvcache-ai.github.io/Mooncake/getting_started/quick-start.html)                    |
| 设备网卡配置   | `/etc/hccn.conf` 存在。驱动安装时写入；容器里把宿主机这份文件挂进来                                                                                                          |


本文验证环境：Atlas 900 A2、两张 Ascend 910B、Python 3.12、torch 2.12.0、torch_npu 2.12.0、mooncake-transfer-engine-npu 0.3.13.post1、镜像 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`。这不是唯一支持组合。

## 1. 加载 CANN 环境

```shell
export PATH=/usr/local/sbin:/usr/local/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

Ascend 传输会读 `/etc/hccn.conf` 里每张卡的设备网卡 IP。没有这份文件时，后面的传输初始化会失败。

```shell #test id="hccn"
ls /etc/hccn.conf
```



## 2. 确认 NPU 在线

```shell
npu-smi info
```

至少两张卡。找不到 `npu-smi` 时，回到 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 检查驱动与设备挂载。

## 3. 安装运行时库


```shell #test id="deps"
apt-get update
apt-get install -y --no-install-recommends \
    libcurl4 libibverbs1 rdma-core librdmacm1 libnuma1 liburing2
dpkg -s libibverbs1 libcurl4 librdmacm1 libnuma1 liburing2
```



## 4. 安装 PyTorch NPU 栈

```shell #test id="install-torch"
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  torch_npu numpy pyyaml
python -c "import numpy, yaml, torch, torch_npu; n = torch.npu.device_count(); assert torch.npu.is_available() and n >= 2, (torch.npu.is_available(), n); print('npu_available', torch.npu.is_available())"
```



## 5. 安装 Mooncake NPU 包

```shell #test id="install"
python -m pip install mooncake-transfer-engine-npu
python -c "from mooncake.store import MooncakeDistributedStore; from mooncake.engine import TransferEngine; print('mooncake_npu_import', 'ok')"
```



## 6. 使用 Ascend Direct 在两张 NPU 之间传输

本例通过公开的 Python API 模拟推理服务迁移一块 KV Cache 数据。接收端占用 0 号 NPU，发送端占用 1 号 NPU，传输协议使用 `ascend`。

接收端在 0 号卡上登记 65536 字节，并把实际监听地址写到 `mooncake_te_handshake.txt`。

```python
import time

import torch
import torch_npu
from mooncake.engine import TransferEngine

torch.npu.set_device(0)
nbytes = 65536
buf = torch.zeros(nbytes, dtype=torch.uint8, device="npu:0")

engine = TransferEngine()
engine.initialize("127.0.0.1:16001", "P2PHANDSHAKE", "ascend", "")
engine.register_memory(buf.data_ptr(), buf.nbytes, "*")

endpoint = f"127.0.0.1:{engine.get_rpc_port()}"
with open("mooncake_te_handshake.txt", "w", encoding="utf-8") as handle:
    handle.write(f"{endpoint}\n{buf.data_ptr()}\n{nbytes}\n{buf.device}\n")
print(f"target_ready {buf.device} {endpoint}", flush=True)
while True:
    time.sleep(60)
```



发送端先把整块数据复制下来，作为传输前的内容。写入接收端后清空本卡缓冲，再把同一块数据读回来。

```python #test id="transfer"
import torch
import torch_npu
from mooncake.engine import TransferEngine

lines = open("mooncake_te_handshake.txt", encoding="utf-8").read().splitlines()
endpoint, ptr_s, nbytes_s, target_device = lines[:4]
remote_ptr = int(ptr_s)
nbytes = int(nbytes_s)

torch.npu.set_device(1)
src = torch.full((nbytes,), 0x5A, dtype=torch.uint8, device="npu:1")
before = src.detach().cpu().clone()

engine = TransferEngine()
engine.initialize("127.0.0.1:16002", "P2PHANDSHAKE", "ascend", "")
engine.register_memory(src.data_ptr(), src.nbytes, "*")
engine.transfer_sync_write(endpoint, src.data_ptr(), remote_ptr, nbytes)
src.zero_()
torch.npu.synchronize()
engine.transfer_sync_read(endpoint, src.data_ptr(), remote_ptr, nbytes)
torch.npu.synchronize()
after = src.detach().cpu()
matched = int((before == after).sum())

shown = " ".join(f"{int(value):02x}" for value in before[:8])
readback = " ".join(f"{int(value):02x}" for value in after[:8])
print(f"target_device {target_device}")
print(f"initiator_device {src.device}")
print(f"before {shown}")
print(f"after {readback}")
print(f"matched_bytes {matched}")
print(f"total_bytes {before.numel()}")
```

输出结果如下：

```text #test-result id="transfer"
target_device npu:0
initiator_device npu:1
before 5a 5a 5a 5a 5a 5a 5a 5a
after 5a 5a 5a 5a 5a 5a 5a 5a
matched_bytes 65536
total_bytes 65536
```



## 7. 更多文档

Store、vLLM / SGLang 对接和多机部署与社区相同，按社区手册继续即可。

- 上游仓库：[kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake)
- 社区 Quick Start 与 NPU 安装：[Quick Start](https://kvcache-ai.github.io/Mooncake/getting_started/quick-start.html)
- 从源码编译与后端选项：[Build Guide](https://kvcache-ai.github.io/Mooncake/getting_started/build.html)
- Ascend Direct 传输：[Ascend Direct Transport](https://kvcache-ai.github.io/Mooncake/design/transfer-engine/ascend_direct_transport.html)
- Store 部署与调优：[Mooncake Store Deployment](https://kvcache-ai.github.io/Mooncake/deployment/mooncake-store-deployment-guide.html)
- 对接 SGLang：[SGLang Integration](https://kvcache-ai.github.io/Mooncake/deployment/integrations/sglang/)
- 对接 vLLM：[vLLM Integration](https://kvcache-ai.github.io/Mooncake/deployment/integrations/vllm/)

