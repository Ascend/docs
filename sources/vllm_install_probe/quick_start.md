# vLLM 安装探测（勿合入）

在 CANN 9.1 镜像里用官方 pip wheel 安装 vLLM 0.23.0 与 vllm-ascend 0.23.0，并确认张量落在 NPU 上。本页只用于验证迁到 `Ascend/docs` 后的默认装法，测完即关，不要合入主干。

## 前置条件

配套镜像：`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`。

| 组件 | 版本 |
| --- | --- |
| CANN | 9.1.0 |
| vllm | 0.23.0 |
| vllm-ascend | 0.23.0 |
| torch / torch-npu | 由 vllm-ascend 装到 2.10.0 / 2.10.0.post4 |
| triton-ascend | 3.2.2 |

## 安装

先装与 vllm-ascend 同版本的 vLLM，再从昇腾 variant 源装插件。不要把两步合成一次 `pip install`。最后单独覆盖社区版 CUDA Triton。

```shell #test id="install-vllm"
python -m pip install --retries 3 vllm==0.23.0
python -m pip install \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/variant \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  vllm-ascend==0.23.0
python -m pip install --force-reinstall --no-deps \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  triton-ascend==3.2.2
python -c "from importlib.metadata import version; print('vllm', version('vllm')); print('vllm-ascend', version('vllm-ascend')); print('torch', version('torch')); print('torch-npu', version('torch-npu')); print('triton-ascend', version('triton-ascend'))"
```

输出结果如下：

```shell #test-result id="install-vllm" fuzzy='xxx'
...vllm 0.23.0
vllm-ascend 0.23.0
torch 2.10.0xxx
torch-npu 2.10.0.post4
triton-ascend 3.2.2
```

## 确认 NPU

```shell #test id="npu-tensor"
python -c "import torch, torch_npu; x = torch.zeros(1, device='npu:0'); print('device', x.device); print('npu_available', torch.npu.is_available())"
```

输出结果如下：

```shell #test-result id="npu-tensor"
device npu:0
npu_available True
```
