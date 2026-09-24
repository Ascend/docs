# FlagScale

FlagScale 是覆盖训练、推理与服务的统一命令行工具。本文在两张昇腾卡上安装 vLLM-Ascend，用它对 [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) 做一次离线推理。

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列，芯片为 Ascend **910B**。本文示例为两张卡，tensor parallel 为 2。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit 与驱动固件已安装，并可 `source set_env.sh` |
| ATB | Ascend Transformer Boost，下文会加载它的环境脚本 |
| Python | 满足 [vLLM-Ascend 安装说明](https://docs.vllm.ai/projects/ascend/en/latest/installation.html) |
| vLLM | `0.23.0`，与下文的 `vllm-ascend` 一起安装 |
| vLLM-Ascend | `0.23.0`，见 [vLLM-Ascend 安装说明](https://docs.vllm.ai/projects/ascend/en/latest/installation.html) |
| FlagScale | 上游最新 Release tag |
| 模型 | [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) |

### 本文验证环境

下表是这次验证用的环境。软件要求见上一节。

| 项目 | 内容 |
| --- | --- |
| 镜像 | `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12` |
| 设备 | 两张 Ascend 910B |
| Python | 3.12，来自上面的镜像 |
| vLLM / vLLM-Ascend | 0.23.0 / 0.23.0 |
| torch / torch_npu | 2.10.0 / 2.10.0.post4 |
| FlagScale | Release tag `v2.0.0`，包装版本 `1.0.0` |

阅读本文前，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备好 CANN 与驱动。

## 1. 加载 CANN 环境

加载 CANN 与 ATB，并把 `/usr/local/sbin` 加入 `PATH`。

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/latest/atb/set_env.sh
export PATH=/usr/local/sbin:/usr/sbin:$PATH
```

## 2. 检查环境是否就绪

### 2.1 确认 NPU 在线

```shell
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
| 6     910B3               | OK            | 91.0        39                0    / 0             |
| 0                         | 0000:41:00.0  | 0           0    / 0          3442 / 65536         |
+===========================+===============+====================================================+
| 7     910B3               | OK            | 88.7        39                0    / 0             |
| 0                         | 0000:42:00.0  | 0           0    / 0          3439 / 65536         |
+===========================+===============+====================================================+
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| No running processes found in NPU 6                                                            |
+===========================+===============+====================================================+
| No running processes found in NPU 7                                                            |
+===========================+===============+====================================================+
```

若 `npu-smi` 找不到，回到 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 检查驱动与设备挂载。

### 2.2 确认工具可用

确认 CANN 已加载，并且 `python` 在 `PATH` 里。

```shell #test id="check-tools"
test -n "$ASCEND_HOME_PATH"
python --version
```

完整输出较长，其中应包含：

```shell #test-result id="check-tools"
Python ...
```

## 3. 安装 vLLM-Ascend

按 [vLLM-Ascend 安装说明](https://docs.vllm.ai/projects/ascend/en/latest/installation.html) 安装 `vllm` 0.23.0 和 `vllm-ascend` 0.23.0，接着以 `--force-reinstall --no-deps` 安装 `triton-ascend` 3.2.2，并卸掉 `flashinfer`。

```shell #test id="install-vllm"
python -m pip install --retries 3 vllm==0.23.0
python -m pip install \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/variant \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  vllm-ascend==0.23.0
python -m pip install --force-reinstall --no-deps \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  triton-ascend==3.2.2
python -m pip uninstall -y flashinfer flashinfer-python flashinfer-cubin
python -c "import importlib.metadata as m
for n in ['torch', 'torch-npu', 'vllm', 'vllm-ascend']:
    print(n, m.version(n))"
```

完整输出较长，其中应包含：

```shell #test-result id="install-vllm"
...
torch 2.10.0...
torch-npu 2.10.0.post4
vllm 0.23.0...
vllm-ascend 0.23.0
```

## 4. 安装 FlagScale

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

克隆 [FlagScale](https://github.com/flagos-ai/FlagScale) 到目录 `FlagScale`，按 Release tag 做可编辑安装，并安装 `hydra-core`、`omegaconf`、`typer`、`pyyaml`、`packaging`。

```shell #test id="install-flagscale" load="upstream_ref>>ref"
if [ ! -d FlagScale/.git ]; then
  rm -rf FlagScale
  GIT_TERMINAL_PROMPT=0 GIT_HTTP_VERSION=HTTP/1.1 \
    git clone --depth 1 --branch <ref> \
    https://github.com/flagos-ai/FlagScale.git FlagScale
fi
git -C FlagScale describe --tags --exact-match
python -m pip install --no-build-isolation --no-deps -e ./FlagScale
python -m pip install hydra-core omegaconf typer pyyaml packaging
python -c "from importlib.metadata import version; print('flagscale', version('flagscale'))"
```

`<ref>` 是上游最新的 Release tag。打印出来的包装版本在本次验证里是 `flagscale 1.0.0`。

完整输出较长，其中应包含：

```shell #test-result id="install-flagscale" load="upstream_ref>>ref"
<ref>
...
flagscale 1.0.0...
```

## 5. 确认昇腾插件选中了 NPU

把下面的脚本存成 `probe_ascend_platform.py`，在 0 号卡和 1 号卡上导入昇腾插件，并打印 vLLM 版本、平台名称和设备类型。

```shell #test id="probe"
cat > probe_ascend_platform.py <<'PY'
import vllm
import vllm_ascend
from vllm.platforms import current_platform as p

print('vllm', vllm.__version__)
print('platform_name', type(p).__name__)
print('device_type', p.device_type)
PY
export VLLM_PLUGINS=ascend
export VLLM_LOGGING_LEVEL=INFO
export ASCEND_RT_VISIBLE_DEVICES=0,1
python probe_ascend_platform.py 2>&1
```

完整输出较长，其中应包含：

```text #test-result id="probe"
...
Platform plugin ascend is activated
...
vllm 0.23.0
platform_name NPUPlatform
device_type npu
```

## 6. 用离线 inference 做一次双卡生成

保存实验配置到 `FlagScale/qs_conf/qwen25_05b_tp2_ascend.yaml`。这份配置选择 vLLM 后端，并使用 0 号卡和 1 号卡。

```yaml
defaults:
  - _self_
  - inference: qwen25_05b_tp2_ascend

experiment:
  exp_name: qwen25_05b
  exp_dir: qs_out
  task:
    type: inference
    backend: vllm
    entrypoint: flagscale/inference/inference_llm.py
  runner:
    hostfile: null
  envs:
    HYDRA_FULL_ERROR: 1
    ASCEND_VISIBLE_DEVICES: "0,1"
    ASCEND_RT_VISIBLE_DEVICES: "0,1"
    HCCL_WHITELIST_DISABLE: 1
    HCCL_CONNECT_TIMEOUT: 600
    PYTHONHASHSEED: 0
    VLLM_TARGET_DEVICE: "npu"
    VLLM_PLUGINS: "ascend"
    VLLM_LOGGING_LEVEL: "INFO"
    VLLM_WORKER_MULTIPROC_METHOD: "spawn"

action: run

hydra:
  run:
    dir: ${experiment.exp_dir}/hydra
```

保存推理配置到 `FlagScale/qs_conf/inference/qwen25_05b_tp2_ascend.yaml`。模型是 Qwen2.5-0.5B，tensor parallel 为 2。

```yaml
llm:
  model: Qwen/Qwen2.5-0.5B
  tokenizer: Qwen/Qwen2.5-0.5B
  trust_remote_code: true
  tensor_parallel_size: 2
  pipeline_parallel_size: 1
  gpu_memory_utilization: 0.4
  seed: 1234
  enforce_eager: true
  max_model_len: 512
  max_num_batched_tokens: 512
  max_num_seqs: 1
  disable_custom_all_reduce: true

generate:
  prompts: [
    "The first President of the United States",
  ]
  sampling:
    top_p: 0.1
    top_k: 1
    temperature: 0.0
    seed: 1234
    max_tokens: 8
```

| 项 | 含义 |
| --- | --- |
| `seed` | 写在推理配置里，值为 1234，用来固定生成文字 |
| `temperature` | 写在推理配置里，值为 0 |

<!--
```shell #test-setup
mkdir -p FlagScale/qs_conf/inference
cat > FlagScale/qs_conf/qwen25_05b_tp2_ascend.yaml <<'YAML'
defaults:
  - _self_
  - inference: qwen25_05b_tp2_ascend

experiment:
  exp_name: qwen25_05b
  exp_dir: qs_out
  task:
    type: inference
    backend: vllm
    entrypoint: flagscale/inference/inference_llm.py
  runner:
    hostfile: null
  envs:
    HYDRA_FULL_ERROR: 1
    ASCEND_VISIBLE_DEVICES: "0,1"
    ASCEND_RT_VISIBLE_DEVICES: "0,1"
    HCCL_WHITELIST_DISABLE: 1
    HCCL_CONNECT_TIMEOUT: 600
    PYTHONHASHSEED: 0
    VLLM_TARGET_DEVICE: "npu"
    VLLM_PLUGINS: "ascend"
    VLLM_LOGGING_LEVEL: "INFO"
    VLLM_WORKER_MULTIPROC_METHOD: "spawn"

action: run

hydra:
  run:
    dir: ${experiment.exp_dir}/hydra
YAML
cat > FlagScale/qs_conf/inference/qwen25_05b_tp2_ascend.yaml <<'YAML'
llm:
  model: Qwen/Qwen2.5-0.5B
  tokenizer: Qwen/Qwen2.5-0.5B
  trust_remote_code: true
  tensor_parallel_size: 2
  pipeline_parallel_size: 1
  gpu_memory_utilization: 0.4
  seed: 1234
  enforce_eager: true
  max_model_len: 512
  max_num_batched_tokens: 512
  max_num_seqs: 1
  disable_custom_all_reduce: true

generate:
  prompts: [
    "The first President of the United States",
  ]
  sampling:
    top_p: 0.1
    top_k: 1
    temperature: 0.0
    seed: 1234
    max_tokens: 8
YAML
```
-->

进入 `FlagScale` 目录，按上面的配置跑一次离线推理。

```shell #test id="inference"
export PYTHONHASHSEED=0
export VLLM_PLUGINS=ascend
export VLLM_LOGGING_LEVEL=INFO
export ASCEND_RT_VISIBLE_DEVICES=0,1
export ASCEND_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
cd FlagScale
flagscale inference qwen25_05b --config "$PWD/qs_conf/qwen25_05b_tp2_ascend.yaml" --test 2>&1
```

完整输出较长，其中应包含：

```text #test-result id="inference"
...
Platform plugin ascend is activated
...
backend=hccl
...
output.outputs[0].text=', George Washington, was born in '
...
```

## 7. 更多用法

训练、在线服务和其余模块与社区文档相同。

- 社区文档：[FlagScale 文档](https://docs.flagos.io/projects/FlagScale/en/latest/)
- 安装：[FlagScale 安装说明](https://docs.flagos.io/projects/FlagScale/en/latest/getting_started/install.html)
