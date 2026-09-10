# FlagScale

在两张昇腾卡上安装 vLLM-Ascend，用 FlagScale 对 [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) 做一次离线推理。

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列，芯片为 Ascend **910B**。本文示例为两张卡，tensor parallel 为 2。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit + 驱动固件已安装，并可 `source set_env.sh` |
| ATB | Ascend Transformer Boost。vLLM 子进程要加载 `libatb.so` |
| Python | 3.12 |
| vLLM | `0.23.0` |
| vLLM-Ascend | `0.23.0`，注册名 `ascend` |
| FlagScale | 上游 Release tag，撰写时为 `v2.0.0` |
| 模型 | [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) |

阅读本文前，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备好 CANN 与驱动。推荐配套镜像：`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`。

## 1. 加载 CANN 环境

常见容器里 `npu-smi` 在 `/usr/local/sbin`，需要把该目录加入 `PATH`。后面的 vLLM 还依赖 ATB。

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

:::{note}
表格里应能看到至少两张卡。若 `npu-smi` 找不到，回到 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 检查驱动与设备挂载。
:::

### 2.2 确认工具可用

下面确认 CANN 已加载，并且 `npu-smi` 与 `python` 都在 `PATH` 里。

```shell #test id="check-tools"
test -n "$ASCEND_HOME_PATH"
command -v npu-smi >/dev/null
python --version
```

输出结果如下：

```shell #test-result id="check-tools"
Python 3.12...
```

## 3. 安装 vLLM-Ascend

分三步安装，不要合成一次 `pip install`。先装社区 vLLM `0.23.0`，再装 `vllm-ascend==0.23.0` 换成昇腾 `torch` 栈。最后一步必须 `--force-reinstall --no-deps`，否则会留下社区 CUDA 版 Triton。社区 wheel 会带上 CUDA 通信库 `flashinfer`，昇腾上没有 `libcudart`，双卡初始化会失败，所以装完后卸掉。

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

输出结果如下：

```shell #test-result id="install-vllm"
...
torch 2.10.0...
torch-npu 2.10.0.post4
vllm 0.23.0...
vllm-ascend 0.23.0
```

## 4. 安装 FlagScale

将 `<ref>` 换成目标 Release tag。克隆目录用 `FlagScale`，不要用当前目录下的 `flagscale/`，以免挡住已安装的包。`--no-deps` 避免元数据去拉一份不匹配的 torch。CLI 还需要 `hydra-core`、`omegaconf` 和 `typer`。`v2.0.0` 是 GitHub Release 的 tag 名；包装版本仍是 `1.0.0`，所以下面打印出来的是 `flagscale 1.0.0`。

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

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

输出结果如下：

```shell #test-result id="install-flagscale" load="upstream_ref>>ref"
<ref>
...
flagscale 1.0.0...
```

## 5. 确认昇腾插件选中了 NPU

推理前设 `VLLM_PLUGINS=ascend`，让 vLLM 选用昇腾平台插件。

保存为 `probe_ascend_platform.py`：

```python
import vllm
import vllm_ascend
from vllm.platforms import current_platform as p

print('vllm', vllm.__version__)
print('platform_name', type(p).__name__)
print('device_type', p.device_type)
```

<!--
```shell #test-setup
cat > probe_ascend_platform.py <<'PY'
import vllm
import vllm_ascend
from vllm.platforms import current_platform as p

print('vllm', vllm.__version__)
print('platform_name', type(p).__name__)
print('device_type', p.device_type)
PY
```
-->

```shell #test id="probe"
export VLLM_PLUGINS=ascend
export VLLM_LOGGING_LEVEL=INFO
export ASCEND_RT_VISIBLE_DEVICES=0,1
python probe_ascend_platform.py 2>&1
```

输出结果如下：

```shell #test-result id="probe"
...Platform plugin ascend is activated...
platform_name NPUPlatform
device_type npu
...
```

`device_type` 必须是 `npu`，`platform_name` 必须是 `NPUPlatform`。若这里是 `cuda` / `cpu`，或没有出现 `Platform plugin ascend is activated`，先回到第 3–4 节，不要开始推理。

## 6. 用离线 inference 做一次双卡生成

下面的 yaml 把规模收到几分钟内可结束，仅供首次验证，不是生产配置。模型 id 走 Hugging Face Hub。`--test` 让推理在前台跑完才返回；末尾的 `2>&1` 把打在 stderr 的设备日志并进标准输出。

保存为 `FlagScale/qs_conf/qwen25_05b_tp2_ascend.yaml`：

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

保存为 `FlagScale/qs_conf/inference/qwen25_05b_tp2_ascend.yaml`：

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

输出结果如下：

```shell #test-result id="inference"
...Platform plugin ascend is activated...backend=hccl...
output.outputs[0].text=...
```

生成的文字每次可能不同，不必和样例一致。`Platform plugin ascend is activated`、`backend=hccl` 是这次走昇腾与两卡 HCCL 的证据；退出码 0 不算数，必须看到 `output.outputs[0].text=`。
