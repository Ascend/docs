# areno

在单卡昇腾 NPU 上安装 AReno，并用 GSPO 算法在 GSM8K 上跑通「安装 → RLVR 训练 → checkpoint 落盘 → OpenAI 兼容推理服务」的最小链路。

本文档是上游 README「Tiny training smoke test」与 [Ascend NPU 后端文档](https://github.com/inclusionAI/AReno/blob/main/docs/concepts/npu.rst)的昇腾落地版：安装走 NPU 源码路径（安装器检测到 `torch_npu` 后自动切换 NPU 依赖清单并编译 `areno.accel._areno_accel_npu` 加速扩展），训练/推理显式指定 `--backend npu`。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，单卡即可（本文示例 Qwen3-0.6B，bf16 权重约 1.2 GB）。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- Python ≥ 3.10
- CANN 9.0.0，且包含**开发组件**：AReno 安装时会现场编译 Ascend C 内核，需要 toolkit 中的 `tikcpp/ascendc_kernel_cmake`、`libascendcl` 等（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- `cmake`、`nm`（binutils）：NPU 内核构建与符号校验工具

### 本文档示例使用的版本

**配套机器**：

- **机器类型**：Atlas 900 A2 PODc（Ascend 910B4，32 GB × 1）
- **操作系统**：Ubuntu 22.04

**配套镜像**：

swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-910b-ubuntu22.04-py3.11

**软件版本**（torch ↔ torch_npu ↔ CANN 按 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)三方兼容矩阵选择，与上游 NPU 文档一致）：

| 组件 | 版本 |
| --- | --- |
| Python | 3.11 |
| CANN | 9.0.0 |
| torch | 2.10.0+cpu |
| torch_npu | 2.10.0.post2 |
| AReno | 最新 release tag（v0.9.0 起已含 NPU 后端；工作流注入 `UPSTREAM_REF`） |
| 模型 | [Qwen/Qwen3-0.6B](https://www.modelscope.cn/Qwen/Qwen3-0.6B)（ModelScope，AReno 默认 hub） |
| 数据集 | `gsm8k:main`（ModelScope） |

### 前置安装

确认能看到 NPU 设备：

```shell
npu-smi info
```

输出类似：

```
+------------------------------------------------------------------------------------------------+
| npu-smi 25.5.2                   Version: 25.5.2                                               |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 5     910B4               | OK            | 89.9        39                0    / 0             |
| 0                         | 0000:41:00.0  | 0           0    / 0          2922 / 32768         |
+===========================+===============+====================================================+
```

> 如果 `npu-smi` 不存在，请回到 [Ascend 官方快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 补装驱动。

检查 Python 版本：

```shell #test id="check-py"
python --version
```

输出结果如下：

```shell #test-result id="check-py" fuzzy='xxx'
Python 3.11.xxx
```

安装 `torch` / `torch_npu`：

```shell #test-setup
uv pip install -f https://mirrors.aliyun.com/pytorch-wheels/cpu torch==2.10.0
uv pip install --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi torch_npu==2.10.0.post2
```

检查 torch / torch_npu 是否装好且 NPU 设备可用：

```shell #test id="check-torch"
python -c "import torch, torch_npu; print('torch=', torch.__version__); print('torch_npu=', torch_npu.__version__); print('is_available:', torch.npu.is_available()); print('count:', torch.npu.device_count())"
```

输出结果如下：

```shell #test-result id="check-torch"
torch= 2.10.0+cpu
torch_npu= 2.10.0.post2
is_available: True
count: 1
```

> 如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。

补齐构建工具与构建后端（AReno 安装时会现场编译 Ascend C 内核与 torch_npu 扩展，`--no-build-isolation` 要求环境里已有 setuptools ≥ 69）：

```shell #test-setup
command -v cmake >/dev/null 2>&1 || { apt-get update -qq >/dev/null; apt-get install -y -qq cmake binutils >/dev/null; }
command -v cmake && command -v nm
uv pip install -U 'setuptools>=69' wheel
```

## 安装 AReno

AReno 的 NPU 支持随**源码安装**生效：安装器检测到 `torch_npu` 后自动改用 `requirements/npu.txt` 依赖清单（保留你 CANN 配套的 torch / torch_npu / triton-ascend，不引入 CUDA 包），并编译 `areno.accel._areno_accel_npu` 扩展——该扩展无 Python 回退，训练与推理都依赖它，所以必须先装好 torch / torch_npu 再装 AReno。

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

克隆上游仓库并 checkout 到 release tag（工作流注入的 `UPSTREAM_REF`），安装并验证：

```shell #test id="install-areno" load="upstream_ref>>ref"
git clone --depth 1 --branch <ref> https://github.com/inclusionAI/AReno.git
cd AReno
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
uv pip install -e . --no-build-isolation
python -c "from importlib.metadata import version; print('areno', version('areno'))"
```

> AReno 自 v0.9.0 起在 release 中自带 NPU 后端，本文档直接安装工作流注入的最新 release tag（`UPSTREAM_REF`）；此前 v0.0.8 无 NPU、临时看护 `main` 的处理已回切为 release。

输出结果类似如下：

```shell #test-result id="install-areno" fuzzy='xxx'
areno xxx
```

- xxx 表示实际安装的版本号

## 安装验证

装好后验证三层：Python 包可导入、NPU 加速扩展已编译、CLI 入口可用：

```shell #test id="areno-import-check"
cd AReno
python -c "
import importlib.util as u
specs = {m: u.find_spec(m) for m in ['areno', 'areno.accel', 'areno.accel._areno_accel_npu']}
for m, s in specs.items():
    print(m, 'ok' if s is not None else 'MISSING')
"
areno train --help >/dev/null && echo "cli_train_ok"
areno serve --help >/dev/null && echo "cli_serve_ok"
```

输出结果如下：

```shell #test-result id="areno-import-check"
areno ok
areno.accel ok
areno.accel._areno_accel_npu ok
cli_train_ok
cli_serve_ok
```

> `areno.accel._areno_accel_npu` 是上一步安装时编译出的 NPU 加速扩展。若显示 MISSING，说明安装时没有走到 NPU 分支——检查 `torch_npu` 是否先于 AReno 安装。

## RLVR 训练（GSPO on GSM8K）

对齐 README「Tiny training smoke test」：Qwen3-0.6B 在 GSM8K 上跑 GSPO，一条命令串起 CLI、数据集加载（`examples/math/dataset_loader.py` 把 GSM8K 的 `question`/`answer` 规整成训练 prompt）、奖励函数（`examples/math/math_verify_reward.py` 用 math-verify 校验 `\boxed{}` 答案）、rollout 与训练步。

本文档为控制 CI 资源把训练压到 1 步（`--max-steps 1`）、生成长度压到 256 token（`--max-new-tokens 256`），并让 checkpoint 落盘（`--save-path` + `--save-interval 1`）供下一节推理服务复用；想跑完整训练，去掉这几个参数即可。注意力后端显式选择 `--attn-backend native`：安装依赖 `flash-attn-npu==0.3.0` 存在时默认的 flash 解码路径会在 NPU graph capture 中触发 `aclrtMemcpy` 107030（capture 模式不支持该操作），native 路径无此问题。模型与数据集走 ModelScope（AReno 默认 hub），首次运行自动下载：

```shell #test-setup id="train-smoke-run"
cd AReno
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -o pipefail
ASCEND_RT_VISIBLE_DEVICES=0 areno train \
  --ckpt Qwen/Qwen3-0.6B \
  --dataset-path gsm8k:main \
  --dataset-loader-fn examples/math/dataset_loader.py \
  --reward-fn-path examples/math/math_verify_reward.py \
  --algo gspo \
  --backend npu \
  --attn-backend native \
  --tp-size 1 \
  --world-size 1 \
  --batch-size 1 \
  --max-steps 1 \
  --max-new-tokens 256 \
  --save-path outputs/qwen3-gspo \
  --save-interval 1 \
  2>&1 | tee /tmp/areno_train_smoke.log
```

训练日志中的 loss、奖励等数值每次都不一样，没法写死预期值，只检查链路关键阶段的标记（rollout 完成 → 奖励计算 → 训练步完成 → checkpoint 落盘 → 步数上限退出）与 checkpoint 目录落盘：

```shell #test id="train-smoke"
ls -d AReno/outputs/qwen3-gspo/step_000001
grep -E 'stage=rollout_end|metric=reward_mean|train_stats=|stage=save_checkpoint_end|stage=max_steps_reached' /tmp/areno_train_smoke.log \
  | sed -E 's/^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2} [A-Z]+ [^ ]+ [^ ]+ - //' \
  | sed -E 's/(metric=reward_mean value=).*/\1.../; s/(train_stats=)\{.*\}/\1{...}/; s|(stage=save_checkpoint_end path=).*|\1...|'
```

输出结果如下：

```shell #test-result id="train-smoke"
AReno/outputs/qwen3-gspo/step_000001
epoch=0 step=0 role=policy stage=rollout_end
epoch=0 step=0 metric=reward_mean value=...
epoch=0 step=0 train_stats={...}
epoch=0 step=0 stage=save_checkpoint_end path=...
epoch=0 step=1 stage=max_steps_reached
```

> `...` 处为随训练变化的数值与路径；`train_stats` 的完整内容是本步训练指标字典。

捕获 checkpoint 路径供推理服务复用：

```shell #test-setup store="areno_ckpt"
realpath AReno/outputs/qwen3-gspo/step_000001
```

## 推理服务

用训练产物启动 OpenAI 兼容服务（后台运行，就绪后返回 `{"status": "ok"}`）：

```shell #test-setup id="serve-start" load="areno_ckpt>>ckpt"
cd AReno
source /usr/local/Ascend/ascend-toolkit/set_env.sh
ASCEND_RT_VISIBLE_DEVICES=0 areno serve \
  --model-path <ckpt> \
  --backend npu \
  --attn-backend native \
  --max-running-prompts 2 \
  --tp-size 1 \
  --world-size 1 \
  --port 8000 > /tmp/areno_serve.log 2>&1 &
echo $! > /tmp/areno_serve.pid
for i in $(seq 1 90); do
  sleep 5
  curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1 && break
done
curl -fsS http://127.0.0.1:8000/health
```

> `areno serve` 的启动日志在 `/tmp/areno_serve.log`。`--attn-backend native` 与训练侧同理（flash-attn-npu 的 kvcache 路径与 NPU graph capture 不兼容）；`--max-running-prompts 2` 控制 KV cache 池规模——默认 16 并发 × 模型 32k 上下文的预留约 60 GB，会把单卡 HBM 占满导致请求期 OOM，冒烟场景 2 个并发槽已足够。

查看服务暴露的模型列表（单条目，`owned_by` 固定为 `areno`）：

```shell #test id="serve-models"
curl -fsS http://127.0.0.1:8000/v1/models | python -c "
import sys, json
models = json.load(sys.stdin)['data']
assert models, 'no models in listing'
print('models_count:', len(models))
print('owned_by:', models[0]['owned_by'])
"
```

输出结果如下：

```shell #test-result id="serve-models"
models_count: 1
owned_by: areno
```

捕获服务端模型 ID，然后发一条 chat 请求验证端到端生成：

```shell #test-setup store="areno_model_id"
curl -fsS http://127.0.0.1:8000/v1/models | python -c "import sys, json; print(json.load(sys.stdin)['data'][0]['id'])"
```

```shell #test id="serve-chat" load="areno_model_id>>model"
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "<model>", "messages": [{"role": "user", "content": "用一句话介绍你自己"}], "max_tokens": 64}' \
  | python -c "
import sys, json
resp = json.load(sys.stdin)
choice = resp['choices'][0]
content = choice['message']['content']
assert content, 'empty completion'
print('finish_reason:', choice.get('finish_reason'))
print('content_len:', len(content))
"
```

输出结果类似如下：

```shell #test-result id="serve-chat" fuzzy='xxx'
finish_reason: xxx
content_len: xxx
```

- `xxx` 表示本次实际生成的结束原因与内容长度；生成内容本身具有随机性，只断言非空。

验证结束后关闭服务：

```shell #test-setup
kill "$(cat /tmp/areno_serve.pid)" 2>/dev/null || true
```

## 外部链接

- GitHub：[inclusionAI/AReno](https://github.com/inclusionAI/AReno)
- 文档中心：[AReno Docs](https://asystem-ai.io/docs/areno/)
- NPU 后端说明：[docs/concepts/npu.rst](https://github.com/inclusionAI/AReno/blob/main/docs/concepts/npu.rst)
