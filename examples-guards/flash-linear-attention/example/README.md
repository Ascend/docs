# 在昇腾上训练一个小型语言模型并续写文本

你将使用一份本地文本训练一个小型语言模型，保存模型与词表，然后加载保存的模型，让它续写你输入的句子。模型的线性注意力由 FLA 的 GatedDeltaNet 提供。

示例自带一份小语料 `corpus.txt`，也支持你自己的 UTF-8 文本。模型从随机权重开始，不下载预训练权重、tokenizer 或外部数据集。这里的字符词表仅用于说明完整流程，不是生产环境的 tokenizer。

## 运行前准备

需要一张可用的 Ascend 910B，以及已经安装好驱动和 CANN 的 Linux 环境。如果使用容器，容器内必须能够访问 NPU 设备；安装 Python 包不会替你完成驱动和设备配置。

以下命令以 FLA v0.5.2 为安装基线。底层配套来自该版本的[依赖声明](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/pyproject.toml)和 [Triton-Ascend 官方镜像配套](https://github.com/triton-lang/triton-ascend/blob/main/docker/OVERVIEW.md#release-321)。

| 组件 | 本文使用的配套 |
| --- | --- |
| 硬件 | Ascend 910B，单卡 |
| 操作系统 | Ubuntu 22.04，aarch64 |
| Python | 3.11 |
| CANN Toolkit | 9.0.0；宿主机驱动须与其兼容 |
| FLA | v0.5.2 |
| PyTorch | 2.7.1 |
| Torch-NPU | 2.7.1.post4 |
| torchvision | 0.22.1，由 FLA 的 NPU extra 安装 |
| Triton-Ascend | 3.2.1 |
| Transformers | 使用 4.x，实测版本为 4.57.6；安装时将 FLA 的 `>=4.45.0` 要求与本示例的 `<5` 兼容约束共同解析 |

默认示例已在上述配套和 Transformers 4.57.6 上[完成 NPU 端到端验证](https://github.com/cosdt-ci-test/workflows/actions/runs/36403948431)：20 步训练、模型保存、重载一致性检查和 32-token 生成均通过。该结果限于本示例配置；换用其他语料、模型尺寸或软件版本时，需要重新验证。FLA v0.5.2 与 Transformers 5.17.0 的模型保存接口不兼容，因此本示例使用 4.x 约束。

先进入存放本示例的 workflows 仓库根目录，以下命令均从该目录执行。按默认安装路径加载 CANN，并确认设备可见：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info
python --version
```

然后在独立的 Python 环境中安装选定版本的 FLA：

```bash
git clone --branch v0.5.2 --depth 1 \
  https://github.com/fla-org/flash-linear-attention.git ./fla-v0.5.2
python -m venv .venv-fla
source .venv-fla/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install pybind11 cmake attrs sympy pyyaml scipy decorator einops
python -m pip install -e './fla-v0.5.2[npu]' \
  --constraint projects/flash-linear-attention/constraints-npu.txt \
  --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple
```

`[npu]` 会安装该版本声明的 Torch、Torch-NPU 和 Triton-Ascend，不需要手工分别挑选版本。换用其他 FLA release 时，请重新核对其依赖和 CANN 配套；不要直接把这些命令中的某个组件升级到最新版。

请保留上面的 `--constraint` 参数。它让 Transformers 在 4.x 范围内选择符合要求的版本，避免 FLA v0.5.2 的模型保存接口与 5.x 不兼容；并没有固定某个 4.x 补丁版本。

安装后检查 NPU 后端和实际包版本：

```bash
python - <<'PY'
from importlib.metadata import version
import torch
import torch_npu
from fla.utils import IS_NPU, device_platform

assert torch.npu.is_available(), "NPU 不可用，请先检查驱动、设备与 CANN 环境"
assert IS_NPU and device_platform == "npu", "FLA 没有识别到 NPU 后端"
for package in ("torch", "torch-npu", "triton-ascend", "transformers", "flash-linear-attention"):
    print(package, version(package))
print("FLA backend:", device_platform)
PY
```

这里确认的是安装和设备识别。训练与生成是否可用，需要继续运行下面的完整流程。CANN 安装可参考 [Quick Start 的环境要求](../docs/Quick-start-Ascend.md)。

## 运行示例

在 workflows 仓库根目录运行：

```bash
python projects/flash-linear-attention/example/train_text.py \
  --output-dir ./fla-text-run
```

默认模型为 2 层、hidden size 256、4 个 attention heads、head dimension 64，使用 BF16。默认 batch size 为 2，序列长度为 128，训练 20 步，然后生成 32 个新字符 token。

执行过程如下：

1. 读取文本，建立字符词表，并切成固定长度的训练片段。
2. 用 FLA 配置构建语言模型，在 NPU 上完成 next-token prediction 和 AdamW 参数更新。
3. 保存模型配置、权重和词表。
4. 从本地 checkpoint 重新加载模型，用 `generate(use_cache=True)` 续写文本。

示例使用 FLA 的完整语言模型接口，Transformers 提供模型创建、保存加载和生成调度；NPU 计算依赖 Torch-NPU、FLA 算子和 Triton-Ascend 与 CANN 的配套。模型训练与生成发生在 NPU，checkpoint 写盘时将权重移到 CPU。

## 使用自己的文本

```bash
python projects/flash-linear-attention/example/train_text.py \
  --text-file ./my-text.txt \
  --prompt 'The small model' \
  --steps 200 \
  --max-new-tokens 64 \
  --output-dir ./my-fla-model
```

文本至少需要包含一个完整训练片段。prompt 中的每个字符都需要出现在训练文本中；省略 `--prompt` 时使用训练文本开头。输出目录必须是新目录，重复运行请换一个目录，避免覆盖已有模型。

这是教学规模的从零训练：20 步可以演示流程，但不能保证生成流畅文本。日志中的初始/最终 loss 在同一组训练片段上计算，仅用于观察学习过程，不代表验证集指标或泛化能力。

## 查看结果

```text
fla-text-run/
├── checkpoint/
│   ├── config.json
│   ├── model.safetensors
│   ├── vocabulary.json
│   └── generation_config.json
├── generated.txt
└── metrics.json
```

`generated.txt` 包含 prompt 和续写文本。`metrics.json` 记录逐步 loss、训练前后 loss、模型是否更新、保存加载前后的预测是否一致，以及本次运行使用的依赖版本。

查看文本和运行指标：

```bash
cat ./fla-text-run/generated.txt
python -m json.tool ./fla-text-run/metrics.json
```

正常完成后，应看到模型文件、非空续写文本，以及有限的 loss 数值。生成文本不需要与某个固定句子一致；如 loss 出现 NaN/Inf 或保存加载检查失败，不应继续将该模型用于后续推理。

## 常见问题

- **`npu-smi` 看不到设备或 `torch.npu.is_available()` 为 False**：先检查宿主机驱动、容器设备访问权限和 CANN 环境，再运行示例。
- **FLA 没有识别到 NPU 后端**：检查当前 Python 环境是否使用目标版本的 `[npu]` 安装，核对 `triton-ascend` 版本；不要只根据 `import triton` 成功判断安装正确。
- **算子编译或执行报错**：保留完整报错和 `pip freeze`，核对 CANN、Torch-NPU、Triton-Ascend 配套。改用另一个上层框架并不能自动解决算子兼容问题。
- **prompt 包含词表外字符**：换成训练文本中出现过的字符，或把需要的字符加入语料后重新训练。
- **输出目录已存在**：选择新的 `--output-dir`，已有模型不会被自动覆盖。
- **续写重复或不通顺**：这是小模型短时训练的正常可能结果，可增加语料和训练步数；本示例不保证生成质量。

维护者的 CI 接入和校验规则见[看护维护文档](../docs/examples-maintenance.md)。

接口参考：[FLA Usage](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/README.md#usage)、[GatedDeltaNetConfig](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/models/gated_deltanet/configuration_gated_deltanet.py)、[GatedDeltaNetForCausalLM](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/models/gated_deltanet/modeling_gated_deltanet.py)。
