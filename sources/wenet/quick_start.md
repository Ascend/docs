# wenet

[WeNet](https://github.com/wenet-e2e/wenet) 是生产级的端到端语音识别工具包，支持流式与非流式识别；昇腾侧通过
`torch-npu` 把计算调度到 NPU。

本文以 [aishell-1](https://www.openslr.org/33) 数据集为例，按 WeNet 官方
[NPU 实验脚本](https://github.com/wenet-e2e/wenet/blob/main/examples/aishell/s0/run_npu.sh)
（[官方教程](https://wenet.org.cn/wenet/tutorial_aishell.html)），在单卡昇腾 NPU 上跑通
**数据下载 → 数据准备 → 训练 → 推理 → 导出** 全流程。

阅读本文前，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)
准备好 CANN 与驱动。

```{admonition} Note
:class: note
WeNet 最新的正式 release（`v3.1.0`，2024-05 发布）**不含** NPU 训练/推理脚本
`examples/aishell/s0/run_npu.sh`，只有 `main` 分支提供该脚本。因此本文以 `main` 分支为准，文中的
`<ref>` 取 `main`。
```

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**）。本文示例为**单卡**。

### 软件

与官方 NPU 实验脚本（`examples/aishell/s0`）的版本要求保持一致：

| 类别 | 最低版本 | 推荐版本 |
| --- | --- | --- |
| CANN | 8.0.RC2.alpha003 | latest |
| Python | 3.10 | 3.10 |
| torch | 2.1.0 | 2.2.0 |
| torch-npu | 2.1.0 | 2.2.0 |
| torchaudio | 2.1.0 | 2.2.0 |
| deepspeed | 0.13.2 | 0.14.4 |

此外需要提前就绪的编译工具 `git` 与音频工具 `sox`。

```{admonition} Note
:class: note
CANN 最低版本为 8.0.rc1，安装 CANN 时请同时安装 Kernel 算子包。
```

### 本文验证环境

| 组件 | 版本 |
| --- | --- |
| NPU | Ascend 910B4 × 1 |
| CANN | 8.0.0 |
| Python | 3.10 |
| torch / torch-npu / torchaudio | 2.2.0 |
| deepspeed | 0.14.4 |
| WeNet 源码 | `main`（仓库 `wenet-e2e/wenet`） |
| 数据集 | aishell-1（Hugging Face `JazerJu/aishell1-full-slr33`，OpenSLR 33 原档） |

本文配套镜像为 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.0.0-910b-ubuntu22.04-py3.10`。

`deepspeed` 最低要求 0.13.2，但 0.16+ 在 torch 2.2.0 下会触发 `torch.library.custom_op`
`AttributeError`，本文显式钉在 **0.14.4**。

## 1. 加载 CANN 环境

新开终端后 CANN 变量不会自动生效。`npu-smi` 常见位于 `/usr/local/sbin`，需要把该目录加入 `PATH`。

```shell #test-setup
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
test -n "$ASCEND_HOME_PATH"
command -v npu-smi
```

`ASCEND_HOME_PATH` 应非空，`npu-smi` 应能找到。

## 2. 检查环境是否就绪

检查 Python 版本：

```shell #test id="check-python"
python --version
```

输出结果如下：

```shell #test-result id="check-python" fuzzy='xxx'
Python 3.10.xxx
```

检查 `torch` / `torch-npu` 是否装好且 NPU 设备可用（下面的命令用 Python 执行）：

```python #test id="check-npu"
import torch
import torch_npu

print('torch:', torch.__version__)
print('torch_npu:', torch_npu.__version__)
print('npu available:', torch.npu.is_available())
print('npu count:', torch.npu.device_count())
```

输出结果如下：

```shell #test-result id="check-npu"
torch: 2.2.0
torch_npu: 2.2.0.post2
npu available: True
npu count: 1
```

```{admonition} Note
:class: note
如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)
检查 torch / torch-npu / CANN 三方兼容矩阵。
```

## 3. 获取 WeNet 源码

将 `<ref>` 替换为 `main`：

<!--
```shell #test-setup store="upstream_ref"
printf '%s\n' "$UPSTREAM_REF"
```
-->

```shell #test-setup id="clone" load="upstream_ref>>ref"
git clone https://github.com/wenet-e2e/wenet.git
cd wenet
git checkout <ref>
test -f examples/aishell/s0/run_npu.sh && echo "run_npu.sh ok"
```

`main` 分支上 `examples/aishell/s0/run_npu.sh` 存在。

## 4. 安装 WeNet 与依赖

安装 WeNet 及其 NPU 依赖。`[torch-npu]` extra 已将 `torch` / `torch-npu` / `torchaudio` 钉在推荐版本
2.2.0，并附带 `numpy<2`；`requirements.txt` 只约束 `deepspeed>=0.14.0`，为避免 pip 解析到与 torch 2.2.0
不兼容的 0.16+，安装后显式回钉 0.14.4。

```shell #test-setup id="install"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd wenet
pip install -e .[torch-npu]
pip install -r requirements.txt
pip install "deepspeed==0.14.4"
# train_utils.py 无条件 import deepspeed 与 tensorboardX；requirements.txt 是开发
# 全量清单（含 flake8/clang-format/openai-whisper 等与训练无关项），aarch64 上任一包
# 解析失败会导致整张清单未安装，且失败被命令块尾部的 rc 静默吞掉，
# 故训练核心依赖在安装步骤显式补齐
pip install tensorboardX
# 依赖校验：train_utils.py 无条件 import deepspeed，缺失要到训练阶段才暴露，
# 且训练块的 rc 会被 run_npu.sh 末尾命令掩盖，故在此显式校验（装不上立即失败）
python -c "import deepspeed, tensorboardX; print('deepspeed', deepspeed.__version__)"
```

安装 `sox`（`sox` 命令与 `libsox-dev` 头文件）：

```shell #test-setup
pip install sox
apt-get update && apt-get install -y sox libsox-dev || true
```

## 5. 下载数据

本步骤用官方脚本下载数据：`run_npu.sh` 调用 `local/download_and_untar.sh`，它按脚本内硬编码的字节数
校验已有归档、必要时用 `wget` 拉取，解压归档后再逐个解开 `data_aishell/wav/` 里按说话人二次打包的
`*.tar.gz`，最后写 `.complete` 标记。

下载源由官方脚本的 `--data_url` 指定，默认是 OpenSLR 原站 `www.openslr.org/resources/33`。下面把它指向
[OpenSLR 33](https://www.openslr.org/33) 两个原档的 [Hugging Face 镜像](https://huggingface.co/datasets/JazerJu/aishell1-full-slr33)
以加快下载；也可以换成 OpenSLR 自己的镜像：

| `--data_url` | 数据源 |
| --- | --- |
| `www.openslr.org/resources/33` | 官方 OpenSLR 原站（不传 `--data_url` 时的默认值） |
| `openslr.magicdatatech.com/resources/33` | OpenSLR 国内镜像 |
| `https://huggingface.co/datasets/JazerJu/aishell1-full-slr33/resolve/main` | 本文使用（Hugging Face 镜像） |

数据统一落在 `/root/asr-data/OpenSLR/33`，与官方教程的路径一致，后续步骤直接沿用。

```shell #test-setup id="download-data"
mkdir -p /root/.cache/asr-data/OpenSLR/33
# 数据与 .complete 标记放在缓存目录，容器重启后仍可复用；/root/asr-data 软链过去，
# 后续步骤因此仍用官方教程的 /root/asr-data/OpenSLR/33 路径。首次运行时该路径还不存在；
# 若已存在但不是软链接（容器内的旧目录），清掉重建，避免重复下载
if [ ! -L /root/asr-data ]; then
  rm -rf /root/asr-data
  ln -s /root/.cache/asr-data /root/asr-data
fi
cd wenet/examples/aishell/s0
# 官方下载脚本：local/download_and_untar.sh（wget）拉取 data_aishell / resource_aishell，
# 按脚本内硬编码字节数校验归档，解压后解开 wav/ 下的内层包，最后写 .complete
# 首次运行需完整下载并解压，耗时较长；命中 .complete 时该步骤直接返回
# --data_url：OpenSLR 33 的 Hugging Face 镜像；去掉该参数即回落到官方 www.openslr.org
bash run_npu.sh --stage -1 --stop_stage -1 \
  --data /root/asr-data/OpenSLR/33 \
  --data_url https://huggingface.co/datasets/JazerJu/aishell1-full-slr33/resolve/main
# run_npu.sh 没有 set -e，退出码只反映最后一条命令，故在此显式确认两个包都已就绪
test -f /root/asr-data/OpenSLR/33/data_aishell/.complete
test -f /root/asr-data/OpenSLR/33/resource_aishell/.complete
echo "=== /root/asr-data/OpenSLR/33 ==="
ls -la /root/asr-data/OpenSLR/33/
```

验证 `data_aishell` 与 `resource_aishell` 两个数据包均已就绪：

```shell #test id="verify-download"
echo "=== 检查下载标记文件 ==="
ls /root/asr-data/OpenSLR/33/data_aishell/.complete /root/asr-data/OpenSLR/33/resource_aishell/.complete
echo "=== 检查数据目录 ==="
ls /root/asr-data/OpenSLR/33/data_aishell/ | head -2
ls /root/asr-data/OpenSLR/33/resource_aishell/ | head -2
```

输出结果如下：

```shell #test-result id="verify-download"
=== 检查下载标记文件 ===
/root/asr-data/OpenSLR/33/data_aishell/.complete
/root/asr-data/OpenSLR/33/resource_aishell/.complete
=== 检查数据目录 ===
transcript
wav
lexicon.txt
speaker.info
```

## 6. 准备训练数据

本步骤由 `local/aishell_data_prep.sh` 把数据重新组织为 `wav.scp` 与 `text`：

```shell #test-setup id="prep-data"
cd wenet/examples/aishell/s0
bash run_npu.sh --stage 0 --stop_stage 0 --data /root/asr-data/OpenSLR/33
```

验证训练 / 验证 / 测试集划分，并确认每条转写都有对应的音频：

```shell #test id="verify-prep"
set -euo pipefail
cd wenet/examples/aishell/s0
for x in train dev test; do
  test -s data/$x/wav.scp
  test -s data/$x/text
  test "$(wc -l < data/$x/wav.scp)" = "$(wc -l < data/$x/text)"
done
echo "train/dev/test splits ok"
```

输出结果如下：

```shell #test-result id="verify-prep"
train/dev/test splits ok
```

## 7. 生成 CMVN、词典与 data.list

本步骤的脚本依次完成三件事：去除转写文本中的词间空格（普通话字符建模标准做法）并计算全局 CMVN 统计；
从训练转写生成字符级词典 `data/dict/lang_char.txt`；把 `wav.scp` / `text` 组织为训练可读的 `data.list`
（raw 格式，每行一条 utterance 的 JSON）。三者都是训练的前置产物，缺少时训练启动即报
`FileNotFoundError: data/dict/lang_char.txt`：

```shell #test-setup id="prep-cmvn-dict"
cd wenet/examples/aishell/s0
bash run_npu.sh --stage 1 --stop_stage 3 --data /root/asr-data/OpenSLR/33
```

验证三个前置产物，并确认 `data.list` 行数与对应的 `wav.scp` 一致：

```shell #test id="verify-cmvn-dict"
set -euo pipefail
cd wenet/examples/aishell/s0
test -f data/train/global_cmvn && echo "global_cmvn ok"
test -f data/dict/lang_char.txt && echo "lang_char.txt ok"
for x in train dev test; do
  test "$(wc -l < data/$x/data.list)" = "$(wc -l < data/$x/wav.scp)"
done
echo "data.list matches wav.scp"
```

输出结果如下：

```shell #test-result id="verify-cmvn-dict"
global_cmvn ok
lang_char.txt ok
data.list matches wav.scp
```

## 8. 模型训练

`run_npu.sh` 中实现了 NPU 卡号的自动获取与相关环境变量设置，可直接启动昇腾 NPU 上的模型训练。为控制时长，
将 `max_epoch` 从 240 缩短到 1（其余参数全部保持脚本默认值）：

```shell #test-setup id="train"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd wenet/examples/aishell/s0
cp conf/train_conformer.yaml conf/train_conformer_1ep.yaml
sed -i 's/max_epoch: .*/max_epoch: 1/' conf/train_conformer_1ep.yaml
# NPU fork-safety 适配：主进程初始化 CANN 后 fork 出的 DataLoader worker 会段
# 错误（torch_npu 2.2 + CANN 8.0 已知问题），必须单进程读取（num_workers=0）。
# torch 2.2 还要求 num_workers=0 时 persistent_workers=False 且
# prefetch_factor=None，否则 DataLoader 构造直接 ValueError，一并条件化
sed -i 's/^num_workers=.*/num_workers=0/' run_npu.sh
sed -i 's/persistent_workers=True/persistent_workers=args.num_workers > 0/g' "$(git rev-parse --show-toplevel)/wenet/utils/train_utils.py"
sed -i 's/prefetch_factor=args.prefetch/prefetch_factor=args.prefetch if args.num_workers > 0 else None/g' "$(git rev-parse --show-toplevel)/wenet/utils/train_utils.py"
# 源码改写校验：sed 路径或匹配失败时立即报错，避免被命令块尾部的 rc=0 吞掉
grep -q "persistent_workers=args.num_workers > 0" "$(git rev-parse --show-toplevel)/wenet/utils/train_utils.py"
grep -q "if args.num_workers > 0 else None" "$(git rev-parse --show-toplevel)/wenet/utils/train_utils.py"
bash run_npu.sh --stage 4 --stop_stage 4 --train_config conf/train_conformer_1ep.yaml --data /root/asr-data/OpenSLR/33
```

训练完成后检查输出：

```shell #test id="verify-train"
cd wenet/examples/aishell/s0
ls -la exp/conformer/train.yaml
ls exp/conformer/epoch_*.pt
```

输出结果如下：

```shell #test-result id="verify-train"
... exp/conformer/train.yaml
exp/conformer/epoch_0.pt
```

## 9. 测试推理

本步骤把测试集中的语音文件识别为文本。此外，当 `average_checkpoint` 为 `true`（脚本默认值）时，本步骤
会把交叉验证集上最佳的 `average_num` 个模型平均，生成增强模型 `avg_1.pt`，供解码与导出使用：

```shell #test-setup id="infer"
# CANN 环境必须显式加载：否则 LD_LIBRARY_PATH 缺少 libascendcl 等库路径，
# recognize.py 里 import torch_npu 会静默失败（ImportError 被
# is_torch_npu_available 吞掉只打印提示），随后 torch.device('npu') 直接崩溃
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd wenet/examples/aishell/s0
bash run_npu.sh --stage 5 --stop_stage 5 --average_num 1 --data /root/asr-data/OpenSLR/33
```

验证推理结果：`avg_1.pt` 已生成，识别文本覆盖全部测试集。

```shell #test id="verify-infer"
set -euo pipefail
cd wenet/examples/aishell/s0
test -f exp/conformer/avg_1.pt && echo "avg_1.pt ok"
test "$(wc -l < exp/conformer/ctc_greedy_search/text)" = "$(wc -l < data/test/wav.scp)"
echo "recognized all test utterances"
head -2 exp/conformer/ctc_greedy_search/text
```

输出结果如下（识别文本随模型收敛情况变化）：

```shell #test-result id="verify-infer"
avg_1.pt ok
recognized all test utterances
...
```

## 10. 导出训练好的模型

`wenet/bin/export_jit.py` 使用 `Libtorch` 导出训练好的模型（基于上一步生成的 `avg_1.pt`），导出的模型可用于
其他编程语言（如 C++）的推理：

```shell #test-setup id="export"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd wenet/examples/aishell/s0
bash run_npu.sh --stage 6 --stop_stage 6 --average_num 1
```

验证导出产物：

```shell #test id="verify-export"
cd wenet/examples/aishell/s0
test -f exp/conformer/final.zip && echo "final.zip ok"
test -f exp/conformer/final_quant.zip && echo "final_quant.zip ok"
```

输出结果如下：

```shell #test-result id="verify-export"
final.zip ok
final_quant.zip ok
```

## 11. 验证完整流程

确认全流程的关键文件均已生成：

```shell #test id="verify-all"
cd wenet/examples/aishell/s0
echo "=== 数据文件 ==="
ls data/dict/lang_char.txt data/train/data.list data/dev/data.list data/test/data.list | sort
echo "=== 训练输出 ==="
ls exp/conformer/train.yaml exp/conformer/final.pt | sort
echo "=== 推理输出 ==="
ls exp/conformer/avg_1.pt exp/conformer/ctc_greedy_search/text exp/conformer/ctc_prefix_beam_search/text | sort
echo "=== 导出输出 ==="
ls exp/conformer/final.zip exp/conformer/final_quant.zip | sort
echo "=== 流程完成 ==="
```

输出结果如下：

```shell #test-result id="verify-all"
=== 数据文件 ===
data/dev/data.list
data/dict/lang_char.txt
data/test/data.list
data/train/data.list
=== 训练输出 ===
exp/conformer/final.pt
exp/conformer/train.yaml
=== 推理输出 ===
exp/conformer/avg_1.pt
exp/conformer/ctc_greedy_search/text
exp/conformer/ctc_prefix_beam_search/text
=== 导出输出 ===
exp/conformer/final.zip
exp/conformer/final_quant.zip
=== 流程完成 ===
```

## 相关链接

- 上游仓库：[wenet-e2e/wenet](https://github.com/wenet-e2e/wenet)
- 官方教程：[AISHELL-1 示例](https://wenet.org.cn/wenet/tutorial_aishell.html)
- NPU 实验脚本：[`examples/aishell/s0/run_npu.sh`](https://github.com/wenet-e2e/wenet/blob/main/examples/aishell/s0/run_npu.sh)
- 数据集：[aishell-1（OpenSLR 33）](https://www.openslr.org/33)；数据下载脚本：[`local/download_and_untar.sh`](https://github.com/wenet-e2e/wenet/blob/main/examples/aishell/s0/local/download_and_untar.sh)，本文把它的 `--data_url` 指向 [Hugging Face 镜像](https://huggingface.co/datasets/JazerJu/aishell1-full-slr33)
- 昇腾 PyTorch 安装：[Ascend/pytorch](https://gitcode.com/Ascend/pytorch)
- 本项目昇腾环境与安装步骤：[安装指南](https://ascend.github.io/docs/sources/wenet/install.html)
