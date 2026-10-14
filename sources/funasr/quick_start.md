## 快速开始

在单卡昇腾 NPU 上跑通 [FunASR](https://github.com/modelscope/FunASR) 的核心能力：

- **语音识别生产级 pipeline**（`asr-pipeline`）：`paraformer-zh` + FSMN-VAD + 标点恢复 + 说话人嵌入，一次 `AutoModel` 同时产出转写文本、时间戳与匿名说话人编号；
- **多语识别 + 情感 + 事件**（`sensevoice`）：`SenseVoiceSmall` 一个模型同时输出语种、情感标签与事件标签；
- **流式识别**（`streaming`）：`paraformer-zh-streaming` 逐块喂音频，增量产出文本；
- **情感识别**（`emotion`）：`emotion2vec_plus_large` 独立情感分类；
- **命令行工具**（`cli`）：`funasr` 一键转写、输出 JSON / 字幕。

### 前置条件

#### 硬件

Atlas 900 A2 / A3 训练系列产品（Ascend 910B4 / 910B 等），并按需完成物理机或容器内的设备挂载。

#### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 与上面 CANN 匹配的 `torch` + `torch_npu` + `torchaudio`，且 `torch` 能正常 `import` 并 `torch.npu.is_available() == True`（参考 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)，按 torch ↔ torch_npu ↔ CANN 三方兼容矩阵选择版本）

#### 本文档示例使用的版本

**配套机器**：

- **机器类型**：Atlas 900 A2 PODc（Ascend 910B4，64 GB × 1）
- **操作系统**：Ubuntu 22.04

**配套镜像**：

swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12

**软件版本**：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| torch | 2.9.0+cpu |
| torch_npu | 2.9.0.post2 |
| torchaudio | 2.9.0 |
| funasr | 最新 release 版本 |
| 模型 | [paraformer-zh](https://github.com/modelscope/FunASR) / [SenseVoiceSmall](https://github.com/FunAudioLLM/SenseVoice) / [emotion2vec_plus_large](https://github.com/modelscope/FunASR) 等，默认从 ModelScope 下载 |

### 前置安装

确认能看到 NPU 设备：

```shell
npu-smi info
```

输出如下类似结果：

```
+------------------------------------------------------------------------------------------------+
| npu-smi 25.5.2                   Version: 25.5.2                                               |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B4               | OK            | 89.9        39                0    / 0             |
| 0                         | 0000:41:00.0  | 0           0    / 0          2922 / 32768         |
+===========================+===============+====================================================+
```

检查 Python 版本：

```shell #test id="check-py"
python --version
```

输出结果如下：

```shell #test-result id="check-py" 
Python 3.12...
```

安装 `torch` / `torch_npu` / `torchaudio`：

```shell #test-setup
uv pip install -f https://mirrors.aliyun.com/pytorch-wheels/cpu torch==2.9.0
uv pip install -f https://mirrors.aliyun.com/pytorch-wheels/cpu torchaudio==2.9.0
uv pip install --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi torch_npu==2.9.0.post2
```

检查 torch / torch_npu / torchaudio 是否装好且 NPU 设备可用（下面代码用 python 运行）：

```python #test id="check-torch"
import torch, torch_npu, torchaudio
print('torch=', torch.__version__)
print('torch_npu=', torch_npu.__version__)
print('torchaudio=', torchaudio.__version__)
print('is_available:', torch.npu.is_available())
print('count:', torch.npu.device_count())
```

输出结果如下：

```shell #test-result id="check-torch" 
torch= 2.9.0+cpu
torch_npu= 2.9.0.post2
torchaudio= 2.9.0
is_available: True
count: 1
```

```{admonition} Note
:class: note
如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。
```

### 安装 FunASR

#### 从源码安装（可选）

<!-- 工作流注入的 UPSTREAM_REF（最新 release tag）通过这个隐藏的 #test-setup 捕获并注入到下方 install 命令中-->
<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

克隆上游仓库并 checkout 到最新 release tag，安装并且验证。

```shell #test-setup load="upstream_ref>>ref"
git clone --depth 1 --branch <ref> https://github.com/modelscope/FunASR.git
cd FunASR
uv pip install -e .
```

```{admonition} Note
:class: note
`<ref>` 为安装的最新的 release tag。
```

<!--
```shell #test-setup
uv pip uninstall funasr -y
```
-->

#### 使用 pip 进行安装

```shell #test id="funasr-install-binary"
uv pip install funasr
python -c "import funasr; print('funasr', funasr.__version__)"
```

输出结果类似如下：

```shell #test-result id="funasr-install-binary" 
Notice: ffmpeg is not installed. torchaudio is used to load audio
...
funasr ...
```

```{admonition} Note
:class: note
`funasr ...` 表示最新的版本号。
```

#### CLI 自检：`funasr --version`

`funasr` 是 FunASR 自带的命令行转写工具，装好后就能用：

```shell #test id="funasr-version"
funasr --version
```

输出结果类似如下：

```shell #test-result id="funasr-version" 
Notice: ffmpeg is not installed. torchaudio is used to load audio
...
funasr ...
```

```{admonition} Note
:class: note
`funasr ...` 是最新的release的版本号。
```

### 语音识别生产级 pipeline（ASR + VAD + 标点 + 说话人）

这是 FunASR 最典型的生产装配：一个 `AutoModel` 同时挂上 `vad_model`（语音活性检测）/ `punc_model`（标点恢复）/ `spk_model`（说话人嵌入），一次 `generate` 产出转写文本、字级时间戳和匿名说话人编号。模型默认从 ModelScope 拉取，`device="npu:0"` 让 ASR / VAD / 标点 / 说话人四个子模型都放到 NPU 上跑（下面代码用 python 运行）：

```python #test id="funasr-asr-pipeline"
import torch, torch_npu
from funasr import AutoModel

model = AutoModel(
    model="paraformer-zh",
    vad_model="fsmn-vad",
    punc_model="ct-punc",
    spk_model="cam++",
    device="npu:0",
    disable_update=True,
)
result = model.generate(
    input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav",
)
r = result[0]
print('device:', next(model.model.parameters()).device)
print('text:', r['text'])
print('sentence_info_keys:', sorted(r['sentence_info'][0].keys()))
```

输出结果类似如下：

```shell #test-result id="funasr-asr-pipeline" 
...
device: npu:0
text: 欢迎大家来体验达摩院推出的语音识别模型。
sentence_info_keys: ['end', 'spk', 'start', 'text', 'timestamp']
```

```{admonition} Note
:class: note
- `device: npu:0` 说明 ASR 模型权重确实落在昇腾设备。
- `sentence_info` 由 VAD 分段 + 标点恢复 + 说话人聚类共同产出，每个分段含 `text` / `start` / `end`（时间戳）与 `spk`（录音内匿名说话人编号）。
```

### 多语识别 + 情感 + 事件（SenseVoiceSmall）

`SenseVoiceSmall` 一个模型同时输出**语种标签**（`<|zh|>` 等）、**情感标签**（`<|NEUTRAL|>` / `<|HAPPY|>` 等）与**事件标签**（`<|Speech|>` / `<|Applause|>` 等），原始 `text` 带上这些标签，用 `rich_transcription_postprocess` 可剥离得到干净文本（下面代码用 python 运行）：

```python #test id="funasr-sensevoice"
import torch, torch_npu
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model = AutoModel(
    model="iic/SenseVoiceSmall",
    device="npu:0",
    disable_update=True,
)
result = model.generate(
    input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav",
)
r = result[0]
print('device:', next(model.model.parameters()).device)
print('raw_text:', r['text'])
print('clean_text:', rich_transcription_postprocess(r['text']))
```

输出结果类似如下：

```shell #test-result id="funasr-sensevoice" 
...
device: npu:0
raw_text: ...
clean_text: 欢迎大家来体验达摩院推出的语音识别模型
```

```{admonition} Note
:class: note
- `raw_text` 一栏能看到 `|zh|`（语种）、`|NEUTRAL|`（情感）、`|Speech|`（事件）三类标签拼接在转写文本两侧——这是 SenseVoice 的多任务输出合同。
- `clean_text` 是剥离标签后的纯文本。
```

### 流式识别（paraformer-zh-streaming）

流式场景按固定时长（`chunk_size`）逐块喂音频，`cache` 在会话内维护上下文状态，每块增量返回文本。先下载官方示例音频到本地：

```shell #test-setup
curl -sL -o /tmp/funasr_sample.wav https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav
```

把整段音频切成 600ms 一块依次喂给流式模型（下面代码用 python 运行）：

```python #test id="funasr-streaming"
import torch, torch_npu
import soundfile as sf
from funasr import AutoModel

model = AutoModel(
    model="paraformer-zh-streaming",
    device="npu:0",
    disable_update=True,
)
audio, sr = sf.read('/tmp/funasr_sample.wav', dtype='float32')
chunk_size = [0, 10, 5]          # 每块 600ms
chunk_stride = chunk_size[1] * 960
cache = {}
parts = []
n_chunks = (len(audio) - 1) // chunk_stride + 1
for i in range(n_chunks):
    chunk = audio[i * chunk_stride:(i + 1) * chunk_stride]
    res = model.generate(
        input=chunk, cache=cache, is_final=(i == n_chunks - 1),
        chunk_size=chunk_size, encoder_chunk_look_back=4, decoder_chunk_look_back=1,
    )
    if res[0].get('text'):
        parts.append(res[0]['text'])
print('device:', next(model.model.parameters()).device)
print('streamed_text:', ''.join(parts))
```

输出结果类似如下：

```shell #test-result id="funasr-streaming" 
...
device: npu:0
streamed_text: 欢迎大家来体验达摩院推出的语音识别模型
```

`streamed_text` 是所有块增量返回文本拼接后的完整转写；生产里把每块非空 `text` 直接推给下游即可实现实时上屏。

### 情感识别（emotion2vec_plus_large）

独立的情感识别模型，`granularity="utterance"` 返回整句的情感标签分布（下面代码用 python 运行）：

```python #test id="funasr-emotion"
import torch, torch_npu
from funasr import AutoModel

model = AutoModel(
    model="iic/emotion2vec_plus_large",
    device="npu:0",
    disable_update=True,
)
result = model.generate(
    input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav",
    granularity="utterance",
)
r = result[0]
labels, scores = r['labels'], r['scores']
top = max(zip(labels, scores), key=lambda kv: kv[1])
print('device:', next(model.model.parameters()).device)
print('n_labels:', len(labels))
print('top_emotion:', top[0])
```

输出结果类似如下：

```shell #test-result id="funasr-emotion" 
...
device: npu:0
n_labels: 9
top_emotion: 中立/neutral
```

`labels` 覆盖 8 种情感（生气/厌恶/恐惧/开心/中立/其他/难过/吃惊）+ 1 个 `other`/`<unk>` 兜底标签，`top_emotion` 是概率最高的那一个。

### 命令行工具

`funasr` 一条命令即可转写（默认 `sensevoice` 模型），支持 `-f json` 结构化输出与字幕生成。下面把之前下载的音频转成 JSON，并从中抽取关键字段做校验（`sed` 从首个 `{` 开始截取，把 FunASR 的 Notice/版本提示行挡在 JSON 之外）：

```shell #test id="funasr-cli"
funasr /tmp/funasr_sample.wav --device npu:0 -f json 2>/dev/null | sed -n '/^{/,$p' | python3 -c "import sys, json; d = json.load(sys.stdin); print('model:', d['model']); print('has_text:', bool(d.get('text')))"
```

输出结果如下：

```shell #test-result id="funasr-cli"
model: sensevoice
has_text: True
```

其他常用 CLI 形态：`funasr audio.wav -f srt -o ./subs` 生成字幕、`funasr audio.wav --spk --timestamps` 带说话人与时间戳、`funasr audio.wav --model paraformer --language zh` 指定模型与语言。
