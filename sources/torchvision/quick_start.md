# torchvision

[Torchvision](https://github.com/pytorch/vision) 是 PyTorch 的计算机视觉库。本文在昇腾 NPU 上跑通它的最小链路：

- **安装**：源码构建 `torchvision`（`FORCE_CUDA=0`），配合 `torch` 2.14.0 + `torch_npu` 2.14.0。
- **transforms v2**：用合成图在 NPU 上验证 9 段核心用法（基础 / 随机裁剪 / 分类流水线 / 检测 / 多类型 TVTensor / `tv_tensors` 语义 / 嵌套结构透传 / `Dataset` 互操作）。

`torchvision` 通过 `torch_npu` 在 NPU 上跑通。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，并按需完成物理机或容器内的设备挂载（`/dev/davinci*` 等）。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 与上面 CANN 匹配的 `torch` + `torch_npu`（参考 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)，按 torch ↔ torch_npu ↔ CANN 三方兼容矩阵选择版本）

### 本文档示例使用的版本

**配套机器**：

- **机器类型**：Atlas 900 A2 PODc（Ascend 910B4，32 GB × 1）
- **操作系统**：Ubuntu 22.04

**配套镜像**：

swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12

**软件版本**：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| torch | 2.14.0+cpu |
| torch_npu | 2.14.0 |
| torchvision | 最新 release（**源码构建** `FORCE_CUDA=0`，见「安装 torchvision」） |
| pillow | `>=10.0`（`torchvision.transforms.functional.to_pil_image` 等的运行时依赖） |

## 前置安装

确认能看到 NPU 设备：

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
| 5     910B4               | OK            | 89.9        39                0    / 0             |
| 0                         | 0000:41:00.0  | 0           0    / 0          2922 / 32768         |
+===========================+===============+====================================================+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| No running processes found in NPU 5                                                            |
+===========================+===============+====================================================+
```

```{admonition} Note
:class: note
如果 `npu-smi` 不存在，请回到 [Ascend 官方快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 补装驱动。
```

检查 Python 版本：

```shell #test id="check-py"
python --version
```

输出结果如下：

```shell #test-result id="check-py" fuzzy='xxx'
Python 3.12.xxx
```

安装 `torch` / `torch_npu`：

```shell #test-setup
uv pip install -f https://mirrors.aliyun.com/pytorch-wheels/cpu torch==2.14.0
uv pip install torch_npu==2.14.0
```

检查 torch / torch_npu 是否装好且 NPU 设备可用(下面的命令用 Python 执行)：

```python #test id="check-torch"
import torch, torch_npu
print('torch=', torch.__version__)
print('torch_npu=', torch_npu.__version__)
print('is_available:', torch.npu.is_available())
print('count:', torch.npu.device_count())
```

输出结果如下：

```shell #test-result id="check-torch" fuzzy='xxx'
torch= 2.14.0+cpu
torch_npu= 2.14.0
is_available: True
count: 1
```

```{admonition} Note
:class: note
如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。
```

安装 `pillow`：

```shell #test-setup
uv pip install 'pillow>=10.0'
```

打印版本(下面的命令用 Python 执行)：

```python #test id="install-deps"
import PIL
print('Pillow', PIL.__version__)
```

输出结果如下：

```shell #test-result id="install-deps" fuzzy='xxx'
Pillow xxx
```

```{admonition} Note
:class: note
xxx 表示 Pillow 的版本号。
```

## 安装 torchvision

PyPI 上的 linux wheel 全是 CUDA 版（链 `libcudart.so`），跟 torch_npu 不兼容，所以只能**从源码构建**：`FORCE_CUDA=0` 让构建脚本跳过 CUDA 依赖。构建工具（g++ / make / cmake / git）基础镜像里已有，无需额外安装。

<!-- 工作流注入的 UPSTREAM_REF（最新 release tag）通过这个隐藏的 #test-setup 捕获并注入到下方 install 命令中-->
<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

克隆上游仓库、`FORCE_CUDA=0` 源码构建：

```shell #test-setup load="upstream_ref>>ref"
git clone --depth 1 --branch <ref> https://github.com/pytorch/vision.git
cd vision
FORCE_CUDA=0 uv pip install -e . --no-build-isolation
```

```{admonition} Note
:class: note
`<ref>` 替换为 pytorch/vision 当前的最新 release 版本。
```

验证装上的版本(下面的命令用 Python 执行)：

```python #test id="stock-torchvision-source"
import torchvision
print('torchvision', torchvision.__version__)
```

输出结果如下：

```shell #test-result id="stock-torchvision-source" fuzzy='xxx'
torchvision xxx
```

```{admonition} Note
:class: note
xxx 表示最新的版本号。
```

## transforms v2 入门

用一个**合成的 256×256 RGB 测试图**走一遍 transforms v2 的核心用法。每节都把图搬到 NPU 上运行（`img.to('npu:0')`），用 v2 的 `BoundingBoxes` / `Mask` / `Video` / `KeyPoints` 配合验证 dispatch 链路。

### 合成一张测试图并确认 NPU 可用

**确认环境就绪**：

- `torch_npu` 装好且能看到 NPU 设备
- `tv_tensors` 跟 `torchvision.io` 的链路通
- 把 numpy 数组装进 `tv_tensors.Image`——这是后面所有 transform 的入口数据类型

下面用 **python** 执行：

```python #test id="v2-setup"
import numpy as np
import torch
import torch_npu
import torchvision
from torchvision.transforms import v2
from torchvision.io import decode_image
from torchvision import tv_tensors

# 固定随机种子，后面每节的合成图都用同一颗种子，结果可复现
torch.manual_seed(1)
np.random.seed(1)

# 合成 256×256 RGB 图：CPU 路径、零网络依赖
# （官方教程用的是 github.com/pytorch/vision gallery 里的 astronaut.jpg，
#  那边要 git clone 整个仓库，本烟雾路径跳过）
arr = (np.random.rand(256, 256, 3) * 255).astype('uint8')
img = tv_tensors.Image(torch.from_numpy(arr).permute(2, 0, 1))
print(f"{type(img) = }, {img.dtype = }, {img.shape = }")
# torchvision.io.decode_image 走 libjpeg-turbo 入口（不是 Pillow），
# 这里只是验 import 通，不需要真去解码文件
print(f"{decode_image.__module__ = }")

# 验证 NPU：先看有没有设备，再做一次最简 cpu → npu → cpu 往返
print(f"npu_available: {torch.npu.is_available()}")
print(f"npu_count: {torch.npu.device_count()}")
x = torch.zeros(2, 3)
x_npu = x.to('npu:0')
x_back = x_npu.cpu()
print(f"npu round-trip: in={tuple(x.shape)} on {x.device.type}, npu={tuple(x_npu.shape)} on {x_npu.device.type}, back={tuple(x_back.shape)} on {x_back.device.type}")
```

输出结果如下：

```shell #test-result id="v2-setup" fuzzy='xxx'
type(img) = <class 'torchvision.tv_tensors._image.Image'>, img.dtype = torch.uint8, img.shape = torch.Size([3, 256, 256])
decode_image.__module__ = 'torchvision.io.image'
npu_available: True
npu_count: xxx
npu round-trip: in=(2, 3) on cpu, npu=(2, 3) on npu, back=(2, 3) on cpu
```

```{admonition} Note
:class: note
xxx 表示当前可见的 NPU 设备数量。
```

### 用 CenterCrop 跑通第一个 transform

v2 transform 用起来跟普通 `nn.Module` 一样：**实例化一次，可以反复调用**。跑最简单的 `CenterCrop`：

下面用 **python** 执行：

```python #test id="v2-basics"
import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

# 跟上一节一样的合成图
np.random.seed(0)
arr = (np.random.rand(256, 256, 3) * 255).astype('uint8')
img = tv_tensors.Image(torch.from_numpy(arr).permute(2, 0, 1))

# 把图搬到 NPU——transform 在哪个 device 上跑由输入 tensor 决定
img_npu = img.to('npu:0')
transform = v2.CenterCrop(size=(224, 224))
out = transform(img_npu)
print(f"in:  type={type(img_npu).__name__} device={img_npu.device.type} shape={tuple(img_npu.shape)}")
print(f"out: type={type(out).__name__} device={out.device.type} shape={tuple(out.shape)}")
```

输出结果如下：

```shell #test-result id="v2-basics"
in:  type=Image device=npu shape=(3, 256, 256)
out: type=Image device=npu shape=(3, 224, 224)
```

### 在 NPU 上裁剪出 224×224 的图

`RandomCrop` 默认 input 必须 ≥ output，否则要 pad。这里 256 ≥ 224，offset 恒为 `(0, 0)`——**这条路径是确定的**，只在 NPU 上跑一次验形状：

下面用 **python** 执行：

```python #test id="v2-randomcrop"
import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

np.random.seed(1)
arr = (np.random.rand(256, 256, 3) * 255).astype('uint8')
img = tv_tensors.Image(torch.from_numpy(arr).permute(2, 0, 1))

# NPU 端 crop
img_npu = img.to('npu:0')
out = v2.RandomCrop(size=(224, 224))(img_npu)
print(f"in:  type={type(img_npu).__name__} device={img_npu.device.type} shape={tuple(img_npu.shape)}")
print(f"out: type={type(out).__name__} device={out.device.type} shape={tuple(out.shape)}")
```

输出结果如下：

```shell #test-result id="v2-randomcrop"
in:  type=Image device=npu shape=(3, 256, 256)
out: type=Image device=npu shape=(3, 224, 224)
```

### 用 `Compose` 搭一条分类预处理流水线

单 transform 容易验；**多个串起来**才是真实训练场景。`Compose` 把一组 transform 按顺序作用在输入上。ImageNet 训练的经典 pipeline：

1. `RandomResizedCrop` — 随机切一块再 resize 到目标尺寸
2. `RandomHorizontalFlip(p=0.5)` — 一半概率水平翻转
3. `ToDtype(float32, scale=True)` — uint8 → float32，顺便除以 255
4. `Normalize(mean, std)` — 用 ImageNet 均值 / 方差归一化

下面用 **python** 执行：

```python #test id="v2-classification"
import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

np.random.seed(1)
arr = (np.random.rand(256, 256, 3) * 255).astype('uint8')
img = tv_tensors.Image(torch.from_numpy(arr).permute(2, 0, 1))

# 锁定 crop / flip 的随机性——这两步都走 torch 的 RNG
torch.manual_seed(1)

img_npu = img.to('npu:0')
transforms = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
out = transforms(img_npu)
print(f"in:  type={type(img_npu).__name__} device={img_npu.device.type} dtype={img_npu.dtype} shape={tuple(img_npu.shape)}")
print(f"out: type={type(out).__name__} device={out.device.type} dtype={out.dtype} shape={tuple(out.shape)}")
# 归一化后值大致落在 [-2.12, 2.64]——mean=0.485/std=0.229 推算：
#   (1 - 0.485) / 0.229 ≈ 2.25   (像素最大值的归一化结果)
#   (0 - 0.485) / 0.229 ≈ -2.12  (像素最小值的归一化结果)
print(f"out stats: min={float(out.min().cpu()):.4f} mean={float(out.mean().cpu()):.4f} max={float(out.max().cpu()):.4f}")
print(f"out finite: {torch.isfinite(out).all().item()}")  # 兜底检查：cast / div 路径没有 NaN/Inf
```

输出结果如下：

```shell #test-result id="v2-classification" fuzzy='xxx'
in:  type=Image device=npu dtype=torch.uint8 shape=(3, 256, 256)
out: type=Image device=npu dtype=torch.float32 shape=(3, 224, 224)
out stats: min=xxx mean=xxx max=xxx
out finite: True
```

### 让标注框跟着图像一起做变换

分类只处理图像；**检测任务**还要同时处理标注框（bounding box）。v2 用 `BoundingBoxes` TVTensor 表达标注框——形状 `(N, 4)`，带 `format`（坐标格式：`XYXY` / `CXCYWH` 等）跟 `canvas_size`（图像尺寸）两个 metadata：

下面用 **python** 执行：

```python #test id="v2-detection"
import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

np.random.seed(1)
arr = (np.random.rand(256, 256, 3) * 255).astype('uint8')
img = tv_tensors.Image(torch.from_numpy(arr).permute(2, 0, 1))
H, W = img.shape[-2:]
# 3 个检测框：XYXY = (x1, y1, x2, y2)
boxes = tv_tensors.BoundingBoxes(
    [[15, 10, 370, 510], [275, 340, 510, 510], [130, 345, 210, 425]],
    format="XYXY", canvas_size=(H, W))

torch.manual_seed(1)

img_npu = img.to('npu:0')
# 图像上 NPU；box 坐标留在 CPU——坐标是几个数字，搬到 NPU 没收益反而慢
transforms = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),   # crop 后 box 跟着重映射
    v2.RandomPhotometricDistort(p=1),                          # 颜色 jitter 链：亮度/对比度/饱和度/色相
    v2.RandomHorizontalFlip(p=1),                              # 永远翻，box 同步翻
])
out_img, out_boxes = transforms(img_npu, boxes)
fmt = lambda b: b.format.name if hasattr(b.format, 'name') else b.format
print(f"in_img: type={type(img_npu).__name__} device={img_npu.device.type} dtype={img_npu.dtype} shape={tuple(img_npu.shape)}")
print(f"in_boxes: type={type(boxes).__name__} format={fmt(boxes)} canvas_size={tuple(boxes.canvas_size)} shape={tuple(boxes.shape)}")
print(f"out_img: type={type(out_img).__name__} device={out_img.device.type} dtype={out_img.dtype} shape={tuple(out_img.shape)}")
print(f"out_boxes: type={type(out_boxes).__name__} format={fmt(out_boxes)} canvas_size={tuple(out_boxes.canvas_size)} shape={tuple(out_boxes.shape)}")
print(f"out_img finite: {torch.isfinite(out_img.float()).all().item()}")  # photometric 链路里某一步意外产 NaN/Inf？
print(f"out_boxes min/max: {int(out_boxes.min())}/{int(out_boxes.max())}")  # crop + flip 后坐标应仍在 [0, 224]
```

输出结果如下：

```shell #test-result id="v2-detection" fuzzy='xxx'
in_img: type=Image device=npu dtype=torch.uint8 shape=(3, 256, 256)
in_boxes: type=BoundingBoxes format=XYXY canvas_size=(256, 256) shape=(3, 4)
out_img: type=Image device=npu dtype=torch.uint8 shape=(3, 224, 224)
out_boxes: type=BoundingBoxes format=XYXY canvas_size=(224, 224) shape=(3, 4)
out_img finite: True
out_boxes min/max: xxx
```

### 一次 transform 同时处理 5 种 TVTensor

v2 不只能 transform image——一次调用可以塞 5 种 TVTensor（Image + BoundingBoxes + Mask + Video + KeyPoints），transform 内部按**类型**分别 dispatch：

下面用 **python** 执行：

```python #test id="v2-vbmk"
import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

np.random.seed(0)
arr = (np.random.rand(256, 256, 3) * 255).astype('uint8')
img = tv_tensors.Image(torch.from_numpy(arr).permute(2, 0, 1))
H, W = img.shape[-2:]

boxes = tv_tensors.BoundingBoxes(
    [[15, 10, 370, 510], [275, 340, 510, 510], [130, 345, 210, 425]],
    format="XYXY", canvas_size=(H, W))
mask = tv_tensors.Mask(torch.zeros((1, H, W), dtype=torch.uint8))           # 分割 mask
video = tv_tensors.Video(torch.randint(0, 256, (3, 3, 32, 32), dtype=torch.uint8))  # (T, C, H, W)
keypoints = tv_tensors.KeyPoints(
    torch.tensor([[100, 100], [200, 200], [50, 150]]),
    canvas_size=(H, W))                                                     # 关键点

# p=0 保证确定性：这次不验证"翻得对不对"，只验证"5 种类型都正确 dispatch 了"
img_npu = img.to('npu:0')
mask_npu = mask.to('npu:0')
video_npu = video.to('npu:0')
# box / keypoints 留 CPU——坐标无需 NPU 计算
out_img, out_boxes, out_mask, out_video, out_kp = v2.RandomHorizontalFlip(p=0)(
    img_npu, boxes, mask_npu, video_npu, keypoints)
print(f"img: type={type(out_img).__name__} device={out_img.device.type} shape={tuple(out_img.shape)}")
print(f"boxes: type={type(out_boxes).__name__} device={out_boxes.device.type} shape={tuple(out_boxes.shape)}")
print(f"mask: type={type(out_mask).__name__} device={out_mask.device.type} shape={tuple(out_mask.shape)}")
print(f"video: type={type(out_video).__name__} device={out_video.device.type} shape={tuple(out_video.shape)}")
print(f"keypoints: type={type(out_kp).__name__} device={out_kp.device.type} shape={tuple(out_kp.shape)}")
```

输出结果如下：

```shell #test-result id="v2-vbmk"
img: type=Image device=npu shape=(3, 256, 256)
boxes: type=BoundingBoxes device=cpu shape=(3, 4)
mask: type=Mask device=npu shape=(1, 256, 256)
video: type=Video device=npu shape=(3, 3, 32, 32)
keypoints: type=KeyPoints device=cpu shape=(3, 2)
```

### 验证 TVTensor 是 `torch.Tensor` 的子类，搬到 NPU 后类型不变

`Image` / `BoundingBoxes` / `Mask` / `Video` / `KeyPoints` 都是 **`torch.Tensor` 的子类**。换句话说：

- `isinstance(img_dp, torch.Tensor)` 永远为 `True`
- 所有原生 tensor 接口（`.sum()` / `.to(...)` / `torch.cat(...)` / `tensor.shape`）都能用
- **transforms 就是按这个子类类型做 dispatch 的**——这正是为什么 `BoundingBoxes` 能跟 `Image` 一起 transform

下面用 **python** 执行：

```python #test id="v2-tvtensors"
import torch
from torchvision import tv_tensors

img_dp = tv_tensors.Image(torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8))
print(f"{isinstance(img_dp, torch.Tensor) = }")
print(f"{img_dp.dtype = }, {img_dp.shape = }, {img_dp.sum() = }")
img_npu = img_dp.to('npu:0')   # TVTensor 子类化不被打断，搬到 NPU 后仍然是 tv_tensors.Image
print(f"{isinstance(img_npu, torch.Tensor) = }")
print(f"{img_npu.dtype = }, {img_npu.shape = }, {img_npu.device.type = }, {img_npu.sum().cpu() = }")
```

输出结果如下：

```shell #test-result id="v2-tvtensors" fuzzy='xxx'
isinstance(img_dp, torch.Tensor) = True
img_dp.dtype = torch.uint8, img_dp.shape = torch.Size([3, 256, 256]), img_dp.sum() = tensor(xxx)
isinstance(img_npu, torch.Tensor) = True
img_npu.dtype = torch.uint8, img_npu.shape = torch.Size([3, 256, 256]), img_npu.device.type = 'npu', img_npu.sum().cpu() = tensor(xxx)
```

### 把 (图像, 标注字典) 一起传进 transform，验证字典结构原样返回

`transforms` 只看 TVTensor **类型**做 dispatch，外来的 str / int / tuple / dict 原样穿透。所以你可以传任意嵌套结构——单 image、`(img, target)` 元组、dict、嵌套 dict——返回**同结构**：

下面用 **python** 执行：

```python #test id="v2-input-structure"
import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

np.random.seed(0)
arr = (np.random.rand(256, 256, 3) * 255).astype('uint8')
img = tv_tensors.Image(torch.from_numpy(arr).permute(2, 0, 1))
H, W = img.shape[-2:]
boxes = tv_tensors.BoundingBoxes(
    [[15, 10, 370, 510], [275, 340, 510, 510], [130, 345, 210, 425]],
    format="XYXY", canvas_size=(H, W))

# 经典检测 dataset 形态：image + target dict
target = {
    "boxes": boxes,                                  # TVTensor → 会被 transform
    "labels": torch.arange(boxes.shape[0]),           # 普通 Tensor → 穿透（原样返回）
    "this_is_ignored": ("arbitrary", {"structure": "!"}),  # 任意对象 → 穿透
}
img_npu = img.to('npu:0')
out_img, out_target = v2.Compose([v2.Resize(size=(128, 128))])(img_npu, target)
print(f"{out_img.device.type = }, {type(out_target).__name__ = }")
print(f"{out_target['boxes'].shape = }, {out_target['boxes'].format.name = }, {tuple(out_target['boxes'].canvas_size) = }")
print(f"{out_target['labels'].tolist() = }")
print(f"{out_target['this_is_ignored'] = }")
```

输出结果如下：

```shell #test-result id="v2-input-structure"
out_img.device.type = 'npu', type(out_target).__name__ = 'dict'
out_target['boxes'].shape = torch.Size([3, 4]), out_target['boxes'].format.name = 'XYXY', tuple(out_target['boxes'].canvas_size) = (128, 128)
out_target['labels'].tolist() = [0, 1, 2]
out_target['this_is_ignored'] = ('arbitrary', {'structure': '!'})
```

### 写一个返回 TVTensor 的自定义 Dataset，直接接上 v2 transform

你只要保证自定义 Dataset 的 `__getitem__` 返回**已经是 TVTensor**的对象，v2 transform 就能直接用：

下面用 **python** 执行：

```python #test id="v2-dataset-interop"
import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

class SyntheticDetectionDataset:
    """模拟 CocoDetection 的 (image, target_dict) 形态，无网络依赖。"""
    def __len__(self):
        return 2
    def __getitem__(self, idx):
        rng = np.random.RandomState(idx)   # 确定性合成
        arr = (rng.rand(256, 256, 3) * 255).astype('uint8')
        img = tv_tensors.Image(torch.from_numpy(arr).permute(2, 0, 1))
        boxes = tv_tensors.BoundingBoxes(
            [[10, 20, 100, 200], [50, 60, 150, 250]],
            format="XYXY", canvas_size=img.shape[-2:])
        return img, {"boxes": boxes, "labels": torch.zeros(2, dtype=torch.int64)}

dataset = SyntheticDetectionDataset()
img, target = dataset[0]
# 先看一眼 dataset 直接返什么——TVTensor + 完整 metadata
print(f"{type(img).__name__ = }, {tuple(img.shape) = }")
print(f"{type(target['boxes']).__name__ = }, {tuple(target['boxes'].shape) = }, {target['boxes'].format.name = }, {tuple(target['boxes'].canvas_size) = }")
# 然后正常过 Compose
img_npu, target_npu = v2.Compose([v2.Resize(size=(128, 128))])(img.to('npu:0'), target)
print(f"after resize → {type(img_npu).__name__ = }, {img_npu.device.type = }, {tuple(img_npu.shape) = }")
print(f"after resize → {type(target_npu['boxes']).__name__ = }, {tuple(target_npu['boxes'].shape) = }, {target_npu['boxes'].format.name = }, {tuple(target_npu['boxes'].canvas_size) = }")
print(f"labels: {target_npu['labels'].tolist()}")
print(f"len(dataset): {len(dataset)}")
```

输出结果如下：

```shell #test-result id="v2-dataset-interop"
type(img).__name__ = 'Image', tuple(img.shape) = (3, 256, 256)
type(target['boxes']).__name__ = 'BoundingBoxes', tuple(target['boxes'].shape) = (2, 4), target['boxes'].format.name = 'XYXY', tuple(target['boxes'].canvas_size) = (256, 256)
after resize → type(img_npu).__name__ = 'Image', img_npu.device.type = 'npu', tuple(img_npu.shape) = (3, 128, 128)
after resize → type(target_npu['boxes']).__name__ = 'BoundingBoxes', tuple(target_npu['boxes'].shape) = (2, 4), target_npu['boxes'].format.name = 'XYXY', tuple(target_npu['boxes'].canvas_size) = (128, 128)
labels: [0, 0]
len(dataset): 2
```

## 外部链接

- GitHub：[pytorch/vision](https://github.com/pytorch/vision)
- 文档中心：[Torchvision Docs](https://pytorch.org/vision/stable/index.html)
