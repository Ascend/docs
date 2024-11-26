API说明
==========

PyTorch-NPU 除了提供了 PyTorch 官方算子实现之外，也提供了大量高性能的自定义算子，详细的算子信息以及描述如下所示：

.. note::

   在运行下述示例之前，需要导入torch_npu扩展包 ``import torch_npu``

.. py:function:: _npu_dropout(self, p) -> (Tensor, Tensor)
    :module: torch_npu

    不使用种子(seed)进行dropout结果计数,与torch.dropout相似，优化NPU设备实现

    :param Tensor self: 输入张量
    :param Float p: 丢弃概率

    :rtype: (Tensor, Tensor)

示例:

.. code-block:: python
    :linenos:

    >>> input = torch.tensor([1.,2.,3.,4.]).npu()
    >>> input
    tensor([1., 2., 3., 4.], device='npu:0')
    >>> prob = 0.3
    >>> output, mask = torch_npu._npu_dropout(input, prob)
    >>> output
    tensor([0.0000, 2.8571, 0.0000, 0.0000], device='npu:0')
    >>> mask
    tensor([ 98, 255, 188, 186, 120, 157, 175, 159,  77, 223, 127,  79, 247, 151,
        253, 255], device='npu:0', dtype=torch.uint8)

.. py:function:: copy_memory_(dst, src, non_blocking=False) -> Tensor
    :module: torch_npu

    从src拷贝元素到self张量，并返回self

    约束说明:

    copy_memory_仅支持NPU张量,copy_memory_的输入张量应具有相同的dtype和设备index

    :param Tensor dst: 拷贝源张量
    :param Tensor sr: 返回张量所需数据类型
    :param non_blocking: 如果设置为True且此拷贝位于CPU和NPU之间，则拷贝可能相对于主机异步发生,在其他情况下，此参数没有效果
    :type non_blocking: Bool,Default: ``False``

    :rtype: Tensor

示例:

.. code-block:: python
    :linenos:

    >>> a=torch.IntTensor([0,  0, -1]).npu()
    >>> b=torch.IntTensor([1, 1, 1]).npu()
    >>> a.copy_memory_(b)
    tensor([1, 1, 1], device='npu:0', dtype=torch.int32)

.. py:function:: empty_with_format(size, dtype, layout, device, pin_memory, acl_format)
    :module: torch_npu

    返回一个填充未初始化数据的张量

    :param ListInt size: 定义输出张量shape的整数序列,可以是参数数量(可变值)，也可以是列表或元组等集合
    :param dtype: 返回张量所需数据类型;如果值为None，请使用全局默认值(请参见torch.set_default_tensor_type()).
    :type dtype: torch.dtype,Default: ``None``
    :param layout: 返回张量所需布局
    :type layout: torch.layout, Default: ``torch.strided``
    :param device: 返回张量的所需设备
    :type device: torch.device, Default: ``None``
    :param pin_memory: 返回张量的所需设备
    :type pin_memory: Bool, Default: ``False``
    :param acl_format: 返回张量所需内存格式
    :type acl_format: Int, Default: ``2``

    :rtype: Tensor

示例:

.. code-block:: python
    :linenos:

    >>> torch_npu.empty_with_format((2, 3), dtype=torch.float32, device="npu")
    tensor([[1., 1., 1.],
            [1., 1., 1.]], device='npu:0')

.. py:function:: fast_gelu(self) -> Tensor
    :module: torch_npu

    gelu的npu实现,支持FakeTensor模式

    :param Tensor self: 输入张量(只float16、float32)

    :rtype: Tensor

示例:

.. code-block:: python
    :linenos:

    # Normal
    >>> x = torch.rand(2).npu()
    >>> x
    tensor([0.5991, 0.4094], device='npu:0')
    >>> torch_npu.fast_gelu(x)
    tensor([0.4403, 0.2733], device='npu:0')

    # FakeTensorMode
    >>> from torch._subclasses.fake_tensor import FakeTensorMode
    >>> with FakeTensorMode():
    ...     x = torch.rand(2).npu()
    ...     torch_npu.fast_gelu(x)
    >>> FakeTensor(..., device='npu:0', size=(2,))

.. py:function:: npu_alloc_float_status(self) -> Tensor
    :module: torch_npu

    生成一个包含8个0的一维张量

    :param Tensor self: 输入张量

    :rtype: Tensor

示例:

.. code-block:: python
    :linenos:

    >>> input    = torch.randn([1,2,3]).npu()
    >>> output = torch_npu.npu_alloc_float_status(input)
    >>> input
    tensor([[[ 2.2324,  0.2478, -0.1056],
            [ 1.1273, -0.2573,  1.0558]]], device='npu:0')
    >>> output
    tensor([0., 0., 0., 0., 0., 0., 0., 0.], device='npu:0')

.. py:function:: npu_anchor_response_flags(self, featmap_size, stride, num_base_anchors) -> Tensor
    :module: torch_npu

    在单个特征图中生成锚点的责任标志

    :param Tensor self: 真值框，shape为[batch, 4]的2D张量
    :param ListInt[2] featmap_size: 特征图大小
    :param ListInt[2] strides: 当前水平的步长
    :param Int num_base_anchors: base anchors的数量

    :rtype: Tensor

示例:

.. code-block:: python
    :linenos:

    >>> x = torch.rand(100, 4).npu()
    >>> y = torch_npu.npu_anchor_response_flags(x, [60, 60], [2, 2], 9)
    >>> y.shape
    torch.Size([32400])

.. py:function:: npu_apply_adam(beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, use_locking, use_nesterov, out = (var, m, v))
    :module: torch_npu

    adam结果计数。

    :param Scalar beta1_power: beta1的幂
    :param Scalar beta2_power: beta2的幂
    :param Scalar lr: 学习率
    :param Scalar beta1: 一阶矩估计值的指数衰减率
    :param Scalar beta2: 二阶矩估计值的指数衰减率
    :param Scalar epsilon: 添加到分母中以提高数值稳定性的项数
    :param Tensor grad: 梯度
    :param Bool use_locking: 设置为True时使用lock进行更新操作
    :param Bool use_nesterov: 设置为True时采用nesterov更新
    :param Tensor var: 待优化变量。
    :param Tensor m: 变量平均值。
    :param Tensor v: 变量方差。

.. py:function:: npu_batch_nms(self, scores, score_threshold, iou_threshold, max_size_per_class, max_total_size, change_coordinate_frame=False, transpose_box=False) -> (Tensor, Tensor, Tensor, Tensor)

    :module: torch_npu

    根据batch分类计算输入框评分，通过评分排序，删除评分高于阈值(iou_threshold)的框，支持多批多类处理。通过NonMaxSuppression(nms)操作可有效删除冗余的输入框，提高检测精度。NonMaxSuppression：抑制不是极大值的元素，搜索局部的极大值，常用于计算机视觉任务中的检测类模型。

    :param Tensor self: 必填值，输入框的tensor,包含batch大小,数据类型Float16，输入示例：[batch_size, num_anchors, q, 4]，其中q=1或q=num_classes
    :param Tensor scores: 必填值，输入tensor，数据类型Float16，输入示例：[batch_size, num_anchors, num_classes]
    :param Float32 score_threshold: 必填值，指定评分过滤器的iou_threshold，用于筛选框，去除得分较低的框，数据类型Float32
    :param Float32 iou_threshold: 必填值，指定nms的iou_threshold，用于设定阈值，去除高于阈值的的框，数据类型Float32
    :param Int max_size_per_class: 必填值，指定每个类别的最大可选的框数，数据类型Int
    :param Int max_total_size: 必填值，指定每个batch最大可选的框数，数据类型Int
    :param Bool change_coordinate_frame: 可选值， 是否正则化输出框坐标矩阵，数据类型Bool(默认False)
    :param Bool transpose_box: 可选值，确定是否在此op之前插入转置，数据类型Bool。True表示boxes使用4,N排布。 False表示boxes使用过N,4排布

    输出说明:
    :param Tensor nmsed_boxes: shape为(batch, max_total_size, 4)的3D张量，指定每批次输出的nms框，数据类型Float16
    :param Tensor nmsed_scores: shape为(batch, max_total_size)的2D张量，指定每批次输出的nms分数，数据类型Float16
    :param Tensor nmsed_classes: shape为(batch, max_total_size)的2D张量，指定每批次输出的nms类，数据类型Float16
    :param Tensor nmsed_num: shape为(batch)的1D张量，指定nmsed_boxes的有效数量，数据类型Int32
    
    :rtype: Tensor

示例:

.. code-block:: python
    :linenos:

    >>> boxes = torch.randn(8, 2, 4, 4, dtype = torch.float32).to("npu")
    >>> scores = torch.randn(3, 2, 4, dtype = torch.float32).to("npu")
    >>> nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = torch_npu.npu_batch_nms(boxes, scores, 0.3, 0.5, 3, 4)
    >>> nmsed_boxes
    >>> nmsed_scores
    >>> nmsed_classes
    >>> nmsed_num

.. py:function:: npu_bert_apply_adam(lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm, weight_decay, step_size=None, adam_mode=0, *, out=(var,m,v))

    :module: torch_npu

    adam结果计数

    :param Tensor var: float16或float32类型张量
    :param Tensor m: 数据类型和shape与exp_avg相同
    :param Tensor v: 数据类型和shape与exp_avg相同
    :param Scalar lr: 数据类型与exp_avg相同
    :param Scalar beta1: 数据类型与exp_avg相同
    :param Scalar beta2: 数据类型与exp_avg相同
    :param Scalar epsilon: 数据类型与exp_avg相同
    :param Tensor grad: 数据类型和shape与exp_avg相同
    :param Scalar max_grad_norm: 数据类型与exp_avg相同
    :param Scalar global_grad_norm: 数据类型与exp_avg相同
    :param Scalar weight_decay: 数据类型与exp_avg相同
    :param Tensor step_size: 默认值为None - shape为(1, ),数据类型与exp_avg一致
    :param Int adam_mode: 选择adam模式。0表示“adam”, 1表示“mbert_adam”, 默认值为0

    关键字参数:
    out (Tensor，可选) - 输出张量。

示例:

.. code-block:: python
    :linenos:

    >>> var_in = torch.rand(321538).uniform_(-32., 21.).npu()
    >>> m_in = torch.zeros(321538).npu()
    >>> v_in = torch.zeros(321538).npu()
    >>> grad = torch.rand(321538).uniform_(-0.05, 0.03).npu()
    >>> max_grad_norm = -1.
    >>> beta1 = 0.9
    >>> beta2 = 0.99
    >>> weight_decay = 0.
    >>> lr = 0.
    >>> epsilon = 1e-06
    >>> global_grad_norm = 0.
    >>> var_out, m_out, v_out = torch_npu.npu_bert_apply_adam(lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm, weight_decay, out=(var_in, m_in, v_in))
    >>> var_out
    tensor([ 14.7733, -30.1218,  -1.3647,  ..., -16.6840,   7.1518,   8.4872], device='npu:0')

.. py:function:: npu_bmmV2(self, mat2, output_sizes) -> Tensor
    :module: torch_npu

    将矩阵“a”乘以矩阵“b”，生成“a*b”。支持FakeTensor模式

    :param Tensor self: 2D或更高维度矩阵张量。数据类型：float16、float32、int32。格式：[ND, NHWC, FRACTAL_NZ]
    :param Tensor mat2: 2D或更高维度矩阵张量。数据类型：float16、float32、int32。格式：[ND, NHWC, FRACTAL_NZ]
    :param ListInt[] output_sizes: 输出的shape，用于matmul的反向传播

    :rtype: Tensor

示例:

.. code-block:: python
    :linenos:

    >>> mat1 = torch.randn(10, 3, 4).npu()
    >>> mat2 = torch.randn(10, 4, 5).npu()
    >>> res = torch_npu.npu_bmmV2(mat1, mat2, [])
    >>> res.shape
    torch.Size([10, 3, 5])

.. py:function:: npu_bounding_box_decode(rois, deltas, means0, means1, means2, means3, stds0, stds1, stds2, stds3, max_shape, wh_ratio_clip) -> Tensor
    :module: torch_npu

    根据rois和deltas生成标注框。自定义FasterRcnn算子

    :param Tensor rois: 区域候选网络(RPN)生成的region of interests(ROI)。shape为(N,4)数据类型为float32或float16的2D张量。“N”表示ROI的数量, “4”表示“x0”、“x1”、“y0”和“y1”
    :param Tensor deltas: RPN生成的ROI和真值框之间的绝对变化。shape为(N,4)数据类型为float32或float16的2D张量。“N”表示错误数,“4”表示“dx”、“dy”、“dw”和“dh”
    :param Float  means0: index
    :param Float  means1: index
    :param Float  means2: index
    :param Float  means33: index, 默认值为[0,0,0,0], "deltas" = "deltas" x "stds" + "means"
    :param Float  stds0: index
    :param Float  stds1: index
    :param Float  stds2: index
    :param Float  stds3: index, 默认值：[1.0,1.0,1.0,1.0], deltas" = "deltas" x "stds" + "means"
    :param ListInt[2] max_shape: shape[h, w], 指定传输到网络的图像大小。用于确保转换后的bbox shape不超过“max_shape”
    :param Float  wh_ratio_clip: 当前水平的步长
    :param Int num_base_anchors: “dw”和“dh”的值在(-wh_ratio_clip, wh_ratio_clip)范围内

    :rtype: Tensor

示例:

.. code-block:: python
    :linenos:

    >>> rois = torch.tensor([[1., 2., 3., 4.], [3.,4., 5., 6.]], dtype = torch.float32).to("npu")
    >>> deltas = torch.tensor([[5., 6., 7., 8.], [7.,8., 9., 6.]], dtype = torch.float32).to("npu")
    >>> output = torch_npu.npu_bounding_box_decode(rois, deltas, 0, 0, 0, 0, 1, 1, 1, 1, (10, 10), 0.1)
    >>> output
    tensor([[2.5000, 6.5000, 9.0000, 9.0000],
            [9.0000, 9.0000, 9.0000, 9.0000]], device='npu:0')