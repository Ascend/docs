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
