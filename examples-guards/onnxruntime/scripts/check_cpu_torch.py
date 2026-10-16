#!/usr/bin/env python3
"""Fail unless torch is a CPU build and torch_npu is not installed."""
import importlib.util

import torch

print('torch', torch.__version__)
cuda = torch.version.cuda
if '+cu' in torch.__version__ or cuda:
    raise SystemExit(
        f'cpu-python must install a CPU torch wheel, got {torch.__version__} cuda={cuda}')
if importlib.util.find_spec('torch_npu') is not None:
    raise SystemExit('cpu-python must not import torch_npu')
print('torch_npu absent')
