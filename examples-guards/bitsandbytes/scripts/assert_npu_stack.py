#!/usr/bin/env python3
"""Fail unless the NPU compile stack matches the pinned pair."""
from __future__ import annotations

import sys
from importlib.metadata import version

import torch
import torch_npu
import triton


def main() -> int:
    torch_ok = torch.__version__.startswith('2.10.0')
    npu_ok = torch_npu.__version__.startswith('2.10.0')
    triton_ok = version('triton-ascend') == '3.2.2'
    npu_up = bool(torch.npu.is_available())
    print('torch', torch.__version__)
    print('torch_npu', torch_npu.__version__)
    print('npu_count', torch.npu.device_count())
    print('triton-ascend', version('triton-ascend'))
    print('triton_file', triton.__file__)
    print('npu_available', npu_up)
    if not torch_ok or not npu_ok or not triton_ok or not npu_up:
        print(
            'NPU compile stack mismatch: '
            f'torch={torch.__version__} torch_npu={torch_npu.__version__} '
            f'triton-ascend={version("triton-ascend")} npu_available={npu_up}',
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
