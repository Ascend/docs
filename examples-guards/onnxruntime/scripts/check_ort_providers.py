#!/usr/bin/env python3
"""Fail unless the installed onnxruntime wheel is the CPU build."""
from importlib.metadata import version

import onnxruntime as ort

providers = ort.get_available_providers()
print('onnxruntime', version('onnxruntime'))
print('onnx', version('onnx'))
print('available_providers', providers)
if 'CANNExecutionProvider' in providers:
    raise SystemExit(
        'CPU profiles must install the CPU onnxruntime wheel, not onnxruntime-cann')
if 'CPUExecutionProvider' not in providers:
    raise SystemExit('CPUExecutionProvider missing after installing onnxruntime')
