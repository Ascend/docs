#!/usr/bin/env bash
# Prepare the CI environment for one supported onnxruntime example.
# $1 is the manifest profile. Unknown profiles fail before any install.
# Do not patch files under TARGET_ROOT.
#
# Both profiles are host CPU. The CANN image may already contain
# torch_npu and an onnxruntime build; those are removed so
# getting_started.py stays on CPUExecutionProvider and quantize_static
# does not open a CANN session.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"
HERE=$(cd "$(dirname "$0")" && pwd)

# Public PyPI. The engine exports the Ascend wheel index as
# PIP_EXTRA_INDEX_URL for torch_npu; these profiles must not see it.
PYPI_INDEX=https://pypi.org/simple
# torch 2.9.0 matches the CPU line this example was verified with:
# getting_started.py calls legacy torch.onnx.export (dynamic_axes),
# which imports onnxscript. The +cpu wheel is fetched by direct URL
# from download.pytorch.org so the resolver cannot substitute a CUDA
# build. Upstream requirements.txt points at the cu128 index; do not
# follow that. Do not install the CANN-paired torch_npu.
TORCH_CPU_VERSION=2.9.0

supported_profiles() {
  declare -F | awk '/^declare -f setup_/ { sub(/^declare -f setup_/, ""); print }' | paste -sd' ' -
}

install_ort_cpu_stack() {
  python3 -m pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-cann || true
  # numpy<2: a NumPy 2 install has failed at `import onnxruntime`
  # before the example started. onnx's own lower bound is 1.23.2.
  python3 -m pip install \
    onnxruntime \
    onnx \
    'numpy>=1.23.2,<2' \
    "$@"
  python3 "$HERE/check_ort_providers.py"
}

install_cpu_torch() {
  python3 -m pip uninstall -y torch torch_npu || true
  python3 -m pip install \
    filelock \
    'networkx>=2.5.1' \
    jinja2 \
    'fsspec>=0.8.5' \
    setuptools \
    'typing-extensions>=4.10.0' \
    'sympy>=1.13.3' \
    onnxscript
  local cp_abi
  cp_abi=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
  python3 -m pip install --no-deps \
    "https://download.pytorch.org/whl/cpu/torch-${TORCH_CPU_VERSION}%2Bcpu-${cp_abi}-${cp_abi}-manylinux_2_28_aarch64.whl"
  python3 "$HERE/check_cpu_torch.py"
}

setup_cpu-python() {
  install_ort_cpu_stack
  install_cpu_torch
}

setup_quant-cpu() {
  install_ort_cpu_stack pillow
}

if ! declare -F "setup_${PROFILE}" >/dev/null 2>&1; then
  echo "unknown profile: ${PROFILE} (supported: $(supported_profiles))" >&2
  exit 1
fi

export PIP_INDEX_URL="$PYPI_INDEX"
unset PIP_EXTRA_INDEX_URL
PIP_CONSTRAINT="$(cd "$HERE/.." && pwd)/constraints-npu.txt"
export PIP_CONSTRAINT

"setup_${PROFILE}"
