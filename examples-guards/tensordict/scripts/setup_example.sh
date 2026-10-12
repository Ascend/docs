#!/usr/bin/env bash
# Install the tested TensorDict checkout without replacing the image's NPU torch.
set -euo pipefail

if [[ "${1:-}" != "tensordict_npu" ]]; then
  echo "unknown TensorDict examples profile: ${1:-<missing>}" >&2
  exit 2
fi

: "${TARGET_ROOT:?TARGET_ROOT is required}"
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# The CANN base image may contain no PyTorch at all (examples run #1).
# Follow this project's quick-start setup: reuse a matching pair when present,
# otherwise install the two pinned wheels from the cluster cache + Ascend index.
if python - <<'PY'
try:
    import torch
    import torch_npu
except Exception as exc:
    print(f"torch stack probe failed: {exc}")
    raise SystemExit(1)
print(f"found torch={torch.__version__} torch_npu={torch_npu.__version__}")
raise SystemExit(
    0 if torch.__version__.startswith("2.9.0")
    and torch_npu.__version__.startswith("2.9.0") else 1
)
PY
then
  echo "reusing compatible torch/torch_npu stack"
else
  echo "installing torch==2.9.0 torch_npu==2.9.0.post2"
  python -m pip install \
    --index-url http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple \
    --trusted-host cache-service.nginx-pypi-cache.svc.cluster.local \
    --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi \
    torch==2.9.0 torch_npu==2.9.0.post2
fi

python - <<'PY'
import torch
import torch_npu

print(f"torch={torch.__version__} torch_npu={torch_npu.__version__}")
if not torch.__version__.startswith("2.9.0") or not torch_npu.__version__.startswith("2.9.0"):
    raise SystemExit("expected the CANN 9.1 image's torch/torch_npu 2.9 stack")
if not torch.npu.is_available() or torch.npu.device_count() < 1:
    raise SystemExit("NPU is unavailable; refusing to run tutorials on CPU")
print(f"NPU device count: {torch.npu.device_count()}")
PY

# Match the already-validated TensorDict quick-start build path. Runtime
# requirements are installed explicitly so the source install cannot resolve a
# PyPI torch wheel over the image's torch_npu-compatible build.
python -m pip install 'uv' 'cmake>=3.22'
python -m pip install 'numpy' 'cloudpickle' 'packaging' 'importlib_metadata' 'orjson' 'pyvers>=0.2,<0.3'
uv pip install --system --no-deps -e "$TARGET_ROOT" --config-settings editable_mode=compat

python - <<'PY'
import os
from pathlib import Path
import tensordict
import torch
import torch_npu

source = Path(tensordict.__file__).resolve()
target = Path(os.environ["TARGET_ROOT"]).resolve()
print(f"tensordict={tensordict.__version__} source={source}")
if not source.is_relative_to(target):
    raise SystemExit(f"TensorDict is not loaded from the tested checkout: {source}")
if not torch.npu.is_available():
    raise SystemExit("NPU became unavailable after TensorDict installation")
PY
