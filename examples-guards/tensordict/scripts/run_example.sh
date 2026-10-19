#!/usr/bin/env bash
# Run the original upstream tutorial with NPU as the default tensor device.
set -euo pipefail

: "${TARGET_ROOT:?TARGET_ROOT is required}"
: "${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"
entry="${1:-}"
case "$entry" in
  tutorials/sphinx_tuto/export.py|\
  tutorials/sphinx_tuto/functional.py|\
  tutorials/sphinx_tuto/tensordict_keys.py|\
  tutorials/sphinx_tuto/tensordict_shapes.py|\
  tutorials/sphinx_tuto/tensordict_preallocation.py|\
  tutorials/sphinx_tuto/tensordict_slicing.py|\
  tutorials/sphinx_tuto/streamed_tensordict.py|\
  tutorials/sphinx_tuto/serialization_speed.py) ;;
  *) echo "unsupported TensorDict tutorial entry: ${entry:-<missing>}" >&2; exit 2 ;;
esac

if [[ "${OVERLAY_ARGS:-[]}" != '[]' ]]; then
  echo "TensorDict tutorials have no CLI overlay arguments" >&2
  exit 2
fi

examples_root="${EXAMPLES_ROOT:-$TARGET_ROOT}"
tutorial="$examples_root/$entry"
if [[ ! -f "$tutorial" ]]; then
  echo "tutorial not found: $tutorial" >&2
  exit 1
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
mkdir -p "$CI_OUTPUT_DIR"
if [[ "$entry" == tutorials/sphinx_tuto/serialization_speed.py ]]; then
  # Headless rendering through the public matplotlib environment setting.
  export MPLBACKEND=Agg
fi

python - "$tutorial" <<'PY'
import runpy
import sys
from pathlib import Path

import torch
import torch_npu
from tensordict import TensorDictBase

if not torch.npu.is_available() or torch.npu.device_count() < 1:
    raise SystemExit("NPU is unavailable; refusing CPU fallback")
torch.npu.set_device(0)
torch.set_default_device("npu:0")

# Run the unmodified upstream file, including its own assertions.
namespace = runpy.run_path(sys.argv[1], run_name="__main__")

def require_npu(value, label):
    if isinstance(value, TensorDictBase):
        leaves = list(value.values(include_nested=True, leaves_only=True))
    elif isinstance(value, torch.Tensor):
        leaves = [value]
    elif isinstance(value, (tuple, list)):
        leaves = [leaf for part in value for leaf in require_npu(part, label)]
    elif isinstance(value, dict):
        leaves = [leaf for part in value.values() for leaf in require_npu(part, label)]
    else:
        raise SystemExit(f"{label}: unexpected result type {type(value).__name__}")
    if not leaves or not all(isinstance(leaf, torch.Tensor) for leaf in leaves):
        raise SystemExit(f"{label}: no tensor leaves to verify")
    bad = [str(leaf.device) for leaf in leaves if leaf.device.type != "npu"]
    if bad:
        raise SystemExit(f"{label}: fell back from NPU; leaf devices={bad}")
    return leaves

name = Path(sys.argv[1]).name
if name == "functional.py":
    checked = require_npu(namespace.get("params_stack"), "parameter ensemble")
    checked += require_npu(namespace.get("y"), "functional_call output")
elif name == "export.py":
    x = namespace.get("x")
    checked = require_npu(x, "export input")
    checked += require_npu(list(namespace["model"].parameters()), "export model")
    exported_output = namespace["model_export"].module()(x=x)
    checked += require_npu(exported_output, "exported module output")
elif name == "tensordict_slicing.py":
    td = namespace["tensordict"]
    checked = require_npu(td, "slicing input")
    # Preserve the original CPU BoolTensor mask and exercise its actual result.
    masked = td[namespace["mask"]]
    checked += require_npu(masked, "CPU-mask/NPU-data selection")
    if list(masked.batch_size) != [6] or not bool((td["a"][:2] == 1).all()):
        raise SystemExit("slicing tutorial did not preserve its assignment/mask semantics")
elif name == "streamed_tensordict.py":
    checked = require_npu(namespace["bucket0"], "stream bucket")
    checked += require_npu(namespace["data_cont"], "initial nested densification")
    checked += require_npu(namespace["td"], "asynchronous nested densification")
elif name == "serialization_speed.py":
    checked = require_npu(list(namespace["LAYOUTS"].values()), "serialization layouts")
else:
    checked = require_npu(namespace.get("tensordict"), "tutorial TensorDict")
torch.npu.synchronize()
print(f"NPU tutorial passed: {sys.argv[1]} ({len(checked)} NPU tensor leaves)")
PY
