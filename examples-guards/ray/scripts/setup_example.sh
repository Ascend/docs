#!/usr/bin/env bash
# Install a released Ray target from PyPI or an exact development commit wheel.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
TORCH_VERSION=2.9.0
TORCH_NPU_VERSION=2.9.0.post2
ASCEND_PIP_INDEX=https://repo.huaweicloud.com/ascend/repos/pypi
CLUSTER_PIP_HOST=cache-service.nginx-pypi-cache.svc.cluster.local
export PIP_INDEX_URL="${PIP_INDEX_URL:-http://${CLUSTER_PIP_HOST}/pypi/simple}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-$CLUSTER_PIP_HOST}"

source_cann() {
  export PATH="/usr/local/sbin:$PATH"
  # The selected CANN container provides this file.
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
}

ensure_torch_stack() {
  if python -c "
import torch, torch_npu
raise SystemExit(
    0 if torch.__version__.startswith('${TORCH_VERSION}')
    and torch_npu.__version__.startswith('${TORCH_NPU_VERSION}')
    else 1
)
"; then
    echo "reusing torch ${TORCH_VERSION} / torch_npu ${TORCH_NPU_VERSION} stack"
  else
    echo "installing torch==${TORCH_VERSION} torch_npu==${TORCH_NPU_VERSION}"
    python -m pip install \
      --extra-index-url "$ASCEND_PIP_INDEX" \
      "torch==${TORCH_VERSION}" \
      "torch_npu==${TORCH_NPU_VERSION}"
  fi
  python -c 'import torch, torch_npu; assert torch.npu.is_available(); print(torch.__version__, torch_npu.__version__, torch.npu.device_count())'
}

install_test_dependencies() {
  local profile="$1"
  local test_requirements="$TARGET_ROOT/python/requirements/test-requirements.txt"
  local train_requirements="$TARGET_ROOT/python/requirements/ml/train-test-requirements.txt"
  local resolver
  local resolver_args=(--requirement "$test_requirements" pytest)
  local resolved
  local packages
  resolver="$(dirname "${BASH_SOURCE[0]}")/resolve_test_requirements.py"

  if [[ "$profile" == "train" ]]; then
    resolver_args+=(--requirement "$train_requirements" boto3)
  fi

  resolved=$(python "$resolver" "${resolver_args[@]}")
  mapfile -t packages <<< "$resolved"
  python -m pip install "${packages[@]}"
}

link_target_ray_tests() {
  local setup_dev="$TARGET_ROOT/python/ray/setup-dev.py"
  local verifier
  verifier="$(dirname "${BASH_SOURCE[0]}")/verify_ray_test_support.py"

  if [[ ! -f "$setup_dev" ]]; then
    echo "target Ray setup-dev.py is missing: $setup_dev" >&2
    exit 1
  fi

  python "$setup_dev" --yes --allow tests
  python "$verifier" "$TARGET_ROOT"
}

ray_version() {
  python - "$TARGET_ROOT/python/ray/_version.py" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(r"""^version = ['\"]([^'\"]+)['\"]""", text, re.MULTILINE)
if not match:
    raise SystemExit(f"could not read Ray version from {sys.argv[1]}")
print(match.group(1))
PY
}

install_target_ray() {
  local extras="$1"
  local sha version py_tag arch wheel_url requirement
  version=$(ray_version)

  if [[ "$version" == *dev* ]]; then
    sha=$(git -C "$TARGET_ROOT" rev-parse HEAD)
    py_tag=$(python -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
    arch=$(uname -m)
    if [[ "$arch" != "aarch64" && "$arch" != "arm64" ]]; then
      echo "Ray Ascend guard requires aarch64, got: $arch" >&2
      exit 1
    fi

    wheel_url="https://s3-us-west-2.amazonaws.com/ray-wheels/master/${sha}/ray-${version}-${py_tag}-${py_tag}-manylinux2014_aarch64.whl"
    if [[ -n "$extras" ]]; then
      requirement="ray[${extras}] @ ${wheel_url}"
    else
      requirement="ray @ ${wheel_url}"
    fi

    echo "installing Ray ${version} from target commit ${sha}"
    if ! python -m pip install "$requirement"; then
      echo "exact master wheel is unavailable: $wheel_url" >&2
      exit 1
    fi
  else
    echo "installing released Ray ${version} from PyPI"
    if [[ -n "$extras" ]]; then
      python -m pip install "ray[${extras}]==${version}"
    else
      python -m pip install "ray==${version}"
    fi
  fi
  python -c 'import ray; print("ray", ray.__version__, ray.__file__)'
}

setup_core() {
  install_target_ray ""
  install_test_dependencies core
  link_target_ray_tests
}

setup_train() {
  ensure_torch_stack
  install_target_ray train
  install_test_dependencies train
  link_target_ray_tests
}

setup_npu() {
  ensure_torch_stack
  install_target_ray ""
  install_test_dependencies core
  link_target_ray_tests
}

setup_data() {
  ensure_torch_stack
  install_target_ray data
  install_test_dependencies core
  link_target_ray_tests
}

setup_serve() {
  ensure_torch_stack
  install_target_ray serve
  install_test_dependencies core
  link_target_ray_tests
}

setup_tune() {
  ensure_torch_stack
  install_target_ray tune
  install_test_dependencies core
  link_target_ray_tests
}

supported_profiles() {
  declare -F | awk '/^declare -f setup_/ { sub(/^declare -f setup_/, ""); print }' | paste -sd' ' -
}

if ! declare -F "setup_${PROFILE}" >/dev/null 2>&1; then
  echo "unknown profile: ${PROFILE} (supported: $(supported_profiles))" >&2
  exit 1
fi

source_cann
python -m pip install -U pip
"setup_${PROFILE}"
