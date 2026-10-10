#!/usr/bin/env bash
set -eo pipefail

if [[ "${1:-}" != "npu" ]]; then
  echo "usage: $0 npu" >&2
  exit 2
fi

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
export PATH="/usr/local/sbin:$PATH"
export PYTHONNOUSERSITE=1
# CANN's environment script can reference unset shell variables.
# shellcheck disable=SC1091
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
npu-smi info
cd "$TARGET_ROOT"
echo "Installing FLA from upstream commit $(git rev-parse HEAD)"

python - <<'PY'
import tomllib
from pathlib import Path

project = tomllib.loads(Path("pyproject.toml").read_text())
extras = project["project"].get("optional-dependencies", {})
for name in ("npu",):
    if not extras.get(name):
        raise SystemExit(f"Target checkout does not declare the [{name}] extra")
    print(f"Upstream [{name}] requirements: {extras[name]}")
PY

# Match Quick Start's in-cluster pip cache for every install, including build dependencies.
export PIP_INDEX_URL="http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple"
export PIP_TRUSTED_HOST="cache-service.nginx-pypi-cache.svc.cluster.local"
echo "pip index: $PIP_INDEX_URL"

# Matches upstream Ascend installation; target extras own the dependency versions.
python -m pip install -U pip setuptools wheel
python -m pip install pybind11 cmake attrs sympy pyyaml scipy decorator einops
python -m pip install -e '.[npu]' \
  --constraint "$PROJECT_ROOT/constraints-npu.txt" \
  --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple
python -m pip freeze
