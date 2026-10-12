#!/usr/bin/env bash
# Prepare the CI environment for one supported speculators example.
# $1 is the manifest profile. Unknown profiles fail before any install.
set -euo pipefail

export PYTHONNOUSERSITE=1

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"
if [[ "$PROFILE" != train && "$PROFILE" != evaluate ]]; then
  echo "unknown profile: ${PROFILE} (supported: train evaluate)" >&2
  exit 1
fi

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"

export PATH="/usr/local/bin:/usr/local/sbin:$PATH"
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  set +u
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  set -u
fi
if [[ -d /usr/local/Ascend/driver/lib64/driver ]]; then
  export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}"
fi

exec python "$PROJECT_ROOT/scripts/setup_example.py" "$PROFILE"
