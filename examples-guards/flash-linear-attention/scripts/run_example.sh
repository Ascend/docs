#!/usr/bin/env bash
set -eo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 example/train_text.py" >&2
  exit 2
fi

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
export PATH="/usr/local/sbin:$PATH"
export PYTHONNOUSERSITE=1
# shellcheck disable=SC1091
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export FLA_DISABLE_BACKEND_DISPATCH=0

exec python "$PROJECT_ROOT/scripts/run_example.py" "$1"
