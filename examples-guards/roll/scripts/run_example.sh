#!/usr/bin/env bash
# Run one supported ROLL example from a CI working copy of the target tree.
# $1 is the manifest entry path (a .yaml config relative to the target
# root). EXEC names the launchable file relative to the target root.
# Overlay CLI args come from OVERLAY_ARGS (a JSON array). Never git
# add/commit/push anything in the target tree.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <example-relpath>" >&2
  exit 2
fi

EXAMPLE_REL="$1"
TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
CI_OUTPUT_DIR="${CI_OUTPUT_DIR:?CI_OUTPUT_DIR is required}"

EXAMPLE_PATH="$TARGET_ROOT/$EXAMPLE_REL"
[[ -e "$EXAMPLE_PATH" ]] || { echo "example config not found: $EXAMPLE_PATH" >&2; exit 1; }

if [[ -z "${EXEC:-}" ]]; then
  echo "ROLL examples are launched through a start_*_pipeline.py; EXEC is required" >&2
  exit 2
fi
LAUNCH_PATH="$TARGET_ROOT/$EXEC"
if [[ ! -f "$LAUNCH_PATH" ]]; then
  echo "launcher not found: $LAUNCH_PATH" >&2
  exit 1
fi
if [[ "$EXAMPLE_REL" != examples/*.yaml ]]; then
  echo "example path must be an examples/*.yaml config, got: $EXAMPLE_REL" >&2
  exit 2
fi

for path in "$EXAMPLE_PATH" "$LAUNCH_PATH"; do
  case "$(realpath "$path")" in
    "$(realpath "$TARGET_ROOT")"/*) ;;
    *)
      echo "refusing to run a path outside the target checkout: $path" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$CI_OUTPUT_DIR"

cleanup_ray() {
  local status=$?
  echo "cleaning up local Ray cluster (status $status)"
  ray stop --force >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup_ray EXIT

# Vendor CANN/ATB env scripts assume a login shell and reference optional
# variables (e.g. $ZSH_VERSION) without ${VAR:-} guards. Under this
# project's `set -u` they die with "unbound variable"; relax strict mode
# only while sourcing vendor code, then restore it.
source_vendor_env() {
  local vendor_file="$1"
  if [[ ! -f "$vendor_file" ]]; then
    echo "vendor env script not found, skipping: $vendor_file"
    return 0
  fi
  set +eu
  # shellcheck disable=SC1090
  source "$vendor_file"
  set -eu
}

source_vendor_env /usr/local/Ascend/ascend-toolkit/set_env.sh
source_vendor_env /usr/local/Ascend/nnal/atb/set_env.sh

# vLLM-Ascend's CaMemAllocator asserts when
# PYTORCH_NPU_ALLOC_CONF=expandable_segments:True (v0.3.0 camem.py,
# tracked upstream at pytorch#147851).  ROLL clears it for vLLM workers,
# but the EngineCore child inherits the job-level value exported during
# setup, so clear it for every profile: vLLM requires the empty value,
# and the one-step FSDP2 smokes do not depend on expandable segments.
unset PYTORCH_NPU_ALLOC_CONF

# Single-node Ray contract for the thin engine. ROLL starts Ray itself and
# derives HCCL ranks from the per-worker ASCEND_RT_VISIBLE_DEVICES.  The
# domestic CANN base image does not pre-set that variable the way the
# upstream quay image does, so pin device 0 before Ray starts; setup
# already exported the multi-card list for train/rlvr profiles, so only
# fill the single-card default when nothing was injected.
# RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1 follows the v0.3.0
# Ascend env guide to keep Ray from rewriting the visibility list.
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=6379
export DASHBOARD_PORT=8265
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export RAY_DEDUP_LOGS=0
export PYTHONPATH="$TARGET_ROOT:${PYTHONPATH:-}"
export MODEL_DOWNLOAD_TYPE="${MODEL_DOWNLOAD_TYPE:-MODELSCOPE}"
export USE_MODELSCOPE="1"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="1"
export TQDM_MININTERVAL="15"

echo "=== diagnostics ==="
df -h /dev/shm
nproc
python - <<'PY'
import os, sys, pathlib
import torch, torch_npu
import roll
target = pathlib.Path(os.environ["TARGET_ROOT"]).resolve()
source = pathlib.Path(roll.__path__[0]).resolve()
print("roll import path:", source)
print("NPU available:", torch.npu.is_available(), "devices:", torch.npu.device_count())
PY

run_config_adapter() {
  python - "$@" <<'PY'
"""用 manifest 中的 Hydra 覆盖参数生成配置，再运行原始 ROLL launcher。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import shlex
import sys
import tempfile

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def expand_overlay(raw: str) -> list[str]:
    if not raw.strip():
        return []
    items = json.loads(raw)
    if items in (None, ''):
        return []
    if not isinstance(items, list):
        raise ValueError('OVERLAY_ARGS must be a JSON array')
    tokens = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise ValueError('OVERLAY_ARGS items must be non-empty strings')
        # 保留 Hydra 引号、列表和插值；普通 --参数允许写成一个 manifest 条目。
        if not item.startswith('--') and (item.startswith('~') or '=' in item):
            tokens.append(item)
        else:
            tokens.extend(shlex.split(os.path.expandvars(item), posix=True))
    return tokens


def compose_config(target_root: Path, config_path: str, config_name: str,
                   overrides: list[str]):
    examples = (target_root / 'examples').resolve()
    directory = (examples / config_path).resolve()
    directory.relative_to(examples)
    if not directory.is_dir():
        raise ValueError(f'upstream config directory not found: {directory}')
    if '/' in config_name or '\\' in config_name:
        raise ValueError('config_name must be a file name without a directory')
    with initialize_config_dir(config_dir=str(directory), version_base='1.1'):
        return compose(config_name=config_name, overrides=overrides)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launcher', required=True)
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--config_name', required=True)
    args, overrides = parser.parse_known_args(argv)
    for item in overrides:
        if item.startswith('-') or not (item.startswith('~') or '=' in item):
            parser.error(f'expected a Hydra key=value override, got: {item}')
    target = Path(os.environ['TARGET_ROOT']).resolve()
    launcher = (target / args.launcher).resolve()
    launcher.relative_to(target / 'examples')
    if not launcher.is_file():
        parser.error(f'launcher not found: {launcher}')
    config = compose_config(target, args.config_path, args.config_name, overrides)
    # 先解析环境变量和上游 defaults 引用，原始配置与源码均保持不变。
    text = OmegaConf.to_yaml(config, resolve=True)
    output = Path(os.environ['CI_OUTPUT_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'resolved_config.yaml').write_text(text, encoding='utf-8')
    # release launcher 的 initialize() 只接受相对自身的配置目录。
    with tempfile.TemporaryDirectory(prefix='.ci-roll-', dir=target / 'examples') as tmp:
        config_file = Path(tmp) / 'ci.yaml'
        config_file.write_text(text, encoding='utf-8')
        print(f'upstream config: {args.config_path}/{args.config_name}.yaml', flush=True)
        print(f'resolved CI config: {output / "resolved_config.yaml"}', flush=True)
        return subprocess.run(
            [sys.executable, str(launcher), '--config_path', Path(tmp).name,
             '--config_name', 'ci'], cwd=target, check=False).returncode


if __name__ == '__main__':
    if sys.argv[1:] == ['--print-overlay']:
        print(shlex.join(expand_overlay(os.environ.get('OVERLAY_ARGS', ''))))
        raise SystemExit(0)
    raise SystemExit(main())
PY
}

expand_overlay() {
  run_config_adapter --print-overlay
}

eval "EXTRA_ARGS=( $(expand_overlay) )"

echo "running $LAUNCH_PATH with ${#EXTRA_ARGS[@]} overlay args"
if ((${#EXTRA_ARGS[@]})); then
  printf 'overlay arg: %q
' "${EXTRA_ARGS[@]}"
fi

cd "$TARGET_ROOT"
run_config_adapter \
  --launcher "$EXEC" "${EXTRA_ARGS[@]}"
