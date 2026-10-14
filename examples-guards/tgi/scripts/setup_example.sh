#!/usr/bin/env bash
# TGI examples guard: prepare the environment for one example job.
#
# $1 is the manifest entry's profile. Only `tgi-serve` exists today:
# build the release-opt launcher + router, install the Python server
# (doc §4 recipe, locked deps), and smoke-check the artifacts. Every
# scenario script then boots the service from the same checkout.
#
# Build economics: cargo target dir + rustup/cargo homes live under
# ~/.cache/tgi-examples, the runner's persistent volume. The first leg
# on a machine pays the full release-opt build (~2 h); later legs hit
# the incremental cache. max_parallel is 1 for tgi (see tgi-examples.yml),
# so no concurrent legs contend on the cargo target-dir lock.
#
# Env (provided by the engine, per the project script contract):
#   TARGET_ROOT / GITHUB_WORKSPACE / GITHUB_ENV
set -euo pipefail

PROFILE="${1:?usage: $0 <profile>}"

# Unknown profile must fail before installing anything (contract).
if [[ "$PROFILE" != "tgi-serve" ]]; then
    echo "unknown profile: ${PROFILE} (supported: tgi-serve)" >&2
    exit 1
fi

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
GITHUB_WORKSPACE="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
GITHUB_ENV="${GITHUB_ENV:?GITHUB_ENV is required}"

CLUSTER_PIP_HOST=cache-service.nginx-pypi-cache.svc.cluster.local
CLUSTER_PIP_INDEX="http://${CLUSTER_PIP_HOST}/pypi/simple"
ASCEND_PIP_INDEX=https://repo.huaweicloud.com/ascend/repos/pypi
ALIYUN_PIP_INDEX=https://mirrors.aliyun.com/pypi/simple/

# 1) pip index: prefer the cluster cache, fall back to the Aliyun
# mirror (same selection as the other projects' setup scripts). The uv
# env below mirrors the proven combo from the quick-start engine
# (UV_INDEX_STRATEGY=unsafe-best-match + UV_INSECURE_HOST for the http
# cluster cache); uv would otherwise refuse an http index.
if python -c "
import os
import urllib.error
import urllib.request
try:
    urllib.request.urlopen(os.environ['CLUSTER_PIP_INDEX'], timeout=3)
except urllib.error.HTTPError:
    pass
" 2>/dev/null; then
    PIP_INDEX_URL="$CLUSTER_PIP_INDEX"
    PIP_TRUSTED_HOST="$CLUSTER_PIP_HOST"
else
    PIP_INDEX_URL="$ALIYUN_PIP_INDEX"
    unset PIP_TRUSTED_HOST
fi
export PIP_INDEX_URL
export PIP_EXTRA_INDEX_URL="$ASCEND_PIP_INDEX"
export PIP_TIMEOUT=120
export UV_INDEX_URL="$PIP_INDEX_URL"
export UV_EXTRA_INDEX_URL="$ASCEND_PIP_INDEX"
export UV_INDEX_STRATEGY=unsafe-best-match
export UV_HTTP_TIMEOUT=120
# Container has no venv: uv pip install must target the system Python
# (same UV_SYSTEM_PYTHON=1 the quick-start engine sets; without it uv
# errors "No virtual environment found"). UV_NO_CACHE=1 keeps uv's
# own cache off the shared persistent volume - the cluster pip cache
# is the single shared cache, uv's would be a redundant corruption
# surface (same as the quick-start engine's proven combo).
export UV_SYSTEM_PYTHON=1
export UV_NO_CACHE=1
# Persistent-volume root (shared across the pool's runners, same disk
# that carries the HF/model caches): the pip wheel cache lives here so
# torch_npu is downloaded once and reused by later legs and runs
# instead of re-downloading every job (the default ~/.cache/pip would
# land on the same disk, but being explicit keeps it in one place).
TGI_CACHE_ROOT="${HOME}/.cache/tgi-examples"
export PIP_CACHE_DIR="$TGI_CACHE_ROOT/pip"
mkdir -p "$PIP_CACHE_DIR"
if [[ -n "${PIP_TRUSTED_HOST:-}" ]]; then
    export PIP_TRUSTED_HOST
    export UV_INSECURE_HOST="$PIP_TRUSTED_HOST"
fi
echo "pip index: $PIP_INDEX_URL"

# 2) CUDA exclusion list (same as the quick-start test class) so pip /
# uv resolvers never pull nvidia-* / cuda-* metapackages or flashinfer.
CONSTRAINTS_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/constraints-npu.txt"
export PIP_CONSTRAINT="$CONSTRAINTS_FILE"
export UV_CONSTRAINT="$CONSTRAINTS_FILE"

# 3) CANN toolkit env (ASCEND_HOME / LD_LIBRARY_PATH etc. for the build
# and the later service boot).
for candidate in /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/cann/set_env.sh; do
    if [[ -f "$candidate" ]]; then
        source "$candidate"
        echo "sourced CANN env from $candidate"
        break
    fi
done

# 4) System build deps (doc §4 prerequisites, same apt line as
# quick_start.md): the launcher links system OpenSSL (openssl-sys)
# which resolves through pkg-config at build time; protobuf-compiler
# provides the protoc binary PROTOC points at below.
apt-get update -qq
apt-get install -y -qq build-essential protobuf-compiler pkg-config libssl-dev curl python3-dev

# 5) torch stack: reuse the image's preinstalled torch_npu when
# importable; otherwise install the current recommended pair from the
# Ascend source (same probe-first approach as the quick-start harness).
if python -c "import torch, torch_npu" >/dev/null 2>&1; then
    python -c "import torch, torch_npu; print('reusing image torch stack:', torch.__version__, torch_npu.__version__)"
else
    echo "installing torch_npu from the Ascend source"
    python -m pip install torch_npu
fi

# 6) Rust toolchain under the persistent volume. rust-toolchain.toml
# pins 1.85.1; a cargo build auto-installs the pinned channel via
# rustup when the default differs. CARGO_TARGET_DIR on the persistent
# volume makes the release-opt build incremental across legs and runs.
export RUSTUP_HOME="$TGI_CACHE_ROOT/rustup"
export CARGO_HOME="$TGI_CACHE_ROOT/cargo"
export CARGO_TARGET_DIR="$TGI_CACHE_ROOT/target"
mkdir -p "$RUSTUP_HOME" "$CARGO_HOME" "$CARGO_TARGET_DIR"
if ! command -v cargo >/dev/null 2>&1; then
    echo "installing rustup (default toolchain 1.85.1)"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain 1.85.1 --profile minimal
fi
export PATH="$CARGO_HOME/bin:$PATH"

# 7) Rust build: release-opt launcher + router (doc §4 recipe).
cd "$TARGET_ROOT"
export PYO3_PYTHON="$(command -v python)"
export PROTOC="$(command -v protoc)"
echo "building release-opt launcher + router (target dir: $CARGO_TARGET_DIR)"
cargo build --profile release-opt -p text-generation-launcher -p text-generation-router-v3

# run-npu.sh resolves the binaries at <checkout>/target/release-opt;
# point the checkout at the persistent target dir so the scenario
# scripts run unchanged.
mkdir -p "$TARGET_ROOT/target"
ln -sfn "$CARGO_TARGET_DIR/release-opt" "$TARGET_ROOT/target/release-opt"

# 8) Python server (doc §4 recipe, locked deps). `-e server` with
# --no-deps: deps come from requirements_ascend.txt only, no drift via
# pyproject ranges. gen-server-raw does proto codegen only (gen-server
# would additionally install from the outdated requirements_gen.txt).
python -m pip install -q uv
uv pip install --no-build-isolation -r server/requirements_ascend.txt
uv pip install --no-build-isolation --no-deps -e server
uv pip install --no-build-isolation "grpcio-tools==1.84.0" "mypy-protobuf==3.6.0"
make -C server gen-server-raw

# 9) Artifact smoke check (doc check-build equivalent).
"$TARGET_ROOT/target/release-opt/text-generation-launcher" --version
"$TARGET_ROOT/target/release-opt/text-generation-router" --version
python -c "import text_generation_server; print('server import ok')"

# 10) Persist what the run step still needs: pip plumbing for
# start-tgi.sh's lazy `pip install modelscope` (first model use) and
# the cargo bin dir on PATH. The uv env must be persisted too:
# start-tgi.sh prefers `uv pip install modelscope` when uv is on PATH
# (step 8 installs it), and the run step is a fresh process - the
# exports above stay in the setup process. Without UV_SYSTEM_PYTHON
# that lazy uv call dies with "No virtual environment found" in the
# container (no venv), which is what the first green-setup run hit.
{
    echo "PIP_INDEX_URL=$PIP_INDEX_URL"
    echo "PIP_EXTRA_INDEX_URL=$ASCEND_PIP_INDEX"
    echo "PIP_CONSTRAINT=$CONSTRAINTS_FILE"
    echo "UV_SYSTEM_PYTHON=1"
    echo "UV_NO_CACHE=1"
    echo "UV_INDEX_URL=$UV_INDEX_URL"
    echo "UV_EXTRA_INDEX_URL=$ASCEND_PIP_INDEX"
    echo "UV_INDEX_STRATEGY=unsafe-best-match"
    echo "UV_HTTP_TIMEOUT=120"
    echo "UV_CONSTRAINT=$CONSTRAINTS_FILE"
    echo "PATH=$CARGO_HOME/bin:$PATH"
    if [[ -n "${PIP_TRUSTED_HOST:-}" ]]; then
        echo "PIP_TRUSTED_HOST=$PIP_TRUSTED_HOST"
        echo "UV_INSECURE_HOST=$PIP_TRUSTED_HOST"
    fi
} >> "$GITHUB_ENV"

echo "setup tgi-serve done"
