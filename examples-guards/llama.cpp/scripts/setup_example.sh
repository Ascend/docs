#!/usr/bin/env bash
# Prepare the CI environment for one supported llama.cpp example.
# $1 is the manifest profile. Unknown profiles fail before any install.
# EXEC and EXAMPLE_PATH come from the workflow as environment variables.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"
EXEC_REL="${EXEC:-}"
EXAMPLE_REL="${EXAMPLE_PATH:-}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# This pool reaches huggingface.co directly. Pin the endpoint so a
# leftover hf-mirror value cannot 308 a bare repo id back to the origin.
export HF_ENDPOINT=https://huggingface.co

# Published qwen2.5-0.5b-instruct-q4_0.gguf is 428730208 bytes. The floor
# rejects a truncated download; the header check is in fetch_gguf.py.
QWEN_MIN_BYTES=300000000
QWEN_HF_URL=https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf
QWEN_MS_URL=https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/master/qwen2.5-0.5b-instruct-q4_0.gguf

require_exec() {
  if [[ -z "$EXEC_REL" ]]; then
    echo "EXEC is required for profile ${PROFILE}" >&2
    exit 1
  fi
}

source_cann() {
  export PATH="/usr/local/sbin:$PATH"
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info
  fi
}

assert_cann_lib() {
  if ! compgen -G "$1/libggml-cann.*" >/dev/null; then
    echo "build produced no CANN backend library under $1" >&2
    exit 1
  fi
}

cmake_llama() {
  local -a extra=("$@")
  cmake -S "$TARGET_ROOT" -B "$TARGET_ROOT/build" \
    -DCMAKE_BUILD_TYPE=Release \
    "${extra[@]}"
}

build_target() {
  require_exec
  cmake --build "$TARGET_ROOT/build" --target "$(basename "$EXEC_REL")" -j "$(nproc)"
}

fetch_qwen() {
  python3 "$HERE/fetch_gguf.py" \
    --name qwen2.5-0.5b-instruct-q4_0.gguf \
    --min-bytes "$QWEN_MIN_BYTES" \
    --hf-url "$QWEN_HF_URL" \
    --ms-url "$QWEN_MS_URL"
}

setup_cann() {
  source_cann
  fetch_qwen
  cmake_llama -DGGML_CANN=on
  build_target
  assert_cann_lib "$TARGET_ROOT/build/bin"
}

setup_cpu() {
  cmake_llama
  build_target
}

setup_cmake-pkg() {
  source_cann
  fetch_qwen
  cmake -S "$TARGET_ROOT" -B "$TARGET_ROOT/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CANN=on \
    -DCMAKE_INSTALL_PREFIX="$TARGET_ROOT/inst"
  cmake --build "$TARGET_ROOT/build" -j "$(nproc)"
  cmake --install "$TARGET_ROOT/build" --prefix "$TARGET_ROOT/inst"
  cmake -S "$TARGET_ROOT/examples/simple-cmake-pkg" \
    -B "$TARGET_ROOT/examples/simple-cmake-pkg/build" \
    -DCMAKE_PREFIX_PATH="$TARGET_ROOT/inst/lib/cmake"
  cmake --build "$TARGET_ROOT/examples/simple-cmake-pkg/build" -j "$(nproc)"
  if ! compgen -G "$TARGET_ROOT/inst/lib/libggml-cann.*" >/dev/null \
    && ! compgen -G "$TARGET_ROOT/build/bin/libggml-cann.*" >/dev/null; then
    echo "build produced no CANN backend library under inst/lib or build/bin" >&2
    exit 1
  fi
  local bin="$TARGET_ROOT/examples/simple-cmake-pkg/build/llama-simple-cmake-pkg"
  if [[ ! -f "$bin" ]]; then
    echo "cmake-pkg binary not found: $bin" >&2
    exit 1
  fi
  # RUNPATH on the consumer binary does not apply to libggml's NEEDED
  # backends. The run step sources CANN again and prepends the toolkit.
  echo "LD_LIBRARY_PATH=$TARGET_ROOT/inst/lib" >> "$GITHUB_ENV"
}

supported_profiles() {
  declare -F | awk '/^declare -f setup_/ { sub(/^declare -f setup_/, ""); print }' | paste -sd' ' -
}

if ! declare -F "setup_${PROFILE}" >/dev/null 2>&1; then
  echo "unknown profile: ${PROFILE} (supported: $(supported_profiles))" >&2
  exit 1
fi

TARGET_ROOT="${TARGET_ROOT:?TARGET_ROOT is required}"
GITHUB_ENV="${GITHUB_ENV:?GITHUB_ENV is required}"

"setup_${PROFILE}"
