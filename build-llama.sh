#!/usr/bin/env bash
# build-llama.sh - check out the pinned llama.cpp version and build it natively
# for this machine (CPU instruction set + local GPU architecture).
#
# Usage: ./build-llama.sh [git-ref]      (default: the ref in LLAMA_REF below)
#        CLEAN=1 ./build-llama.sh        (wipe build/ first)
set -euo pipefail

LLAMA_REF="${1:-v1.0.0}"
ROOT="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
SRC="$ROOT/llama.cpp"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

if [[ ! -d "$SRC/.git" ]]; then
  git clone https://github.com/ggml-org/llama.cpp "$SRC"
fi

git -C "$SRC" fetch --tags origin
git -C "$SRC" -c advice.detachedHead=false checkout "$LLAMA_REF"

[[ "${CLEAN:-0}" == 1 ]] && rm -rf "$SRC/build"

# Native build: GGML_NATIVE and CMAKE_CUDA_ARCHITECTURES=native target exactly
# this CPU and GPU, so the binaries are not portable to other machines.
cmake -S "$SRC" -B "$SRC/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -DGGML_NATIVE=ON \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_NCCL=ON \
  -DGGML_CCACHE=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_MTMD=OFF \
  -DLLAMA_BUILD_UI=OFF

cmake --build "$SRC/build" --config Release -j "$(nproc)"

echo
echo "Built llama.cpp $(git -C "$SRC" describe --tags --always) -> $SRC/build/bin/llama-server"
