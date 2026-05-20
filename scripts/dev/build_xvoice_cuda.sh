#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PROJECT_DIR="$ROOT_DIR/projects/x-voice-ggml-cpp"
BUILD_DIR="${XVOICE_CPP_CUDA_BUILD_DIR:-$PROJECT_DIR/build-cuda}"
JOBS="${XVOICE_BUILD_JOBS:-8}"
ZH_FRONTEND="${XVOICE_ENABLE_ZH_FRONTEND:-AUTO}"
CUDA_NO_VMM="${XVOICE_GGML_CUDA_NO_VMM:-ON}"
CUDA_ARCH="${XVOICE_CUDA_ARCHITECTURES:-}"

if [[ -z "$CUDA_ARCH" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.')"
  fi
fi
CUDA_ARCH="${CUDA_ARCH:-89}"

cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" \
  -DXVOICE_ALLOW_WORKBENCH_GGML_FALLBACK=ON \
  -DXVOICE_ENABLE_ZH_FRONTEND="$ZH_FRONTEND" \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_NO_VMM="$CUDA_NO_VMM" \
  -DGGML_NATIVE=OFF \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "$BUILD_DIR" -j "$JOBS"

echo "xvoice cuda build: ok"
echo "binary: $BUILD_DIR/x-voice-cli"
echo "cuda_architectures: $CUDA_ARCH"
