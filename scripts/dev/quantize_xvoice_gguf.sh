#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-cuda}"
MODEL="${MODEL:-/root/code/ggbond/models/x-voice-f32.gguf}"
OUT_DIR="${OUT_DIR:-$(dirname "${MODEL}")}"
JOBS="${JOBS:-$(nproc)}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DGGML_CUDA="${GGML_CUDA:-ON}" \
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
cmake --build "${BUILD_DIR}" --target xvoice-quantize-gguf -j "${JOBS}"

for type in f16 q8_0 q6_k q4_k; do
  out="${OUT_DIR}/x-voice-${type}.gguf"
  "${BUILD_DIR}/xvoice-quantize-gguf" \
    --input "${MODEL}" \
    --output "${out}" \
    --type "${type}" \
    --policy "${QUANT_POLICY:-linear}"
  ls -lh "${out}"
done
