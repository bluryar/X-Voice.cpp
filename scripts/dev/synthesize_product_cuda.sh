#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PROJECT_DIR="$ROOT_DIR/projects/x-voice-ggml-cpp"
BUILD_DIR="${XVOICE_CPP_CUDA_BUILD_DIR:-$PROJECT_DIR/build-cuda}"
BIN="${XVOICE_CLI:-$BUILD_DIR/x-voice-cli}"
MODEL="${XVOICE_MODEL:-$ROOT_DIR/models/x-voice-f32.gguf}"
REF_WAV="${XVOICE_REF_WAV:-$ROOT_DIR/models/test.wav}"
TEXT="${XVOICE_TEXT:-发帖人（弟弟）在详细描述其五口之家的现状后，寻求处理家庭问题的建议，并提出了自己的初步计划。}"
TEXT_KIND="${XVOICE_TEXT_KIND:-plain}"
LANGUAGE="${XVOICE_LANGUAGE:-zh}"
OUT_DIR="${XVOICE_OUT_DIR:-/tmp/xvoice-cpp-product}"
OUTPUT_WAV="${XVOICE_OUTPUT_WAV:-$OUT_DIR/xvoice-product.wav}"
METADATA_JSON="${XVOICE_METADATA_JSON:-$OUT_DIR/xvoice-product.json}"
THREADS="${XVOICE_THREADS:-8}"

if [[ ! -x "$BIN" ]]; then
  bash "$PROJECT_DIR/scripts/dev/build_xvoice_cuda.sh"
fi

mkdir -p "$(dirname "$OUTPUT_WAV")" "$(dirname "$METADATA_JSON")"

NVIDIA_TF32_OVERRIDE="${NVIDIA_TF32_OVERRIDE:-0}" "$BIN" \
  --model "$MODEL" \
  --load-tensors \
  --synthesize \
  --text "$TEXT" \
  --text-kind "$TEXT_KIND" \
  --language "$LANGUAGE" \
  --ref-wav "$REF_WAV" \
  --preset product \
  --output-wav "$OUTPUT_WAV" \
  --metadata-json "$METADATA_JSON" \
  --progress \
  -b cuda \
  -t "$THREADS"

echo "xvoice product synthesis: ok"
echo "wav: $OUTPUT_WAV"
echo "metadata: $METADATA_JSON"
