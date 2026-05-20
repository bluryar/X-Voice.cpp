#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PROJECT_DIR="$ROOT_DIR/projects/x-voice-ggml-cpp"
BUILD_DIR="${XVOICE_CPP_BUILD_DIR:-$PROJECT_DIR/build}"
BIN="$BUILD_DIR/x-voice-cli"
MODEL="${XVOICE_MODEL:-$ROOT_DIR/models/x-voice-f32.gguf}"
MANIFEST="${XVOICE_MANIFEST:-$ROOT_DIR/models/x-voice-f32.manifest.json}"
EXPORTER="$ROOT_DIR/projects/x-voice-ggml-py/scripts/cli/export_x_voice_gguf.py"

if [[ ! -x "$BIN" ]]; then
  cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" \
    -DXVOICE_ALLOW_WORKBENCH_GGML_FALLBACK=ON \
    -DCMAKE_BUILD_TYPE=Release
  cmake --build "$BUILD_DIR" -j "${XVOICE_BUILD_JOBS:-4}"
fi

inspect="$("$BIN" --model "$MODEL" --inspect)"
token_count="$(printf '%s\n' "$inspect" | awk '/^tokenizer_tokens:/ {print $2}')"
if [[ "${token_count:-0}" != "0" && "${1:-}" != "--force-write" ]]; then
  echo "self-contained GGUF: ok tokenizer_tokens=$token_count"
  echo "model: $MODEL"
  echo "manifest: $MANIFEST"
  exit 0
fi

if [[ "${1:-}" != "--force-write" ]]; then
  echo "self-contained GGUF: tokenizer tokens are missing; rerun with --force-write to rebuild $MODEL" >&2
  exit 1
fi

UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" \
  uv run python "$EXPORTER" \
    --write-tensors \
    --output "$MODEL" \
    --manifest "$MANIFEST"

"$BIN" --model "$MODEL" --inspect | awk '/^tokenizer_tokens:/ { if ($2 == 0) exit 1; print }'
echo "self-contained GGUF: rebuilt $MODEL"
