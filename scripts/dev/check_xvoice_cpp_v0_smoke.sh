#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PROJECT_DIR="$ROOT_DIR/projects/x-voice-ggml-cpp"
BACKEND="${XVOICE_BACKEND:-cpu}"
if [[ -n "${XVOICE_CPP_BUILD_DIR:-}" ]]; then
  BUILD_DIR="$XVOICE_CPP_BUILD_DIR"
elif [[ "$BACKEND" == "cuda" ]]; then
  BUILD_DIR="$PROJECT_DIR/build-cuda"
else
  BUILD_DIR="$PROJECT_DIR/build"
fi
BIN="$BUILD_DIR/x-voice-cli"
MODEL="${XVOICE_MODEL:-$ROOT_DIR/models/x-voice-f32.gguf}"
REF_WAV="${XVOICE_REF_WAV:-$ROOT_DIR/models/test.wav}"
OUT_DIR="${XVOICE_CPP_SMOKE_DIR:-/tmp/xvoice-cpp-smoke}"
TEXT="${XVOICE_TEXT:-fa1|tie1|ren2 di4|di}"
LANGUAGE="${XVOICE_LANGUAGE:-zh}"
THREADS="${XVOICE_THREADS:-4}"
JOBS="${XVOICE_BUILD_JOBS:-4}"
REF_MAX_FRAMES="${XVOICE_REF_MAX_FRAMES:-64}"

CMAKE_ARGS=(
  -DXVOICE_ALLOW_WORKBENCH_GGML_FALLBACK=ON
  -DXVOICE_ENABLE_ZH_FRONTEND=AUTO
  -DCMAKE_BUILD_TYPE=Release
)

if [[ "$BACKEND" == "cuda" ]]; then
  CUDA_ARCH="${XVOICE_CUDA_ARCHITECTURES:-}"
  if [[ -z "$CUDA_ARCH" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.')"
  fi
  CUDA_ARCH="${CUDA_ARCH:-89}"
  CMAKE_ARGS+=(
    -DGGML_CUDA=ON
    -DGGML_CUDA_NO_VMM="${XVOICE_GGML_CUDA_NO_VMM:-ON}"
    -DGGML_NATIVE=OFF
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
  )
elif [[ "$BACKEND" != "cpu" ]]; then
  echo "xvoice smoke: unsupported XVOICE_BACKEND=$BACKEND (expected cpu or cuda)" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" -j "$JOBS"

"$BIN" --model "$MODEL" --inspect > "$OUT_DIR/inspect.txt"
if ! grep -q '^tokenizer_tokens: [1-9]' "$OUT_DIR/inspect.txt"; then
  echo "xvoice smoke: tokenizer.ggml.tokens is missing from GGUF; pass --vocab or re-export the bundle" >&2
  exit 1
fi

"$BIN" --model "$MODEL" --validate-tensors > "$OUT_DIR/tensor_contract.txt"
"$BIN" \
  --model "$MODEL" \
  --text "$TEXT" \
  --text-kind ipa \
  --language "$LANGUAGE" \
  --print-tokens > "$OUT_DIR/tokens.txt"

if [[ -f "$REF_WAV" ]]; then
  "$BIN" \
    --model "$MODEL" \
    --load-tensors \
    --synthesize \
    --text "$TEXT" \
    --text-kind ipa \
    --language "$LANGUAGE" \
    --ref-wav "$REF_WAV" \
    --ref-max-frames "$REF_MAX_FRAMES" \
    --sampler-mode no_cfg \
    --step-count 1 \
    --speed-value 6.75 \
    --output-wav "$OUT_DIR/xvoice-cpp-smoke.wav" \
    --metadata-json "$OUT_DIR/xvoice-cpp-smoke.json" \
    -b "$BACKEND" \
    -t "$THREADS" > "$OUT_DIR/synthesize.txt"
fi

if [[ "${XVOICE_CPP_FULL_SMOKE:-0}" == "1" ]]; then
  "$BIN" \
    --model "$MODEL" \
    --load-tensors \
    --synthesize \
    --text "$TEXT" \
    --text-kind ipa \
    --language "$LANGUAGE" \
    --ref-wav "$REF_WAV" \
    --ref-max-frames "$REF_MAX_FRAMES" \
    --sampler-mode default \
    --speed-value 6.75 \
    --output-wav "$OUT_DIR/xvoice-cpp-full.wav" \
    --metadata-json "$OUT_DIR/xvoice-cpp-full.json" \
    -b "$BACKEND" \
    -t "$THREADS" > "$OUT_DIR/synthesize-full.txt"
fi

echo "xvoice cpp v0 smoke: ok"
echo "backend: $BACKEND"
echo "artifacts: $OUT_DIR"
