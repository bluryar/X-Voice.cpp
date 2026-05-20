# X-Voice GGUF Quantization

`xvoice-quantize-gguf` rewrites a self-contained X-Voice GGUF into lighter variants.

The default policy is deliberately conservative: only large matrix weights used by
GGML `mul_mat` paths are converted. Conv, norm, bias, embedding, positional, and
small tensors stay in their original type. This keeps the Stage-2/SRP/Vocos graph
contract stable while still reducing the largest FFN, attention, and Vocos
pointwise projection weights.

Build:

```bash
cmake -S . -B build-cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda --target xvoice-quantize-gguf -j "$(nproc)"
```

Generate all common variants:

```bash
MODEL=/path/to/x-voice-f32.gguf OUT_DIR=/path/to/out \
  scripts/dev/quantize_xvoice_gguf.sh
```

Manual commands:

```bash
build-cuda/xvoice-quantize-gguf --input x-voice-f32.gguf --output x-voice-f16.gguf --type f16
build-cuda/xvoice-quantize-gguf --input x-voice-f32.gguf --output x-voice-q8_0.gguf --type q8_0
build-cuda/xvoice-quantize-gguf --input x-voice-f32.gguf --output x-voice-q6_k.gguf --type q6_k
build-cuda/xvoice-quantize-gguf --input x-voice-f32.gguf --output x-voice-q4_k.gguf --type q4_k
```

After quantization, validate metadata and tensor contracts:

```bash
build-cuda/x-voice-cli --model x-voice-q8_0.gguf --inspect --validate-tensors -b cuda
```

For release-quality artifacts, run a short CUDA synthesis smoke against each
variant and compare metadata duration/RMS/peak/high-frequency ratio before upload.
