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

## Local Benchmark

Measured on an RTX 4060 Ti with `NVIDIA_TF32_OVERRIDE=0`, `--preset product`,
`cfg_nonlayered`, 32 sampler steps, the default zh sample text, and
`/root/code/ggbond/models/test.wav`:

| model | size | wall | sampler | load | mel max abs vs f32 | mel mean abs vs f32 | wav max abs vs f32 | wav mean abs vs f32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| f32 | 1544.1 MiB | 10.74s | 9.222s | 0.515s | 0 | 0 | 0 | 0 |
| f16 | 1060.5 MiB | 8.89s | 7.316s | 0.513s | 1.40578 | 0.01232 | 0.89212 | 0.01086 |
| q8_0 | 847.4 MiB | 8.52s | 7.268s | 0.303s | 0.78026 | 0.02397 | 1.01920 | 0.01708 |
| q6_k | 790.0 MiB | 8.66s | 7.407s | 0.301s | 0.89031 | 0.04256 | 0.68326 | 0.02387 |
| q4_k | 728.8 MiB | 8.61s | 7.277s | 0.319s | 5.72519 | 0.28411 | 1.22644 | 0.07608 |

These are single-run regression numbers, not a perceptual MOS. The generated mel
and decoded WAV are compared against the f32 output from the same CLI command.
Use f16/q8_0/q6_k as the practical quality candidates; treat q4_k as
size-first until more listening tests are collected.
