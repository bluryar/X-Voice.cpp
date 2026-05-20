# X-Voice GGML C++ Release Checklist

This project should be releasable as a standalone repository and then mounted in
`ggbond` as `projects/x-voice-ggml-cpp` through a git submodule.

## Repository Contents

Keep:

- `CMakeLists.txt`
- `include/`
- `src/`
- `resources/`
- `scripts/`
- `docs/`
- `tools/`
- `vendor/` source trees or submodule pointers

Do not keep:

- `build/`, `build-*`
- benchmark output directories
- generated WAV/metadata artifacts
- local model weights such as `x-voice-f32.gguf`

## License Gate

Runtime source is licensed under Apache-2.0, following the sibling GGML C++
projects in this workbench. Keep third-party license files under `vendor/` and
document the model-weight license separately from the runtime source license.
This runtime repository should not imply redistribution rights for
`x-voice-f32.gguf`.

## Required External Artifact

The runtime expects a GGUF bundle such as:

```text
/root/code/ggbond/models/x-voice-f32.gguf
```

The model file is intentionally not part of the C++ source repository. Users pass
it with `--model`.

Recommended model artifact names for the companion Hugging Face repository:

```text
x-voice-f32.gguf
x-voice-f16.gguf
x-voice-q8_0.gguf
x-voice-q6_k.gguf
x-voice-q4_k.gguf
```

Generate the quantized variants with:

```bash
MODEL=/path/to/x-voice-f32.gguf OUT_DIR=/path/to/out scripts/dev/quantize_xvoice_gguf.sh
```

## Product Smoke

Build CUDA:

```bash
bash scripts/dev/build_xvoice_cuda.sh
```

Run a product synthesis regression:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-project --with numpy python \
  scripts/dev/check_xvoice_synthesis_regression.py \
  --model /path/to/x-voice-f32.gguf \
  --ref-wav /path/to/reference.wav
```

Expected contracts:

- output sample rate: `24000`
- sampler mode under `--preset product`: `cfg_nonlayered`
- automatic speed is clamped into `8..12` unless an explicit speed is passed
- generated/decode frames are positive
- WAV RMS/peak/clip and optional `hf>8k` stay within the script bounds

Convenience CUDA synthesis wrapper:

```bash
bash scripts/dev/synthesize_product_cuda.sh
```

It uses the same `--preset product` defaults and accepts environment overrides:
`XVOICE_MODEL`, `XVOICE_REF_WAV`, `XVOICE_TEXT`, `XVOICE_OUTPUT_WAV`,
`XVOICE_METADATA_JSON`, and `XVOICE_THREADS`.

## Release Preflight

Workbench mode:

```bash
scripts/dev/check_release_tree.py
```

Standalone publish mode:

```bash
scripts/dev/check_release_tree.py --standalone --strict-clean
```

Workbench mode permits the parent `ggbond/vendor/ggml` fallback. Standalone mode
requires `vendor/ggml/CMakeLists.txt` and should be used before pushing the
extracted repository.

## Submodule Use From ggbond

After publishing this directory as its own repository:

```bash
git submodule add <x-voice-ggml-cpp-repo-url> projects/x-voice-ggml-cpp
git submodule update --init --recursive projects/x-voice-ggml-cpp
```

For local development inside `ggbond`, CMake can use the parent workbench GGML:

```bash
cmake -S projects/x-voice-ggml-cpp -B projects/x-voice-ggml-cpp/build-cuda \
  -DXVOICE_ALLOW_WORKBENCH_GGML_FALLBACK=ON \
  -DGGML_CUDA=ON
```

Standalone releases should keep `vendor/ggml` available instead of relying on
that fallback.

When extracting this directory into its own repository, copy
`.gitmodules.standalone` to `.gitmodules` before committing submodule pointers:

```bash
cp .gitmodules.standalone .gitmodules
git submodule update --init --recursive
```
