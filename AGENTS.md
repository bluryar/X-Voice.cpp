# AGENTS.md

## Mission

`x-voice-ggml-cpp` is the standalone C++/GGML runtime target for the
`x-voice-f32.gguf` bundle staged by `projects/x-voice-ggml-py`.

The project should preserve the Python staging project's graph, tensor,
metadata, tokenizer, and audio-boundary decisions while removing Python, Torch,
and ggbond binding dependencies from inference.

## Sources Of Truth

Prefer these in order:

1. `/root/code/ggbond/models/x-voice-f32.gguf` and its GGUF metadata/tensors.
2. `projects/x-voice-ggml-py/docs/specs/x-voice-cpp-handoff.md`.
3. `projects/x-voice-ggml-py/x_voice_ggml_py/graphs/*.py`.
4. Upstream X-Voice source under `projects/x-voice-ggml-py/third_party/X-Voice`.

## Current Boundary

- C++ v0 must support `--text-kind ipa` and `--text-kind tokens`.
- Raw plain-text Chinese/English frontends are v1 work and must not block the
  Stage-2/SRP/Vocos runtime port.
- Stage-2/SRP/Vocos graph builders must read GGUF metadata instead of
  hardcoding released checkpoint sizes.
- Tensor shapes must be documented and validated in GGML order
  `(ne0, ne1, ne2, ne3)`.
- Keep Vocos ISTFT as project-local C++ audio reconstruction unless a later ADR
  explicitly moves it into GGML.

## GGML Alignment

- Prefer upstream GGML C names: `ggml_mul_mat`, `ggml_get_rows`,
  `ggml_soft_max`, `ggml_rope`, etc.
- Preserve operand order from the Python GGML staging graph.
- Do not add Torch-style tensor helpers or implicit dimension reversal.
- If source-order host arrays are used, label them as source-order views and
  state their GGML `ne` crossing contract.

## Verification

Initial local smoke:

```bash
cmake -S projects/x-voice-ggml-cpp -B projects/x-voice-ggml-cpp/build \
  -DXVOICE_ALLOW_WORKBENCH_GGML_FALLBACK=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build projects/x-voice-ggml-cpp/build -j8
projects/x-voice-ggml-cpp/build/x-voice-cli \
  --model /root/code/ggbond/models/x-voice-f32.gguf \
  --inspect \
  --validate-tensors
```

For strict graph parity, compare against the Python fixtures listed in the
handoff spec before claiming a graph stage is ported.
