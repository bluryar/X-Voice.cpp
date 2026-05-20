# X-Voice GGML C++

Standalone C++/GGML runtime target for the X-Voice Stage-2 GGUF bundle staged
by `projects/x-voice-ggml-py`.

The default model artifact is:

```text
/root/code/ggbond/models/x-voice-f32.gguf
```

## Status

This project is past the first C++ graph parity stage and now has a
source-order reference-mel synthesis smoke path.

Implemented:

- CMake project boundary for a future standalone repository.
- GGUF metadata parsing for Stage-2, SRP, Vocos, sampler, audio, language, and
  tokenizer contracts.
- GGUF tensor header indexing without loading the 1.6 GiB payload by default.
- Optional GGUF tensor payload loading into a GGML backend buffer:
  `--load-tensors -b cpu`.
- Representative tensor `ne` validation in GGML order.
- X-Voice IPA v6/token input frontend:
  - `--text-kind ipa`
  - `--text-kind tokens`
  - raw vocab ids before the upstream Stage-2 `+1` offset
- CLI `--inspect` and `--print-tokens` smoke paths.
- SRP input embedding and 16-block logits GGML graph parity against the Python
  Torch reference fixture.
- Stage-2 time embedding, text embedding, input embedding, selected DiT anchor
  blocks, 22-block DiT stack, output projection, and combined DiT forward GGML
  graph parity against the Python Torch reference fixture.
- Stage-2 no-CFG, non-layered CFG, and layered CFG 32-step sampler scheduling
  parity against the Python host trajectory fixture.
- Vocos neural GGML graph plus project-local ISTFT waveform reconstruction.
- CLI WAV writing from raw Vocos mel fixtures.
- CLI `--synthesize-ref-mel` path:
  IPA/tokens text plus source-order reference mel -> SRP speed policy ->
  Stage-2 host prep/sampler -> generated mel slice -> Vocos decode -> WAV.
- CLI `--synthesize-ref-wav` path with project-local C++ Vocos log-mel
  frontend.
- Product-facing `--synthesize` alias for the current reference-WAV path.
- Optional `--metadata-json` sidecar for synthesis runs.
- tqdm-style CLI progress reporting for synthesis. It is enabled automatically
  on interactive stderr, can be forced with `--progress`, and can be disabled
  with `--no-progress`.
- One-command v0 smoke and WAV frontend fixture scripts under `scripts/dev`.
- Chinese raw-text frontend for `--text-kind plain --language zh`, enabled
  automatically when vendored `cpp-pinyin` and `cppjieba` dependencies are
  available. It uses the generated merged cpp-pinyin dictionary under
  `resources/zh_pinyin_dict` plus an X-Voice override TSV for narration words
  and common polyphones.

Current GGUF/tokenizer state:

- `/root/code/ggbond/models/x-voice-f32.gguf` is self-contained in the current
  workspace and reports `tokenizer_tokens: 820`.
- The C++ CLI still supports `--vocab` as a transitional fallback for older
  pre-bootstrap GGUF files.
- `scripts/dev/refresh_self_contained_gguf.sh` checks this contract and can
  force a full re-export if needed.

English raw-text frontend and full 30-language phonemization remain out of the
default runtime boundary.

## Build

From `/root/code/ggbond`:

```bash
cmake -S projects/x-voice-ggml-cpp -B projects/x-voice-ggml-cpp/build \
  -DXVOICE_ALLOW_WORKBENCH_GGML_FALLBACK=ON \
  -DXVOICE_ENABLE_ZH_FRONTEND=AUTO \
  -DCMAKE_BUILD_TYPE=Release
cmake --build projects/x-voice-ggml-cpp/build -j8
```

CUDA build, using the current NVIDIA GPU architecture by default:

```bash
bash projects/x-voice-ggml-cpp/scripts/dev/build_xvoice_cuda.sh
```

The CUDA helper writes `projects/x-voice-ggml-cpp/build-cuda/x-voice-cli`.
Override the detected architecture or build dir with:

```bash
XVOICE_CUDA_ARCHITECTURES=89 \
XVOICE_CPP_CUDA_BUILD_DIR=projects/x-voice-ggml-cpp/build-cuda \
bash projects/x-voice-ggml-cpp/scripts/dev/build_xvoice_cuda.sh
```

When this project is split into a standalone repository, vendor `ggml` into
`vendor/ggml` and build without the workbench fallback.

Chinese raw text is controlled by:

```bash
-DXVOICE_ENABLE_ZH_FRONTEND=AUTO  # default intent for this workspace
```

Use `ON` to require the dependencies and fail if they are missing, or `OFF` to
force the IPA/tokens-only build. The default dictionary paths are:

```text
cpp-pinyin: projects/x-voice-ggml-cpp/resources/zh_pinyin_dict
cppjieba:   projects/x-voice-ggml-cpp/vendor/cppjieba/dict
override:   projects/x-voice-ggml-cpp/resources/zh_frontend_overrides.tsv
```

The generated pinyin dictionary is now aligned to python-pinyin:

```text
source:     projects/x-voice-ggml-cpp/vendor/python-pinyin/pypinyin/{phrases_dict,pinyin_dict}.json
runtime:    projects/x-voice-ggml-cpp/resources/zh_pinyin_dict
support:    projects/x-voice-ggml-cpp/vendor/cpp-pinyin/res/dict/mandarin/trans_word.txt
```

`vendor/python-pinyin` is kept as a git submodule for debugging and offline
resource generation only. It is not linked by CMake, and the runtime frontend
loads only the generated cpp-pinyin dictionary under `resources/zh_pinyin_dict`.
Compare python-pinyin's own loaded `PHRASES_DICT`/`PINYIN_DICT` view against the
generated runtime dictionary with:

```bash
uv run --no-project python \
  projects/x-voice-ggml-cpp/scripts/dev/compare_python_pinyin_cpp_dict.py
```

The audit writes under the ignored build tree:

```text
projects/x-voice-ggml-cpp/build/python_pinyin_cpp_alignment/summary.json
projects/x-voice-ggml-cpp/build/python_pinyin_cpp_alignment/*_conflicts.jsonl
projects/x-voice-ggml-cpp/build/python_pinyin_cpp_alignment/*_only_*.jsonl
```

Current generated-dictionary audit result: phrase conflicts `0`, word conflicts
`0`. `word_only_python=15148` is expected because cpp-pinyin's single-character
dictionary is keyed by `char16_t`; non-BMP python-pinyin codepoints are skipped
instead of being written incorrectly.

At runtime, `--pinyin-dict DIR`, `--jieba-dict-dir DIR`, and `--zh-override TSV`
can point the frontend at a stronger dictionary or project-specific override
file.

Rebuild the merged dictionary offline with:

```bash
projects/x-voice-ggml-cpp/build/xvoice-merge-pinyin-dicts \
  --python-pinyin projects/x-voice-ggml-cpp/vendor/python-pinyin \
  --cpp-pinyin-dict projects/x-voice-ggml-cpp/vendor/cpp-pinyin/res/dict \
  --out projects/x-voice-ggml-cpp/resources/zh_pinyin_dict
```

## Smoke

One-command v0 smoke:

```bash
bash projects/x-voice-ggml-cpp/scripts/dev/check_xvoice_cpp_v0_smoke.sh
```

This builds if needed, validates tensor headers, checks self-contained tokenizer
metadata, prints IPA tokens, and runs a short reference-WAV synthesis smoke.

Inspect the default GGUF metadata and representative tensor shapes:

```bash
projects/x-voice-ggml-cpp/build/x-voice-cli \
  --model /root/code/ggbond/models/x-voice-f32.gguf \
  --inspect \
  --validate-tensors
```

Check the C++ IPA frontend boundary:

```bash
projects/x-voice-ggml-cpp/build/x-voice-cli \
  --model /root/code/ggbond/models/x-voice-f32.gguf \
  --vocab /root/code/ggbond/models/X-Voice/XVoice_Base_Stage1/vocab.txt \
  --text 'h|ə|l|oʊ' \
  --text-kind ipa \
  --language en \
  --print-tokens
```

Use IPA/pinyin-style X-Voice input for the default v0 path:

```bash
projects/x-voice-ggml-cpp/build/x-voice-cli \
  --model /root/code/ggbond/models/x-voice-f32.gguf \
  --text 'fa1|tie1|ren2 di4|di' \
  --text-kind ipa \
  --language zh \
  --print-tokens
```

Use direct Chinese text when the zh frontend is enabled:

```bash
projects/x-voice-ggml-cpp/build/x-voice-cli \
  --model /root/code/ggbond/models/x-voice-f32.gguf \
  --text '发帖人（弟弟）在详细描述其五口之家的现状后，寻求处理家庭问题的建议，并提出了自己的初步计划。' \
  --text-kind plain \
  --language zh \
  --print-tokens
```

Product-facing reference-WAV synthesis entry:

```bash
projects/x-voice-ggml-cpp/build/x-voice-cli \
  --model /root/code/ggbond/models/x-voice-f32.gguf \
  --load-tensors \
  --synthesize \
  --text 'fa1|tie1|ren2 di4|di' \
  --text-kind ipa \
  --language zh \
  --ref-wav /root/code/ggbond/models/test.wav \
  --ref-max-frames 64 \
  --output-wav /tmp/xvoice-cpp.wav \
  --metadata-json /tmp/xvoice-cpp.json \
  --progress \
  -b cpu -t 4
```

Direct Chinese text can use the same synthesis path by changing
`--text-kind plain` and passing the raw text.

CUDA synthesis with the product preset:

```bash
NVIDIA_TF32_OVERRIDE=0 projects/x-voice-ggml-cpp/build-cuda/x-voice-cli \
  --model /root/code/ggbond/models/x-voice-f32.gguf \
  --load-tensors \
  --synthesize \
  --text '发帖人（弟弟）在详细描述其五口之家的现状后，寻求处理家庭问题的建议，并提出了自己的初步计划。' \
  --text-kind plain \
  --language zh \
  --ref-wav /root/code/ggbond/models/test.wav \
  --preset product \
  --output-wav /tmp/xvoice-cpp-zh-cuda.wav \
  --metadata-json /tmp/xvoice-cpp-zh-cuda.json \
  --progress \
  -b cuda -t 8
```

`--preset product` expands to the current open-source v0 defaults:

- `--sampler-mode cfg_nonlayered`
- `--step-count 32`
- `--ref-auto-trim --ref-max-frames 384`
- automatic SRP speed clamped to `--auto-speed-min 8 --auto-speed-max 12`

For the sample sentence above, raw SRP predicts `speed_value=4.5`, which would
stretch the utterance and can sound word-by-word. The product preset records
both `raw_speed_value` and final `speed_value` in metadata; local CUDA regression
currently clamps it to `8.0`, yielding about `10.25s`. You can still pass an
explicit `--speed-value 10.0` when you want tighter narration pacing.

The GGUF default sampler grid is `32` steps. The CLI now treats
`--step-count` as a runtime sampler-grid size, so values above `32` are valid;
they are slower and are not guaranteed to sound better, but they are useful for
artifact checks.

For reference WAVs with leading silence, avoid taking only the first tiny prompt
slice. The local `test.wav` begins with roughly 0.35 seconds of near-silence, so
`--ref-max-frames 128` can produce metallic/high-frequency artifacts. Prefer a
longer reference window and let the CLI pick the first voiced frame with
`--ref-auto-trim`. Explicit `--ref-start-frame` still wins when you want a
manual crop.

Synthesis benchmark sweep:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-project --with numpy python \
  projects/x-voice-ggml-cpp/scripts/dev/bench_xvoice_synthesis.py \
  --output-dir /tmp/xvoice-bench
```

The harness runs the CLI over sampler mode, step count, and reference-window
matrices, then writes `bench_xvoice_synthesis.jsonl` plus
`bench_xvoice_synthesis_summary.json`. Each metadata sidecar includes phase
timings such as `load tensors`, `reference mel`, `text frontend`,
`stage2 sampler`, `vocos decode`, and optional `stage2 branch/full|null|text`
conditioned-DiT aggregates when `--profile-stage2-branches` is enabled.

End-to-end CUDA product regression:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-project --with numpy python \
  projects/x-voice-ggml-cpp/scripts/dev/check_xvoice_synthesis_regression.py
```

This is not a waveform-exact parity check. It runs the public CLI with
`--preset product` and validates metadata plus WAV properties: sampler mode,
speed policy, text unit count, generated duration, sample rate, RMS/peak/clip,
and optional `hf>8k` energy. The default output goes to
`/tmp/xvoice-cpp-regression`.

Convenience product synthesis wrapper:

```bash
bash projects/x-voice-ggml-cpp/scripts/dev/synthesize_product_cuda.sh
```

Override inputs with environment variables such as `XVOICE_TEXT`,
`XVOICE_REF_WAV`, `XVOICE_OUTPUT_WAV`, `XVOICE_METADATA_JSON`, and
`XVOICE_MODEL`.

The Stage-2 sampler reuses local time/input/forward GGML graph runners during a
single synthesis call. CFG branch execution now uses a fused conditioned DiT
runner for `stage2.inp.* -> 22x DiT -> stage2.out`, so the intermediate hidden
input embedding stays inside the GGML graph instead of round-tripping through
host memory. Tensor values remain unchanged in the parity fixtures.

Experimental CFG batch-step execution can be enabled with
`--stage2-batch-dit-forward`. It packs CFG branches into GGML `ne2`, keeps
`stage2.inp.* -> 22x DiT -> stage2.out`, CFG combine, and Euler update inside
one graph step, and returns only the next sampled mel. It is kept off by default
because the current conditioned single-branch path is still faster on the local
CUDA backend.

WAV frontend fixture coverage:

```bash
uv run --no-project --with numpy python \
  projects/x-voice-ggml-cpp/scripts/dev/check_wav_frontend_fixtures.py
```

Self-contained GGUF check/re-export helper:

```bash
bash projects/x-voice-ggml-cpp/scripts/dev/refresh_self_contained_gguf.sh
```

Use `--force-write` only when you intentionally want to rebuild the 1.6 GiB
GGUF from the Python exporter.

Quantize the self-contained GGUF into f16/q8/q6/q4 variants:

```bash
MODEL=/root/code/ggbond/models/x-voice-f32.gguf \
OUT_DIR=/root/code/ggbond/models \
  projects/x-voice-ggml-cpp/scripts/dev/quantize_xvoice_gguf.sh
```

The default quantization policy is conservative: large GGML `mul_mat` matrix
weights are converted, while conv, norm, bias, embedding, positional, and small
tensors remain in their original type. See `docs/quantization.md` for manual
commands and validation notes.

Local quantization benchmark, measured on an RTX 4060 Ti with
`NVIDIA_TF32_OVERRIDE=0`, `--preset product`, `cfg_nonlayered`, `32` steps, the
default zh sample text, and `/root/code/ggbond/models/test.wav`:

| model | size | wall | sampler | load | mel max abs vs f32 | mel mean abs vs f32 | wav max abs vs f32 | wav mean abs vs f32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| f32 | 1544.1 MiB | 10.74s | 9.222s | 0.515s | 0 | 0 | 0 | 0 |
| f16 | 1060.5 MiB | 8.89s | 7.316s | 0.513s | 1.40578 | 0.01232 | 0.89212 | 0.01086 |
| q8_0 | 847.4 MiB | 8.52s | 7.268s | 0.303s | 0.78026 | 0.02397 | 1.01920 | 0.01708 |
| q6_k | 790.0 MiB | 8.66s | 7.407s | 0.301s | 0.89031 | 0.04256 | 0.68326 | 0.02387 |
| q4_k | 728.8 MiB | 8.61s | 7.277s | 0.319s | 5.72519 | 0.28411 | 1.22644 | 0.07608 |

Interpretation: f16/q8_0/q6_k are the preferred release candidates. q4_k is
useful when size matters most, but its larger generated-mel drift should be
auditioned carefully before treating it as a quality default.

## Submodule Packaging

This directory is intended to become a standalone open-source repository and to
be consumed by `ggbond` as a git submodule. The extraction boundary is:

- keep `include/`, `src/`, `resources/`, `scripts/`, `docs/`, `tools/`,
  `CMakeLists.txt`, and vendored third-party source/submodules;
- do not commit build directories, benchmark outputs, local WAVs, or model
  weights;
- keep `x-voice-f32.gguf` external and pass it through `--model`;
- after publishing, consume it from the parent workbench with:

```bash
git submodule add <x-voice-ggml-cpp-repo-url> projects/x-voice-ggml-cpp
git submodule update --init --recursive projects/x-voice-ggml-cpp
```

The local CMake fallback `-DXVOICE_ALLOW_WORKBENCH_GGML_FALLBACK=ON` is for
development inside `ggbond`; a standalone release should vendor or submodule
GGML under `vendor/ggml`.

For an extracted standalone repository, copy `.gitmodules.standalone` to
`.gitmodules` before committing the nested submodule pointers.

Run the release-tree preflight in the current workbench:

```bash
projects/x-voice-ggml-cpp/scripts/dev/check_release_tree.py
```

Before publishing the standalone repository, run:

```bash
projects/x-voice-ggml-cpp/scripts/dev/check_release_tree.py --standalone --strict-clean
```

At the moment, `--standalone --strict-clean` intentionally fails while local
`build*` directories exist; remove generated directories before publishing.

## License

The runtime source is licensed under Apache-2.0. Model weights such as
`x-voice-f32.gguf` are external artifacts and must follow their own upstream
license and redistribution terms.

Export SRP raw fixtures for C++ parity:

```bash
uv run --no-project --with numpy python \
  projects/x-voice-ggml-cpp/scripts/dev/export_stage2_srp_fixture_raw.py
```

Run SRP input/logits parity using the real `x-voice-f32.gguf` tensor payload:

```bash
projects/x-voice-ggml-cpp/build/x-voice-cli \
  --model /root/code/ggbond/models/x-voice-f32.gguf \
  --load-tensors \
  --check-stage2-time \
  --stage2-time-hidden /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-srp/stage2_time_hidden.f32 \
  --stage2-time-language-ids /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-srp/stage2_time_language_ids.i32 \
  --stage2-time-ref /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-srp/stage2_time_embed.f32 \
  --threshold 1e-4 \
  -b cpu -t 4

projects/x-voice-ggml-cpp/build/x-voice-cli \
  --model /root/code/ggbond/models/x-voice-f32.gguf \
  --load-tensors \
  --check-stage2-text \
  --seq-len 16 \
  --stage2-text-ids /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-srp/stage2_text_ids_plus_one.i32 \
  --stage2-text-language-ids /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-srp/stage2_text_language_ids.i32 \
  --stage2-text-no-lang-mask /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-srp/stage2_text_no_lang_fusion_mask.f32 \
  --stage2-text-keep-mask /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-srp/stage2_text_keep_mask.f32 \
  --stage2-text-pos-embed /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-srp/stage2_text_pos_embed.f32 \
  --stage2-text-ref /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-srp/stage2_text_embed.f32 \
  --threshold 1e-4 \
  -b cpu -t 4

projects/x-voice-ggml-cpp/build/x-voice-cli \
  --model /root/code/ggbond/models/x-voice-f32.gguf \
  --load-tensors \
  --check-stage2-input \
  --seq-len 16 \
  --stage2-x /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-srp/stage2_x.f32 \
  --stage2-cond /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-srp/stage2_cond.f32 \
  --stage2-input-text-embed /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-srp/stage2_input_text_embed.f32 \
  --stage2-input-ref /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-srp/stage2_input_embed.f32 \
  --threshold 1e-4 \
  -b cpu -t 4

projects/x-voice-ggml-cpp/build/x-voice-cli \
  --model /root/code/ggbond/models/x-voice-f32.gguf \
  --load-tensors \
  --check-stage2-block \
  --seq-len 16 \
  --block-index 21 \
  --stage2-block-input /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-srp/stage2_block21_input.f32 \
  --stage2-block-time-embed /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-srp/stage2_time_embed.f32 \
  --stage2-block-ref /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-srp/stage2_block21.f32 \
  --threshold 3e-4 \
  -b cpu -t 4

projects/x-voice-ggml-cpp/build/x-voice-cli \
  --model /root/code/ggbond/models/x-voice-f32.gguf \
  --load-tensors \
  --check-stage2-stack \
  --seq-len 16 \
  --stack-blocks 22 \
  --stage2-stack-input /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-srp/stage2_input_embed.f32 \
  --stage2-stack-time-embed /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-srp/stage2_time_embed.f32 \
  --stage2-stack-ref /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-srp/stage2_stack22.f32 \
  --threshold 5e-4 \
  -b cpu -t 4

projects/x-voice-ggml-cpp/build/x-voice-cli \
  --model /root/code/ggbond/models/x-voice-f32.gguf \
  --load-tensors \
  --check-stage2-output \
  --seq-len 16 \
  --stage2-output-input /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-srp/stage2_stack22.f32 \
  --stage2-output-time-embed /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-srp/stage2_time_embed.f32 \
  --stage2-output-ref /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-srp/stage2_output.f32 \
  --threshold 5e-4 \
  -b cpu -t 4

projects/x-voice-ggml-cpp/build/x-voice-cli \
  --model /root/code/ggbond/models/x-voice-f32.gguf \
  --load-tensors \
  --check-stage2-forward \
  --seq-len 16 \
  --stack-blocks 22 \
  --stage2-forward-input /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-srp/stage2_input_embed.f32 \
  --stage2-forward-time-embed /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-srp/stage2_time_embed.f32 \
  --stage2-forward-ref /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-srp/stage2_output.f32 \
  --threshold 5e-4 \
  -b cpu -t 4

projects/x-voice-ggml-cpp/build/x-voice-cli \
  --model /root/code/ggbond/models/x-voice-f32.gguf \
  --load-tensors \
  --check-srp-input \
  --seq-len 16 \
  --srp-mel /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-srp/srp_mel.f32 \
  --srp-input-ref /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-srp/srp_input_embed.f32 \
  --threshold 1e-4 \
  -b cpu -t 4

projects/x-voice-ggml-cpp/build/x-voice-cli \
  --model /root/code/ggbond/models/x-voice-f32.gguf \
  --load-tensors \
  --check-srp-logits \
  --seq-len 16 \
  --srp-mel /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-srp/srp_mel.f32 \
  --srp-logits-ref /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-srp/srp_logits.f32 \
  --threshold 1e-4 \
  -b cpu -t 4
```

Export Stage-2 host sampler raw fixtures and run full no-CFG sampler parity:

```bash
uv run --no-project --with numpy python \
  projects/x-voice-ggml-cpp/scripts/dev/export_stage2_host_fixture_raw.py

projects/x-voice-ggml-cpp/build/x-voice-cli \
  --model /root/code/ggbond/models/x-voice-f32.gguf \
  --load-tensors \
  --check-stage2-no-cfg-sampler \
  --seq-len 251 \
  --step-count 32 \
  --stage2-sampler-text-ids /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-host/stage2_text_ids_plus_one.i32 \
  --stage2-sampler-text-language-ids /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-host/stage2_text_language_ids.i32 \
  --stage2-sampler-text-no-lang-mask /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-host/stage2_text_no_lang_fusion_mask.f32 \
  --stage2-sampler-text-keep-mask /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-host/stage2_text_keep_mask.f32 \
  --stage2-sampler-text-pos-embed /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-host/stage2_text_pos_embed.f32 \
  --stage2-sampler-time-language-ids /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-host/stage2_time_language_ids.i32 \
  --stage2-sampler-cond /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-host/stage2_cond.f32 \
  --stage2-sampler-cond-mask /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-host/stage2_cond_mask.f32 \
  --stage2-sampler-fixed-noise /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-host/stage2_fixed_noise.f32 \
  --stage2-sampler-timesteps /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-host/stage2_sampler_timesteps.f32 \
  --stage2-sampler-sampled-ref /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-host/stage2_host_sampler_y32.f32 \
  --stage2-sampler-out-ref /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-host/stage2_host_sampler_out.f32 \
  --threshold 2e-3 \
  -b cpu -t 4
```

Swap the check/ref paths to validate CFG modes:

```bash
projects/x-voice-ggml-cpp/build/x-voice-cli \
  --model /root/code/ggbond/models/x-voice-f32.gguf \
  --load-tensors \
  --check-stage2-nonlayered-cfg-sampler \
  --seq-len 251 \
  --step-count 32 \
  --stage2-sampler-text-ids /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-host/stage2_text_ids_plus_one.i32 \
  --stage2-sampler-text-language-ids /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-host/stage2_text_language_ids.i32 \
  --stage2-sampler-text-no-lang-mask /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-host/stage2_text_no_lang_fusion_mask.f32 \
  --stage2-sampler-text-keep-mask /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-host/stage2_text_keep_mask.f32 \
  --stage2-sampler-text-pos-embed /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-host/stage2_text_pos_embed.f32 \
  --stage2-sampler-time-language-ids /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-host/stage2_time_language_ids.i32 \
  --stage2-sampler-cond /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-host/stage2_cond.f32 \
  --stage2-sampler-cond-mask /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-host/stage2_cond_mask.f32 \
  --stage2-sampler-fixed-noise /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-host/stage2_fixed_noise.f32 \
  --stage2-sampler-timesteps /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-host/stage2_sampler_timesteps.f32 \
  --stage2-sampler-sampled-ref /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-host/stage2_host_cfg_nonlayered_y32.f32 \
  --stage2-sampler-out-ref /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-host/stage2_host_cfg_nonlayered_out.f32 \
  --threshold 2e-3 \
  -b cpu -t 4

projects/x-voice-ggml-cpp/build/x-voice-cli \
  --model /root/code/ggbond/models/x-voice-f32.gguf \
  --load-tensors \
  --check-stage2-layered-cfg-sampler \
  --seq-len 251 \
  --step-count 32 \
  --stage2-sampler-text-ids /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-host/stage2_text_ids_plus_one.i32 \
  --stage2-sampler-text-language-ids /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-host/stage2_text_language_ids.i32 \
  --stage2-sampler-text-no-lang-mask /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-host/stage2_text_no_lang_fusion_mask.f32 \
  --stage2-sampler-text-keep-mask /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-host/stage2_text_keep_mask.f32 \
  --stage2-sampler-text-pos-embed /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-host/stage2_text_pos_embed.f32 \
  --stage2-sampler-time-language-ids /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-host/stage2_time_language_ids.i32 \
  --stage2-sampler-cond /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-host/stage2_cond.f32 \
  --stage2-sampler-cond-mask /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-host/stage2_cond_mask.f32 \
  --stage2-sampler-fixed-noise /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-host/stage2_fixed_noise.f32 \
  --stage2-sampler-timesteps /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-host/stage2_sampler_timesteps.f32 \
  --stage2-sampler-sampled-ref /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-host/stage2_host_cfg_layered_y32.f32 \
  --stage2-sampler-out-ref /root/code/ggbond/models/x-voice-cpp-fixtures/stage2-host/stage2_host_cfg_layered_out.f32 \
  --threshold 2e-3 \
  -b cpu -t 4
```

Run the current ref-mel synthesis smoke. This skips WAV-to-mel frontend work by
feeding a source-order reference mel fixture directly:

```bash
projects/x-voice-ggml-cpp/build/x-voice-cli \
  --model /root/code/ggbond/models/x-voice-f32.gguf \
  --load-tensors \
  --synthesize-ref-mel \
  --text 'fa1|tie1|ren2 di4|di' \
  --text-kind ipa \
  --language zh \
  --ref-mel projects/x-voice-ggml-cpp/build/fixtures/vocos/vocos_mel.f32 \
  --ref-mel-frames 8 \
  --sampler-mode no_cfg \
  --step-count 1 \
  --speed-value 6.75 \
  --output-wav /tmp/xvoice-cpp-refmel-smoke.wav \
  -b cpu -t 4
```

Run the current IPA/tokens + reference WAV synthesis smoke:

```bash
projects/x-voice-ggml-cpp/build/x-voice-cli \
  --model /root/code/ggbond/models/x-voice-f32.gguf \
  --load-tensors \
  --synthesize-ref-wav \
  --text 'fa1|tie1|ren2 di4|di' \
  --text-kind ipa \
  --language zh \
  --ref-wav /tmp/xvoice-vocos-fixture.wav \
  --sampler-mode no_cfg \
  --step-count 1 \
  --speed-value 6.75 \
  --output-wav /tmp/xvoice-cpp-refwav-smoke.wav \
  -b cpu -t 4
```

For longer reference files, keep smoke tests tractable by trimming the reference
mel after the real WAV frontend has run:

```bash
/usr/bin/time -f 'elapsed=%E maxrss_kb=%M' \
projects/x-voice-ggml-cpp/build/x-voice-cli \
  --model /root/code/ggbond/models/x-voice-f32.gguf \
  --load-tensors \
  --synthesize-ref-wav \
  --text 'fa1|tie1|ren2 di4|di' \
  --text-kind ipa \
  --language zh \
  --ref-wav /root/code/ggbond/models/test.wav \
  --ref-max-frames 64 \
  --sampler-mode default \
  --speed-value 6.75 \
  --output-wav /tmp/xvoice-cpp-testwav-default-full-ref64.wav \
  -b cpu -t 4
```

Observed on the current CPU smoke box: `test.wav` has `797` source prompt
frames at 32 kHz. With `--ref-max-frames 64`, default non-layered CFG and
`nfe_step=32` finish in about `49s` with `~1.6GB` max RSS. The uncropped
`797`-frame prompt with default CFG and `--step-count 2` takes about `18s`,
so full uncropped CPU synthesis is expected to be minutes rather than seconds.

## Planned Runtime Pipeline

```text
GGUF load
  -> IPA/tokens frontend
  -> reference WAV to Vocos log-mel (C++ project-local DSP complete)
  -> SRP logits and speed policy
  -> Stage-2 drop-text host prep
  -> Stage-2 32-step sampler with CFG
  -> Vocos neural backbone/head in GGML (C++ parity complete)
  -> project-local ISTFT (C++ parity complete)
  -> WAV (IPA/tokens + ref-wav CLI smoke complete)
```

See `docs/specs/x-voice-cpp-v0.md` for the current C++ contract and
`../x-voice-ggml-py/docs/specs/x-voice-cpp-handoff.md` for the parity anchors.
