# X-Voice C++ v0 Contract

## Goal

Create the smallest useful C++ boundary for `x-voice-f32.gguf` before graph
execution is fully ported. SRP is now the first executed graph in this boundary.

## Supported Inputs

Required:

- `--text-kind ipa`
- `--text-kind tokens`
- `--language <code>`

Optional v1 frontend staging:

- `--text-kind plain --language zh` when built with
  `-DXVOICE_ENABLE_ZH_FRONTEND=AUTO` and local dictionaries are found, or with
  `-DXVOICE_ENABLE_ZH_FRONTEND=ON`
- pipeline:
  `longest X-Voice override match -> cppjieba -> cpp-pinyin TONE3 -> basic tone sandhi/override -> ipa_v6 tokens`
- default dictionaries:
  - generated cpp-pinyin root `resources/zh_pinyin_dict`
  - vendored cppjieba `vendor/cppjieba/dict`
  - `resources/zh_frontend_overrides.tsv`

Not supported in the default v0 build:

- non-Chinese `--text-kind plain`
- automatic language detection
- English G2P
- full 30-language phonemization

## Frontend Contract

The Stage-2 graph consumes raw X-Voice vocab ids before the upstream `+1`
offset. The C++ tokenizer mirrors Python `text_frontend.py`:

```text
IPA string
  -> split on spaces into words
  -> split each word on "|"
  -> group [A-Za-z U+00C0..U+02AF] runs
  -> emit all other UTF-8 codepoints as individual tokens
  -> insert " " between non-final words
  -> map tokens through tokenizer.ggml.tokens
  -> fallback unknown multi-codepoint tokens to UTF-8 codepoints
```

Example:

```text
fa1|tie1|ren2 di4|di
-> fa 1 | tie 1 | ren 2 | " " | di 4 | di
-> raw vocab ids
```

The `+1` offset, prefix token id, anchor ids, language-id masks, and padding
belong to Stage-2 host prep, not the frontend.

The optional Chinese plain-text frontend is deliberately a project-local
preprocessor that produces the same IPA-v6 string boundary above. It is not a
GGML graph concern and does not change Stage-2 tensor contracts.

Runtime flags can replace the frontend resources without changing the model
graph:

```text
--pinyin-dict DIR
--jieba-dict-dir DIR
--zh-override TSV
```

The generated cpp-pinyin root is produced by
`xvoice-merge-pinyin-dicts`. It converts vendored python-pinyin
`phrases_dict.json` and `pinyin_dict.json` into cpp-pinyin loadable files:

```text
phrases_dict.json -> mandarin/phrases_dict.txt
pinyin_dict.json  -> mandarin/word.txt
derived chars     -> mandarin/phrases_map.txt
```

The generated `mandarin/user_dict.txt` is intentionally empty except for a
comment. Project-specific narration/polyphone fixes live in
`resources/zh_frontend_overrides.tsv`, not inside the generated python-pinyin
runtime dictionary.

`vendor/python-pinyin` is a submodule used for debug/resource generation only.
It is not a runtime or CMake dependency. The comparison script imports
python-pinyin's real `PHRASES_DICT`/`PINYIN_DICT` after its JSON loaders run,
then compares that loaded view against the generated runtime dictionary:

```bash
uv run --no-project python \
  projects/x-voice-ggml-cpp/scripts/dev/compare_python_pinyin_cpp_dict.py
```

Audit output lives under the ignored build directory
`build/python_pinyin_cpp_alignment`. Current generated-dictionary findings:

```text
phrase shared=47111 aligned=47111 conflicts=0 only_python=0 only_cpp=0
word   shared=26775 aligned=26775 conflicts=0 only_python=15148 only_cpp=0
```

The `word_only_python` count is expected: cpp-pinyin's single-character
dictionary is keyed by `char16_t`, so non-BMP python-pinyin codepoints are
skipped instead of being written into an unsafe surrogate-key table.

## Metadata Contract

C++ graph builders must read GGUF metadata fields instead of hardcoding:

- `tokenizer.ggml.tokens`
- `xvoice.languages`
- `xvoice.audio.*`
- `xvoice.stage2.*`
- `xvoice.srp.*`
- `xvoice.vocos.*`
- `xvoice.sampler.*`

Existing pre-bootstrap GGUF files can use `--vocab` as a transitional fallback,
but the standalone C++ runtime target is a self-contained GGUF.

## Tensor Contract

Tensor shapes are validated in GGML order `(ne0, ne1, ne2, ne3)`.

Representative required tensors:

| Tensor | GGML `ne` |
| --- | --- |
| `stage2.txt.emb.weight` | `(512, 822)` |
| `stage2.txt.lang_ada.weight` | `(512, 1024)` |
| `stage2.txt.blk.0.dwconv.weight` | `(7, 1, 512)` |
| `stage2.txt.blk.0.pwconv1.weight` | `(512, 1024)` |
| `stage2.txt.blk.0.pwconv2.weight` | `(1024, 512)` |
| `stage2.inp.proj.weight` | `(712, 1024)` |
| `stage2.inp.pos.0.weight` | `(31, 64, 1024)` |
| `stage2.blk.0.attn.q.weight` | `(1024, 1024)` |
| `stage2.blk.0.ff.up.weight` | `(1024, 2048)` |
| `stage2.out.weight` | `(1024, 100)` |
| `srp.mel_proj.weight` | `(100, 512)` |
| `srp.blk.0.attn.q.weight` | `(512, 504)` |
| `srp.cls.3.weight` | `(512, 32)` |
| `vocos.backbone.embed.weight` | `(7, 100, 512)` |
| `vocos.backbone.convnext.0.dwconv.weight` | `(7, 1, 512)` |
| `vocos.backbone.convnext.0.pwconv1.weight` | `(512, 1536)` |
| `vocos.backbone.convnext.0.pwconv2.weight` | `(1536, 512)` |
| `vocos.head.out.weight` | `(512, 1026)` |

## SRP Runtime Contract

The SRP graph is ported from
`projects/x-voice-ggml-py/x_voice_ggml_py/graphs/srp.py`.

Inputs:

- `srp_mel`: GGML `ne=(100, seq_len)`, raw F32 memory matching numpy
  source-order shape `(seq_len, 100)`.
- positions are generated in C++ as I32 `0..seq_len-1`.

Outputs:

- `srp_input_embed`: GGML `ne=(512, seq_len)`.
- `srp_logits`: GGML `ne=(32, 1)`.
- speed value policy: `(argmax(logits) + 1) * 0.25`.

The graph uses upstream GGML C operators directly, including:

- `ggml_mul_mat`
- `ggml_im2col`
- `ggml_concat`
- `ggml_rope`
- `ggml_soft_max_ext`
- `ggml_norm`
- `ggml_gelu_erf`

## Stage-2 Runtime Contract

The Stage-2 graph port is proceeding in the same narrow parity slices as the
Python staging graphs in `x_voice_ggml_py/graphs/stage2.py`.

Implemented slices:

- `Stage2TimeEmbeddingGraph`
  - inputs: `time_hidden` GGML `ne=(256, batch)`,
    `time_language_ids` I32 `ne=(batch)`
  - output: `stage2_time_embed` GGML `ne=(1024, batch)`
  - current CPU F32 parity: max_abs `2.86102e-06`
- `Stage2InputEmbeddingGraph`
  - inputs: `x` and `cond` GGML `ne=(100, seq_len)`,
    `text_embed` GGML `ne=(512, seq_len)`
  - output: `stage2_input_embed` GGML `ne=(1024, seq_len)`
  - current CPU F32 parity: max_abs `2.86102e-06`
- `Stage2TextEmbeddingGraph`
  - inputs: `text_ids_plus_one` I32 `ne=(seq_len)`,
    `language_ids` I32 `ne=(seq_len)`, masks GGML `ne=(1, seq_len)`,
    `text_pos_embed` GGML `ne=(512, seq_len)`
  - output: `stage2_text_embed` GGML `ne=(512, seq_len)`
  - current CPU F32 parity: max_abs `1.2517e-06`
- `Stage2DiTBlockGraph` block 0
  - inputs: `x` GGML `ne=(1024, seq_len)`,
    `time_embed` GGML `ne=(1024, 1)`
  - generated input: I32 rotary positions `0..seq_len-1`
  - output: `stage2_block0` GGML `ne=(1024, seq_len)`
  - current CPU F32 parity: max_abs `3.05176e-05`
- `Stage2DiTBlockGraph` anchor blocks 1, 7, 15, and 21
  - same input/output contracts as block 0
  - current CPU F32 parity max_abs values:
    `2.28882e-05`, `1.14441e-05`, `7.62939e-05`, `2.44141e-04`
- `Stage2DiTStackGraph`
  - inputs: `x` GGML `ne=(1024, seq_len)`,
    `time_embed` GGML `ne=(1024, 1)`
  - generated input: I32 rotary positions `0..seq_len-1`
  - output: `stage2_stack22` GGML `ne=(1024, seq_len)`
  - current CPU F32 parity: max_abs `3.66211e-04`
- `Stage2OutputGraph`
  - inputs: `x` GGML `ne=(1024, seq_len)`,
    `time_embed` GGML `ne=(1024, 1)`
  - output: `stage2_output` GGML `ne=(100, seq_len)`
  - current CPU F32 parity: max_abs `9.53674e-07`
- `Stage2DiTForwardGraph`
  - inputs: `x` GGML `ne=(1024, seq_len)`,
    `time_embed` GGML `ne=(1024, 1)`
  - generated input: I32 rotary positions `0..seq_len-1`
  - graph: 22 DiT blocks followed by final AdaLayerNorm and mel projection
  - output: `stage2_forward22` GGML `ne=(100, seq_len)`
  - current CPU F32 parity: max_abs `2.5928e-06`
- `Stage2NoCfgSampler`
  - inputs: text ids/language ids/masks/position embedding, time language id,
    `cond`, `cond_mask`, `fixed_noise`, and `sampler_timesteps`
  - text-side language ids may use the reserved unknown-language row
    `len(languages)` for prefix/anchor tokens
  - time language id must be a real language id
  - per-step graph order:
    `text_embed` once, then `time_embed -> input_embed -> dit_forward -> Euler`
  - final output applies the conditioning region with
    `torch.where(cond_mask, cond, sampled)` semantics
  - output: sampled mel and conditioned mel, both GGML `ne=(100, seq_len)`
  - current CPU F32 parity:
    - `stage2_no_cfg_sampler_y1` max_abs `2.38419e-07`
    - `stage2_no_cfg_sampler_y32` max_abs `1.44005e-04`
    - `stage2_no_cfg_sampler_out` max_abs `1.44005e-04`
- `Stage2NonLayeredCfgSampler`
  - branches per step:
    - full: current sample, reference `cond`, full text embedding
    - null: current sample, zero `cond`, null text embedding
  - null text embedding uses zero text ids and language ids where normal text
    positions are the reserved unknown-language row `len(languages)`
  - CFG schedule mirrors `stage2_sampler.py::cfg_schedule_values`
  - combination: `full + (full - null) * cfg`
  - current CPU F32 parity:
    - `stage2_nonlayered_cfg_sampler_y32` max_abs `1.32561e-04`
    - `stage2_nonlayered_cfg_sampler_out` max_abs `1.32561e-04`
- `Stage2LayeredCfgSampler`
  - branches per step:
    - full: current sample, reference `cond`, full text embedding
    - text-only: current sample, zero `cond`, full text embedding
    - null: current sample, zero `cond`, null text embedding
  - combination:
    `null + (1 + cfg2 * min(t / 0.01, 1)) * (text - null) + (1 + cfg) * (full - text)`
  - current CPU F32 parity:
    - `stage2_layered_cfg_sampler_y32` max_abs `2.87056e-04`
    - `stage2_layered_cfg_sampler_out` max_abs `2.87056e-04`

The synthesis runner is implemented:

- `XVoiceRuntime::synthesize_from_ref_mel`
- `XVoiceRuntime::synthesize_from_wav`
- CLI: `--synthesize-ref-mel`
- CLI: `--synthesize-ref-wav`
- CLI product alias: `--synthesize`
- CLI sidecar: `--metadata-json`
- input: IPA/tokens text plus reference mel source-order `(ref_frames, 100)`
- input: IPA/tokens text plus PCM reference WAV
- path:
  `SRP speed -> Stage-2 drop-text host prep -> Stage-2 sampler -> generated mel slice -> Vocos decode -> WAV`
- CLI smoke/perf control: `--ref-max-frames N` trims the source-order
  reference mel after the real WAV frontend has run. This preserves the WAV
  frontend path while keeping CPU Stage-2 CFG tests tractable on long reference
  clips.

The WAV frontend is project-local C++ DSP, not a root GGML binding:

- PCM WAV mono load
- linear resample to `xvoice.audio.sample_rate`
- RMS boost when below `target_rms`
- reflect center padding
- periodic Hann STFT
- HTK mel filterbank
- `log(clamp(mel, 1e-5))`

Vocos decode parity is now anchored at:

- `vocos_head_logits` max_abs `2.86102e-05`
- `vocos_head_real` max_abs `4.15603e-07`
- `vocos_head_imag` max_abs `3.50177e-07`
- `vocos_waveform` max_abs `6.14091e-09`

## Verification

```bash
cmake -S projects/x-voice-ggml-cpp -B projects/x-voice-ggml-cpp/build \
  -DXVOICE_ALLOW_WORKBENCH_GGML_FALLBACK=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build projects/x-voice-ggml-cpp/build -j8

projects/x-voice-ggml-cpp/build/x-voice-cli \
  --model /root/code/ggbond/models/x-voice-f32.gguf \
  --inspect \
  --validate-tensors

projects/x-voice-ggml-cpp/build/x-voice-cli \
  --model /root/code/ggbond/models/x-voice-f32.gguf \
  --text 'fa1|tie1|ren2 di4|di' \
  --text-kind ipa \
  --language zh \
  --print-tokens

UV_CACHE_DIR=/tmp/uv-cache uv run --no-project --with numpy python \
  projects/x-voice-ggml-cpp/scripts/dev/export_vocos_fixture_raw.py

projects/x-voice-ggml-cpp/build/x-voice-cli \
  --model /root/code/ggbond/models/x-voice-f32.gguf \
  --load-tensors \
  --check-vocos-head \
  --seq-len 8 \
  --vocos-mel projects/x-voice-ggml-cpp/build/fixtures/vocos/vocos_mel.f32 \
  --vocos-head-logits-ref projects/x-voice-ggml-cpp/build/fixtures/vocos/vocos_head_logits.f32 \
  --vocos-real-ref projects/x-voice-ggml-cpp/build/fixtures/vocos/vocos_head_real.f32 \
  --vocos-imag-ref projects/x-voice-ggml-cpp/build/fixtures/vocos/vocos_head_imag.f32 \
  --threshold 5e-5 \
  -b cpu -t 4

projects/x-voice-ggml-cpp/build/x-voice-cli \
  --model /root/code/ggbond/models/x-voice-f32.gguf \
  --load-tensors \
  --check-vocos-waveform \
  --seq-len 8 \
  --vocos-mel projects/x-voice-ggml-cpp/build/fixtures/vocos/vocos_mel.f32 \
  --vocos-waveform-ref projects/x-voice-ggml-cpp/build/fixtures/vocos/vocos_waveform.f32 \
  --threshold 1e-6 \
  -b cpu -t 4

projects/x-voice-ggml-cpp/build/x-voice-cli \
  --model /root/code/ggbond/models/x-voice-f32.gguf \
  --load-tensors \
  --synthesize-vocos-mel \
  --seq-len 8 \
  --vocos-mel projects/x-voice-ggml-cpp/build/fixtures/vocos/vocos_mel.f32 \
  --output-wav /tmp/xvoice-vocos-fixture.wav \
  -b cpu -t 4

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

projects/x-voice-ggml-cpp/build/x-voice-cli \
  --model /root/code/ggbond/models/x-voice-f32.gguf \
  --check-ref-mel \
  --ref-wav /tmp/xvoice-vocos-fixture.wav \
  --ref-mel-ref /tmp/xvoice-vocos-fixture-refmel.f32 \
  --threshold 2e-4

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

UV_CACHE_DIR=/tmp/uv-cache uv run --no-project --with numpy python \
  projects/x-voice-ggml-cpp/scripts/dev/export_stage2_srp_fixture_raw.py

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
