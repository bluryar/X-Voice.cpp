# X-Voice GGML C++ Status

## Current Snapshot

Date: 2026-05-20
Phase: C++ reference-WAV inference entry frozen; zh plain-text frontend enabled when local dictionaries are available

## Current State

- Project boundary exists under `projects/x-voice-ggml-cpp`.
- Build system links directly against upstream GGML C/C++ through CMake.
- Local workbench fallback is explicit and opt-in:
  `-DXVOICE_ALLOW_WORKBENCH_GGML_FALLBACK=ON`.
- `XVoiceRuntime` loads GGUF metadata and tensor headers without loading tensor
  payloads by default, and can optionally load all 728 tensor payloads into a
  GGML backend buffer with `RuntimeOptions::load_tensor_data=true`.
- The current F32 bundle loads into CPU as:
  - `loaded_tensors=728`
  - `weight_buffer_bytes=1619068640`
- CUDA build is available through:
  `bash projects/x-voice-ggml-cpp/scripts/dev/build_xvoice_cuda.sh`.
  On the local RTX 4060 Ti test machine, `build-cuda/x-voice-cli -b cuda`
  initializes `CUDA0`, loads the same 728 tensors into a CUDA backend weight
  buffer, and reports `weight_buffer_bytes=1619070592`.
  Strict CUDA-vs-CPU fixture parity is looser than CPU reference parity even
  with `NVIDIA_TF32_OVERRIDE=0`: observed SRP logits `max_abs=0.00191545`
  with the same speed class, and Stage-2 forward `max_abs=0.00554371`.
  Treat CUDA as the product-speed backend and CPU as the strict parity backend
  until GGML CUDA math-mode behavior is tuned further.
- `XVoiceSpec` mirrors the Python staging metadata contracts for:
  - audio
  - Stage-2
  - SRP
  - Vocos
  - sampler
  - languages
- `XVoiceTokenizer` mirrors the Python `ipa` and `tokens` input boundary:
  - IPA string -> X-Voice ipa_v6 tokens
  - token list -> raw vocab ids
  - unknown token fallback to UTF-8 characters
- The current `/root/code/ggbond/models/x-voice-f32.gguf` is self-contained and
  reports `tokenizer_tokens=820`; the C++ runtime still supports transitional
  `RuntimeOptions::vocab_path` and CLI `--vocab` for old pre-bootstrap bundles.
- C++ still preserves the deterministic IPA/tokens boundary for parity, and now
  also supports zh-only `--text-kind plain --language zh` when built with
  `-DXVOICE_ENABLE_ZH_FRONTEND=AUTO` or `ON`.
  - `AUTO` enables the frontend when the vendored `cpp-pinyin` and `cppjieba`
    sources exist.
  - cpp-pinyin now uses the generated
    `resources/zh_pinyin_dict` root, and that root is generated directly from
    vendored python-pinyin `phrases_dict.json` and `pinyin_dict.json`.
  - `vendor/python-pinyin` is a git submodule pinned at
    `8595294b1a97845e30f11ecfdb3caa4e61ac3988` for debug/resource generation
    only. It is not linked into CMake targets or required at runtime.
  - `scripts/dev/compare_python_pinyin_cpp_dict.py` imports python-pinyin's own
    `PHRASES_DICT`/`PINYIN_DICT` view and compares it with the generated runtime
    dictionary. Current audit output is under
    `build/python_pinyin_cpp_alignment`; it reports `0` phrase reading conflicts
    and `0` single-character reading conflicts. `word_only_python=15148` is
    expected because cpp-pinyin's single-character table is `char16_t`-keyed and
    non-BMP python-pinyin codepoints are skipped.
  - `resources/zh_frontend_overrides.tsv` adds X-Voice narration words and
    common polyphone overrides, with longest override matching before cppjieba.
  - CLI overrides: `--pinyin-dict`, `--jieba-dict-dir`, `--zh-override`.
- Stage-2 has full pre-output graph footholds:
  - `stage2_time_embed`: max_abs `2.86102e-06`
  - `stage2_text_embed`: max_abs `1.2517e-06`
  - `stage2_input_embed`: max_abs `2.86102e-06`
  - `stage2_block0`: max_abs `3.05176e-05`
  - `stage2_block1`: max_abs `2.28882e-05`
  - `stage2_block7`: max_abs `1.14441e-05`
  - `stage2_block15`: max_abs `7.62939e-05`
  - `stage2_block21`: max_abs `2.44141e-04`
  - `stage2_stack22`: max_abs `3.66211e-04`
  - `stage2_output`: max_abs `9.53674e-07`
  - `stage2_forward22`: max_abs `2.5928e-06`
- Stage-2 host sampler now runs all upstream 32-step modes:
  - `stage2_no_cfg_sampler_y1`: max_abs `2.38419e-07`
  - `stage2_no_cfg_sampler_y32`: max_abs `1.44005e-04`
  - `stage2_no_cfg_sampler_out`: max_abs `1.44005e-04`
  - `stage2_nonlayered_cfg_sampler_y32`: max_abs `1.32561e-04`
  - `stage2_nonlayered_cfg_sampler_out`: max_abs `1.32561e-04`
  - `stage2_layered_cfg_sampler_y32`: max_abs `2.87056e-04`
  - `stage2_layered_cfg_sampler_out`: max_abs `2.87056e-04`
  - text-side language ids allow the reserved unknown-language row
    `len(languages)` for prefix/anchor tokens; time language ids remain real
    language ids only.
- SRP has crossed the first real graph parity boundary:
  - `srp_input_embed`: max_abs `1.43051e-06`
  - `srp_logits`: max_abs `9.53674e-07`
  - speed class parity: C++ `26`, reference `26`, speed `6.75`
- Vocos decode now runs as C++ GGML neural graph plus project-local ISTFT:
  - input mel source order is `(frames, mel_channel_count)`, loaded into GGML
    `ne=(mel_channel_count, frames)`
  - `vocos_head_logits`: max_abs `2.86102e-05`
  - `vocos_head_real`: max_abs `4.15603e-07`
  - `vocos_head_imag`: max_abs `3.50177e-07`
  - `vocos_waveform`: max_abs `6.14091e-09`
  - `--synthesize-vocos-mel` writes a mono 24 kHz PCM WAV from raw mel fixtures.
  - ISTFT/waveform reconstruction is project-local and is not a root GGML
    binding.
- The first synthesis glue is in place:
  - `XVoiceRuntime::synthesize_from_ref_mel()` takes IPA/tokens text plus a
    source-order reference mel `(ref_frames, 100)`.
  - It runs SRP speed policy unless `--speed-value` is supplied.
  - It prepares Stage-2 drop-text host tensors, runs `no_cfg`,
    `cfg_nonlayered`, or `cfg_layered`, slices the generated mel region, and
    decodes through Vocos.
  - CLI smoke:
    `--synthesize-ref-mel --sampler-mode no_cfg --step-count 1` wrote
    `/tmp/xvoice-cpp-refmel-smoke.wav` with `47616` samples at `24000` Hz.
- Reference WAV frontend is now project-local C++ DSP:
  - PCM WAV mono loading, linear resampling, RMS boost, reflect center padding,
    periodic Hann STFT, HTK mel filterbank, and `log(clamp(1e-5))`.
  - `--check-ref-mel` matched the Python frontend on the tiny WAV fixture with
    `max_abs=1.38283e-05`.
  - `--synthesize-ref-wav --sampler-mode no_cfg --step-count 1` wrote
    `/tmp/xvoice-cpp-refwav-smoke.wav` with `47616` samples at `24000` Hz.
- Longer real-reference WAV smoke is now covered:
  - `/root/code/ggbond/models/test.wav` produces `797` source prompt frames at
    `32000` Hz before resampling/mel frontend output.
  - uncropped prompt, no-CFG, `--step-count 1`: `6.95s`, `~1.76GB` max RSS.
  - uncropped prompt, default non-layered CFG, `--step-count 2`: `18.31s`,
    `~1.78GB` max RSS.
  - `--ref-max-frames 64`, default non-layered CFG, full `nfe_step=32`:
    `49.18s`, `~1.62GB` max RSS, wrote
    `/tmp/xvoice-cpp-testwav-default-full-ref64.wav`.
  - `--ref-max-frames` trims the source-order reference mel after the real WAV
    frontend has run, so it is a smoke/perf control knob rather than a separate
    audio path.
- Product-facing CLI entry is now `--synthesize` as an alias for the
  reference-WAV path, with `--metadata-json` for a compact run sidecar.
- `--preset product` is the recommended v0 synthesis entry. It selects
  `cfg_nonlayered`, `32` sampler steps, `--ref-auto-trim --ref-max-frames 384`,
  and automatic SRP speed clamping into `8..12` unless `--speed-value` is
  explicitly provided.
- Synthesis metadata records `raw_speed_class`, `raw_speed_value`, final
  `speed_value`, `speed_policy`, and active automatic speed bounds.
- CLI synthesis progress reporting is in place:
  - interactive stderr enables it automatically
  - `--progress` forces it on
  - `--no-progress` disables it
  - phases currently include tensor loading, reference mel, text frontend, SRP
    duration, Stage-2 prep/sampler, Vocos decode, and WAV writing
  - synthesis metadata records the same phase timings under `profile`
  - `--profile-stage2-branches` additionally records cumulative Stage-2 DIT
    forward time for CFG branches: `stage2 branch/full`,
    `stage2 branch/null`, and for layered CFG `stage2 branch/text`
- One-command smoke script:
  `bash projects/x-voice-ggml-cpp/scripts/dev/check_xvoice_cpp_v0_smoke.sh`.
  Set `XVOICE_BACKEND=cuda` to use the CUDA build tree and run the same short
  smoke on `-b cuda`.
- CUDA synthesis timing/debug snapshot for the zh raw-text sample:
  - `cfg_nonlayered`, `--step-count 24`, `--speed-value 8.0`:
    `7.24s`, `960` generated frames, predicted `10.25s`.
  - `cfg_nonlayered`, `--step-count 24`, `--speed-value 10.0`:
    `5.69s`, `768` generated frames, predicted `8.2s`.
  - `cfg_nonlayered`, `--step-count 24`, `--speed-value 12.0`:
    `4.90s`, `640` generated frames, predicted `6.83s`.
  - The same sample with automatic SRP previously predicted a slow speaking
    speed and could sound word-by-word. With `--preset product`, the current
    local result is `raw_speed_value=4.5`, final `speed_value=8`, and
    `speed_policy=auto_clamped`.
  - The local `test.wav` starts with roughly `0.35s` of near-silence. Using only
    `--ref-max-frames 128` raises measured high-frequency output energy
    (`hf>8k ~= 0.0276`) and can sound metallic. Longer/skipped prompt windows
    such as `--ref-start-frame 40 --ref-max-frames 384` reduce the same metric
    to about `0.0063`.
  - `--step-count` now controls the generated runtime timestep grid. Values
    above the GGUF default `nfe_step=32` are accepted for artifact experiments
    instead of failing against the default 33-point grid.
- Reference-WAV crop selection:
  - `--ref-auto-trim` detects the first voiced-ish mel frame and then applies
    `--ref-max-frames`. On the local `test.wav`, the current default threshold
    selects `ref_start_frame=35`, close to the hand-tuned `40`.
  - explicit `--ref-start-frame` still wins over auto trim.
- Benchmark harness:
  - `scripts/dev/bench_xvoice_synthesis.py` runs CLI synthesis matrices for
    sampler modes, `step_count=24/32/36/40`, and reference windows such as
    `40:256`, `40:384`, and `full`, writing JSONL plus a summary JSON.
  - quick CUDA smoke on the zh sample (`cfg_nonlayered`, `speed_value=10`,
    `ref_max_frames=384`) produced:
    - `24` steps, manual `40:384`: wall `8.07s`, sampler `6.32s`,
      `hf>8k=0.00023`
    - `24` steps, auto trim: wall `8.01s`, sampler `6.30s`,
      `hf>8k=0.00032`
    - `32` steps, manual `40:384`: wall `10.04s`, sampler `8.42s`,
      `hf>8k=0.00022`
    - `32` steps, auto trim: wall `10.06s`, sampler `8.40s`,
      `hf>8k=0.00032`
  - branch profiling for the `32`-step non-layered CFG run shows
    `stage2 branch/full ~= 3.93s` and `stage2 branch/null ~= 3.92s`, versus
    total `stage2 sampler ~= 8.42s`. This confirms CFG branch DIT forward
    batching is the next meaningful optimization target.
- End-to-end CUDA product regression is covered by
  `scripts/dev/check_xvoice_synthesis_regression.py`. The current local run
  passes with:
  - `sampler.mode=cfg_nonlayered`
  - `raw_speed_value=4.5`, `speed_value=8`, `speed_policy=auto_clamped`
  - `duration_seconds=10.2293`, `rms=0.113526`, `peak=0.8396`
  - `clip_ratio=0`, `hf_ratio_gt_8k=0.000560577`
  - `stage2 sampler=9.512682s`, `load tensors=0.619937s`
- `scripts/dev/synthesize_product_cuda.sh` wraps the product preset synthesis
  command and accepts environment overrides for model, reference WAV, text,
  output WAV/metadata, and threads.
- GGUF quantization tooling is now available:
  - target: `xvoice-quantize-gguf`
  - script: `scripts/dev/quantize_xvoice_gguf.sh`
  - docs: `docs/quantization.md`
  - policy: conservative large-matrix conversion only; conv/norm/bias/embedding
    and positional tensors remain in their source type
  - generated local artifacts:
    `x-voice-f16.gguf`, `x-voice-q8_0.gguf`, `x-voice-q6_k.gguf`,
    `x-voice-q4_k.gguf`
  - CUDA metadata/tensor contract validation passes for all four variants.
  - A short CUDA `no_cfg --step-count 1` q8 smoke wrote
    `/tmp/xvoice-q8-smoke.wav`.
- CMake install now installs the runtime, headers, resources, docs, and scripts
  while excluding Python `__pycache__`/`.pyc` files. Local check:
  `cmake --install projects/x-voice-ggml-cpp/build-cuda --prefix /tmp/xvoice-cpp-install`
  followed by a script-cache scan passed.
- Stage-2 sampler graph scheduling optimization:
  - Time embedding, input embedding, and 22-block DiT forward now use
    sampler-local reusable GGML graph runners. Each call still rebinds the graph
    allocation with `ggml_gallocr_alloc_graph()` before setting inputs, because
    reusing the graph without that step changed generated audio.
  - The sampler path now also has a fused conditioned DiT runner that builds
    `stage2.inp.* -> 22x DiT -> stage2.out` as one reusable graph per branch.
    This removes the intermediate hidden `ne=(1024, seq_len)` readback from the
    old `input_embedding -> host -> forward` boundary while preserving GGML
    operator order.
  - The optimization preserves sampler fixture parity:
    - no-CFG out max_abs `0.000144005`
    - non-layered CFG out max_abs `0.000132561`
    - layered CFG out max_abs `0.000287056`
  - The zh quick CUDA benchmark after runner reuse:
    - `24` steps, manual `40:384`: wall `7.60s`, sampler `5.91s`
    - `24` steps, auto trim: wall `7.54s`, sampler `5.91s`
    - `32` steps, manual `40:384`: wall `9.55s`, sampler `7.89s`
    - `32` steps, auto trim: wall `9.58s`, sampler `7.91s`
  - The zh quick CUDA benchmark after the conditioned DiT fusion:
    - `24` steps, manual `40:384`: wall `8.30s`, sampler `5.82s`
    - `24` steps, auto trim: wall `7.46s`, sampler `5.83s`
    - `32` steps, manual `40:384`: wall `9.46s`, sampler `7.77s`
    - `32` steps, auto trim: wall `9.41s`, sampler `7.78s`
  - Experimental CFG batch-step execution is implemented behind
    `--stage2-batch-dit-forward`. It packs branch inputs into GGML `ne2`, keeps
    `stage2.inp.* -> 22x DiT -> stage2.out`, CFG combine, and Euler update
    inside one graph step, and reads back only the next sampled mel:
    - non-layered CFG uses branch `ne2=2` for full/null.
    - layered CFG uses branch `ne2=3` for full/text/null.
    - input embedding batch parity passes with
      `stage2_input_embed max_abs=1.43051e-05`.
    - non-layered CFG CUDA parity passes with
      `max_abs=0.000354767`.
    - layered CFG CUDA parity passes with
      `max_abs=0.000280857`.
  - The CFG batch-step path is experimental rather than default. On the local
    CUDA serial benchmark for `cfg_nonlayered`, `32` steps, manual `40:384`, it
    is still slightly slower than conditioned DiT fusion:
    - default conditioned path: wall `9.517s`, sampler `7.761s`
    - CFG batch-step path: wall `9.689s`, sampler `7.933s`
    The most likely reason is that branchwise input-position conv and larger
    per-step activation residency offset the reduced host readback/combine work.
  - The default conditioned path was re-verified after adding the experimental
    flag:
    - `24` steps, manual `40:384`: wall `7.50s`, sampler `5.82s`
    - `24` steps, auto trim: wall `7.11s`, sampler `5.72s`
    - `32` steps, manual `40:384`: wall `9.25s`, sampler `7.73s`
    - `32` steps, auto trim: wall `9.39s`, sampler `7.78s`
  - WAV output for the manual `40:384` quick benchmark is bit-identical to the
    pre-optimization output.
  - A simple two-branch full/null graph copy was tested and rejected: on the
    local 4060 Ti it triggered GGML allocator failure (`tensor buffer not set`),
    likely because both 22-block activation sets must be resident at once. The
    next real branch optimization should be batch-aware attention/linear graph
    construction, not just duplicating two full forward graphs.
- WAV frontend fixture script covers native 24 kHz, non-24 kHz resampling,
  low-RMS boost, stereo downmix, and invalid WAV rejection:
  `uv run --no-project --with numpy python projects/x-voice-ggml-cpp/scripts/dev/check_wav_frontend_fixtures.py`.
- Self-contained GGUF helper:
  `bash projects/x-voice-ggml-cpp/scripts/dev/refresh_self_contained_gguf.sh`.
- Open-source/submodule checklist lives in `docs/release.md`. Runtime source now
  carries Apache-2.0, while model-weight redistribution rights remain separate
  from runtime source packaging.
- `scripts/dev/check_release_tree.py` now passes in workbench mode and reports
  the expected release warning that generated build directories are present.
  `vendor/ggml` is now available as a nested submodule pinned to upstream
  `v0.9.11`.
- Standalone CUDA configure/build was verified from a clean `/tmp` build tree
  without `XVOICE_ALLOW_WORKBENCH_GGML_FALLBACK`, using local
  `projects/x-voice-ggml-cpp/vendor/ggml` at `49f84a92` (`v0.9.11`).
  The standalone binary passed `--inspect --validate-tensors -b cuda` and a zh
  plain-text tokenization smoke (`重庆银行重新排队。`).

## Next Work

1. Keep expanding the zh frontend override/tone-sandhi fixture list before
   making it a default product input.
2. Keep `resources/zh_pinyin_dict` regenerated from python-pinyin when the
   submodule is updated, and keep debug comparison output in the ignored
   `build/python_pinyin_cpp_alignment` directory.
3. Keep the compact end-to-end regression fixture passing on CUDA. It compares
   C++ metadata JSON and WAV properties, not waveform exactness.
4. Use the benchmark harness to choose product defaults and to validate sampler
   scheduling refactors.
5. Keep the default sampler on conditioned DiT fusion. Use
   `--stage2-batch-dit-forward` only for experiments; current CFG batch-step
   parity is good, but local CUDA speed is still slightly worse than default.
6. Expand the SRP/duration policy guard fixtures for zh product text. The first
   guard is in place through automatic speed clamp metadata and
   `--preset product`.
7. Add English frontend later with CMUdict/Flite or another explicitly selected
   non-eSpeak route.

## Boundary

Do not add root GGML FFT/ISTFT bindings for X-Voice C++ until a concrete
cross-project need exists. Keep waveform reconstruction project-local.
