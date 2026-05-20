# X-Voice GGML C++ Roadmap

## Milestone 0: Bootstrap

- Create standalone C++ project structure.
- Build against upstream GGML.
- Parse GGUF metadata and tensor headers.
- Validate representative tensor `ne` contracts.
- Implement IPA/tokens frontend contract.

## Milestone 1: Audio And Host Prep

- Port SRP speed-to-duration policy.
- Port Stage-2 drop-text host prep:
  - prefix token padding
  - `. ` anchor ids
  - token language ids
  - prompt-token language bypass mask
  - text keep mask
  - fixed noise
  - sampler timesteps
- Add WAV read/write.
- Add reference audio normalization and Vocos log-mel frontend.

Status: complete for the C++ v0 IPA/tokens boundary.

- WAV write is implemented for mono 16-bit PCM output.
- Stage-2 drop-text host prep is implemented for source-order reference mel
  inputs in `XVoiceRuntime::synthesize_from_ref_mel()`.
- Reference WAV read/log-mel frontend is implemented as project-local C++ DSP:
  PCM WAV, linear resample, RMS boost, reflect padding, periodic Hann STFT, HTK
  mel filterbank, and log clamp.

## Milestone 2: SRP Graph

- Port SRP input embedding and all transformer blocks.
- Preserve GGML operator names and operand order.
- Validate logits against
  `/root/code/ggbond/models/x-voice-stage2-srp-torch-ref.npz`.

Status: complete for CPU F32 parity.

- `srp_input_embed` max_abs `1.43051e-06`
- `srp_logits` max_abs `9.53674e-07`
- speed class/value parity: `26` / `6.75`

## Milestone 3: Stage-2 Graph

- Port time, text, input, DiT stack, final norm, and output graphs.
- Port no-CFG, non-layered CFG, and layered CFG sampler modes.
- Validate full sampler trajectory against
  `/root/code/ggbond/models/x-voice-stage2-host-torch-ref.npz`.

Status: complete for CPU F32 parity.

- `stage2_time_embed` max_abs `2.86102e-06`
- `stage2_text_embed` max_abs `1.2517e-06`
- `stage2_input_embed` max_abs `2.86102e-06`
- `stage2_block0` max_abs `3.05176e-05`
- `stage2_block1` max_abs `2.28882e-05`
- `stage2_block7` max_abs `1.14441e-05`
- `stage2_block15` max_abs `7.62939e-05`
- `stage2_block21` max_abs `2.44141e-04`
- `stage2_stack22` max_abs `3.66211e-04`
- `stage2_output` max_abs `9.53674e-07`
- `stage2_forward22` max_abs `2.5928e-06`
- no-CFG sampler:
  - `stage2_no_cfg_sampler_y1` max_abs `2.38419e-07`
  - `stage2_no_cfg_sampler_y32` max_abs `1.44005e-04`
  - `stage2_no_cfg_sampler_out` max_abs `1.44005e-04`
- non-layered CFG sampler:
  - `stage2_nonlayered_cfg_sampler_y32` max_abs `1.32561e-04`
  - `stage2_nonlayered_cfg_sampler_out` max_abs `1.32561e-04`
- layered CFG sampler:
  - `stage2_layered_cfg_sampler_y32` max_abs `2.87056e-04`
  - `stage2_layered_cfg_sampler_out` max_abs `2.87056e-04`

## Milestone 4: Vocos And Waveform

- Port Vocos neural backbone/head into GGML. Done:
  - `vocos_head_logits` max_abs `2.86102e-05`
  - `vocos_head_real` max_abs `4.15603e-07`
  - `vocos_head_imag` max_abs `3.50177e-07`
- Reconstruct waveform with project-local ISTFT. Done:
  - `vocos_waveform` max_abs `6.14091e-09`
- `--synthesize-vocos-mel` can write a mono PCM WAV from a raw mel fixture.
- `--synthesize-ref-mel` can run IPA/tokens plus a source-order reference mel
  through Stage-2 sampler, generated-mel slicing, Vocos decode, and WAV write.
- `--synthesize-ref-wav` can run IPA/tokens plus a real reference WAV through
  the project-local WAV/log-mel frontend, Stage-2 sampler, generated-mel
  slicing, Vocos decode, and WAV write.
- `--ref-max-frames` and `--ref-auto-trim` are available for smoke/perf tests
  and product-ish reference windows; they trim the source-order reference mel
  after the real WAV frontend has run.

## Milestone 5: Product Inputs

- Longer real-reference WAV smoke and performance coverage is in place for the
  current `--synthesize-ref-wav` path.
- `--synthesize` is now the product-facing alias for the reference-WAV path.
- `--metadata-json` writes a compact run sidecar.
- `scripts/dev/check_xvoice_cpp_v0_smoke.sh` freezes the current v0 smoke.
  It can run against CUDA with `XVOICE_BACKEND=cuda`.
- `scripts/dev/build_xvoice_cuda.sh` configures `GGML_CUDA=ON`,
  `GGML_CUDA_NO_VMM=ON`, and the detected NVIDIA architecture into
  `build-cuda`.
- `scripts/dev/check_wav_frontend_fixtures.py` covers native 24 kHz, non-24 kHz,
  low-RMS, stereo downmix, and invalid WAV frontend behavior.
- `scripts/dev/refresh_self_contained_gguf.sh` verifies `tokenizer.ggml.tokens`
  and can force a full bundle re-export.
- `--preset product` is now the recommended v0 reference-WAV synthesis preset:
  `cfg_nonlayered`, `32` steps, auto reference trim with `384` frames, and SRP
  automatic speed clamped into `8..12`.
- `scripts/dev/check_xvoice_synthesis_regression.py` runs the public CUDA CLI
  and checks product metadata plus WAV metrics rather than waveform exactness.
- `scripts/dev/synthesize_product_cuda.sh` is the one-command product preset
  wrapper for local CUDA synthesis.
- Chinese raw text frontend is available behind `XVOICE_ENABLE_ZH_FRONTEND`:
  `AUTO` enables it when vendored cpp-pinyin/cppjieba sources exist, and `ON`
  requires them. It uses the generated `resources/zh_pinyin_dict` root plus
  `resources/zh_frontend_overrides.tsv`.
- `xvoice-merge-pinyin-dicts` builds `resources/zh_pinyin_dict` from vendored
  python-pinyin dictionaries, while cpp-pinyin source data only provides
  `trans_word.txt` support data.
- `vendor/python-pinyin` remains a debug/resource-generation submodule; it is
  not linked into the runtime frontend.
- Add English frontend later with CMUdict/Flite or an explicitly selected
  alternative.
- Keep full 30-language phonemization out of the core C++ runtime until the
  model graph is complete.
- Keep expanding SRP/duration policy fixtures. The first product guard is in
  place: automatic SRP speed is recorded as `raw_speed_value` and clamped by
  the product preset before duration calculation.
- Automatic reference-audio trim is available through `--ref-auto-trim`; keep
  tuning it with fixture coverage before making it the only recommended path.
- Sampler-quality sweeps are available through
  `scripts/dev/bench_xvoice_synthesis.py`. Use them to choose product presets
  from measured latency, `hf>8k`, peak/RMS, and generated-frame metadata rather
  than one-off listening logs.
- Next sampler optimization target: batch or fuse Stage-2 CFG branch work.
  Current CUDA branch profiling shows non-layered CFG spends most sampler time
  in full/null DIT forward passes. Two safe steps are complete: sampler code now
  reuses time/input/forward GGML graph runners while rebinding allocations per
  call, and the sampler branch path now fuses `stage2.inp.* -> 22x DiT ->
  stage2.out` so the intermediate hidden input embedding no longer round-trips
  through host memory. A true branch-batched DiT forward also exists behind
  `--stage2-batch-dit-forward`, with CFG branches packed into GGML `ne2`.
  It now keeps input embedding, DIT forward, CFG combine, and Euler update
  inside one graph step, but it is still slightly slower than the default
  conditioned fusion path on the local CUDA benchmark. Do not use a naive
  two-copy full/null graph: it was tested on the 4060 Ti and exceeded the
  practical GGML allocator/activation boundary. The next performance attempt
  should target lower-level CUDA/GGML scheduling overhead or a custom fused
  branch op only if profiling shows a clear backend-side win.

## Milestone 6: Open-Source/Submodule Readiness

- `docs/release.md` defines the standalone repository boundary and the ggbond
  submodule consumption path.
- `scripts/dev/check_release_tree.py` gates release-tree structure in workbench
  or standalone mode.
- `vendor/ggml` is available as a nested submodule pinned to upstream
  `v0.9.11`, and `.gitmodules.standalone` records the extracted-repository
  submodule template.
- Runtime source license is Apache-2.0; model weights remain external artifacts
  governed by their own upstream terms.
- Source release should keep `include/`, `src/`, `resources/`, `scripts/`,
  `docs/`, `tools/`, and vendored third-party source/submodule pointers.
- Build trees, benchmarks, generated WAV/JSON outputs, and model weights remain
  outside the source repository.
- A standalone release should carry `vendor/ggml`; the parent-workbench
  `-DXVOICE_ALLOW_WORKBENCH_GGML_FALLBACK=ON` path is development-only.
