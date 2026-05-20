# ADR-0001: C++ v0 Runtime Boundary

## Status

Accepted

## Context

The Python staging project already proves the default X-Voice pipeline through
GGML graphs and a single Stage-2/SRP/Vocos GGUF. The risky remaining gap for a
C++ runtime is not the choice of model components; it is preserving the same
metadata, tensor, and frontend contracts without pulling Python phonemization
dependencies into the first C++ port.

## Decision

C++ v0 supports only deterministic frontend inputs:

- `ipa`
- `tokens`

Raw plain text, Chinese G2P, English G2P, and automatic language detection are
deferred until after the Stage-2/SRP/Vocos graph path is running.

The runtime loads GGUF metadata and tensor headers first, validates
representative tensor shapes in GGML `ne` order, and keeps graph execution as
explicit follow-up milestones.

## Consequences

- The C++ project can quickly become a stable parity target for the Python
  graphs.
- The first C++ CLI is useful for inspecting `x-voice-f32.gguf` and validating
  IPA/token frontend behavior.
- Product-grade plain text remains a separate frontend project:
  `cppjieba + cppinyin + X-Voice tone-sandhi/override` for Chinese, with English
  frontend choices evaluated later.

## Amendment: 2026-05-20

The model graph boundary remains unchanged: Stage-2 still receives deterministic
X-Voice ipa_v6 vocab ids, and the Chinese frontend is project-local host code,
not a GGML graph concern.

For product CLI usability, `--text-kind plain --language zh` is now enabled when
the build can find the vendored cpp-pinyin and cppjieba sources. The CMake switch
is `XVOICE_ENABLE_ZH_FRONTEND=AUTO|ON|OFF`. The implementation uses a generated
cpp-pinyin dictionary root aligned to python-pinyin data, cppjieba segmentation,
and a project override TSV for common narration words and polyphones.

The generated root is built offline from the vendored `mozillazg/python-pinyin`
submodule. The submodule is for debug/resource generation only; it is not linked
into runtime targets. cpp-pinyin source data only provides support files such as
traditional-to-simplified conversion.
