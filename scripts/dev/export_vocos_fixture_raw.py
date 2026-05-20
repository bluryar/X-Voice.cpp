#!/usr/bin/env python3
"""Export Vocos fixture arrays to raw little-endian F32 files for C++ checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export x-voice Vocos NPZ fixture/ref arrays to raw F32 files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fixture", default="/root/code/ggbond/models/x-voice-vocos-fixture.npz")
    parser.add_argument("--torch-ref", default="/root/code/ggbond/models/x-voice-vocos-torch-ref.npz")
    parser.add_argument("--out-dir", default="/root/code/ggbond/projects/x-voice-ggml-cpp/build/fixtures/vocos")
    return parser


def write_f32(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(values, dtype="<f4", order="C").tofile(path)


def main() -> None:
    args = get_parser().parse_args()
    out_dir = Path(args.out_dir)
    fixture = np.load(args.fixture)
    torch_ref = np.load(args.torch_ref)

    mel = np.asarray(fixture["vocos_mel"], dtype=np.float32)
    head_logits = np.asarray(torch_ref["vocos_head_logits"], dtype=np.float32)
    real = np.asarray(torch_ref["vocos_head_real"], dtype=np.float32)
    imag = np.asarray(torch_ref["vocos_head_imag"], dtype=np.float32)
    waveform = np.asarray(torch_ref["vocos_waveform"], dtype=np.float32).reshape(-1)
    if mel.ndim != 2:
        raise ValueError(f"vocos_mel must have source-order shape (frames, mel), got {mel.shape}")
    frames = int(mel.shape[0])

    paths = {
        "vocos_mel": out_dir / "vocos_mel.f32",
        "vocos_head_logits": out_dir / "vocos_head_logits.f32",
        "vocos_head_real": out_dir / "vocos_head_real.f32",
        "vocos_head_imag": out_dir / "vocos_head_imag.f32",
        "vocos_waveform": out_dir / "vocos_waveform.f32",
    }
    write_f32(paths["vocos_mel"], mel)
    write_f32(paths["vocos_head_logits"], head_logits)
    write_f32(paths["vocos_head_real"], real)
    write_f32(paths["vocos_head_imag"], imag)
    write_f32(paths["vocos_waveform"], waveform)

    manifest = {
        "schema_version": 1,
        "fixture_kind": "xvoice.vocos.cpp_raw",
        "source_fixture": str(Path(args.fixture)),
        "source_torch_ref": str(Path(args.torch_ref)),
        "frames": frames,
        "arrays": {
            "vocos_mel": {
                "path": str(paths["vocos_mel"]),
                "source_order_shape": list(mel.shape),
                "ggml_ne": [int(mel.shape[1]), frames],
            },
            "vocos_head_logits": {
                "path": str(paths["vocos_head_logits"]),
                "torch_shape": list(head_logits.shape),
                "ggml_ne": [int(head_logits.shape[-1]), frames],
            },
            "vocos_head_real": {
                "path": str(paths["vocos_head_real"]),
                "torch_shape": list(real.shape),
                "ggml_ne": [frames, int(real.shape[1])],
            },
            "vocos_head_imag": {
                "path": str(paths["vocos_head_imag"]),
                "torch_shape": list(imag.shape),
                "ggml_ne": [frames, int(imag.shape[1])],
            },
            "vocos_waveform": {
                "path": str(paths["vocos_waveform"]),
                "torch_shape": list(torch_ref["vocos_waveform"].shape),
                "source_order_shape": [int(waveform.shape[0])],
            },
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
