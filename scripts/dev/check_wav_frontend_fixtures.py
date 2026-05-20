#!/usr/bin/env python3
"""Generate small WAV frontend fixtures and compare C++ mel output to Python reference.

The Python side uses the project-local X-Voice NumPy frontend as the reference.
The C++ side is checked through ``x-voice-cli --check-ref-mel``. Mel arrays are
stored in source-order ``(frames, mel_channel_count)`` while the C++ tensor
reports GGML ``ne=(mel_channel_count, frames)``.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
PY_PROJECT_ROOT = REPO_ROOT / "projects" / "x-voice-ggml-py"
for path in (REPO_ROOT, PY_PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from x_voice_ggml_py.audio_frontend import reference_mel_from_wav
from x_voice_ggml_py.spec import XVoiceSpec


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check C++ reference WAV frontend against Python reference fixtures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=str(REPO_ROOT / "models" / "x-voice-f32.gguf"))
    parser.add_argument("--manifest", default=str(REPO_ROOT / "models" / "x-voice-f32.manifest.json"))
    parser.add_argument("--binary", default=str(REPO_ROOT / "projects" / "x-voice-ggml-cpp" / "build" / "x-voice-cli"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "projects" / "x-voice-ggml-cpp" / "build" / "fixtures" / "wav_frontend"))
    parser.add_argument("--threshold", type=float, default=2e-2)
    parser.add_argument("--target-rms", type=float, default=0.1)
    return parser


def write_wav_i16(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(samples, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    clipped = np.clip(values, -1.0, 1.0)
    pcm = np.rint(clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(int(pcm.shape[1]))
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def sine(sample_rate: int, seconds: float, freq: float, amp: float) -> np.ndarray:
    count = max(int(round(sample_rate * seconds)), 1)
    t = np.arange(count, dtype=np.float64) / float(sample_rate)
    return (amp * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)


def chirp(sample_rate: int, seconds: float, f0: float, f1: float, amp: float) -> np.ndarray:
    count = max(int(round(sample_rate * seconds)), 1)
    t = np.arange(count, dtype=np.float64) / float(sample_rate)
    k = (f1 - f0) / max(seconds, 1e-9)
    phase = 2.0 * math.pi * (f0 * t + 0.5 * k * t * t)
    return (amp * np.sin(phase)).astype(np.float32)


def build_cases() -> list[dict[str, object]]:
    left = sine(48000, 0.42, 330.0, 0.12)
    right = sine(48000, 0.42, 550.0, 0.08)
    return [
        {
            "name": "mono_24k_normal",
            "sample_rate": 24000,
            "samples": sine(24000, 0.40, 440.0, 0.14),
            "purpose": "native 24 kHz mono reference",
        },
        {
            "name": "mono_32k_low_rms",
            "sample_rate": 32000,
            "samples": sine(32000, 0.35, 220.0, 0.01),
            "purpose": "non-24 kHz input plus RMS boost",
        },
        {
            "name": "mono_16k_long",
            "sample_rate": 16000,
            "samples": chirp(16000, 1.20, 120.0, 1600.0, 0.10),
            "purpose": "longer non-24 kHz input",
        },
        {
            "name": "stereo_48k_downmix",
            "sample_rate": 48000,
            "samples": np.stack([left, right], axis=1),
            "purpose": "stereo input accepted by deterministic downmix",
        },
    ]


def load_spec(manifest_path: Path) -> XVoiceSpec:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return XVoiceSpec.from_mapping(payload["metadata"])


def run_check(binary: Path, model: Path, wav_path: Path, ref_path: Path, threshold: float, target_rms: float) -> str:
    cmd = [
        str(binary),
        "--model",
        str(model),
        "--check-ref-mel",
        "--ref-wav",
        str(wav_path),
        "--ref-mel-ref",
        str(ref_path),
        "--target-rms",
        str(target_rms),
        "--threshold",
        str(threshold),
    ]
    completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return completed.stdout.strip()


def run_invalid_check(binary: Path, model: Path, invalid_path: Path, ref_path: Path) -> str:
    cmd = [
        str(binary),
        "--model",
        str(model),
        "--check-ref-mel",
        "--ref-wav",
        str(invalid_path),
        "--ref-mel-ref",
        str(ref_path),
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.returncode == 0:
        raise RuntimeError("invalid WAV fixture unexpectedly passed")
    for line in (completed.stderr or completed.stdout).strip().splitlines():
        if line.startswith("error:"):
            return line
    return (completed.stderr or completed.stdout).strip().splitlines()[0]


def main() -> None:
    args = get_parser().parse_args()
    model = Path(args.model)
    binary = Path(args.binary)
    out_dir = Path(args.out_dir)
    spec = load_spec(Path(args.manifest))

    if not binary.is_file():
        raise FileNotFoundError(f"missing x-voice-cli binary: {binary}")
    if not model.is_file():
        raise FileNotFoundError(f"missing model GGUF: {model}")

    rows: list[dict[str, object]] = []
    for case in build_cases():
        name = str(case["name"])
        sample_rate = int(case["sample_rate"])
        wav_path = out_dir / f"{name}.wav"
        ref_path = out_dir / f"{name}.f32"
        write_wav_i16(wav_path, np.asarray(case["samples"], dtype=np.float32), sample_rate)
        mel, rms = reference_mel_from_wav(wav_path, spec=spec, target_rms=float(args.target_rms))
        np.asarray(mel, dtype="<f4", order="C").tofile(ref_path)
        output = run_check(binary, model, wav_path, ref_path, float(args.threshold), float(args.target_rms))
        rows.append(
            {
                "name": name,
                "purpose": case["purpose"],
                "wav": str(wav_path),
                "ref": str(ref_path),
                "source_sample_rate": sample_rate,
                "frames": int(mel.shape[0]),
                "mel_channel_count": int(mel.shape[1]),
                "rms_before_boost": float(rms),
                "check": output,
            }
        )

    invalid_path = out_dir / "invalid_not_wav.wav"
    invalid_path.write_bytes(b"this is not a wav")
    invalid_ref = out_dir / "invalid_ref.f32"
    invalid_ref.write_bytes(struct.pack("<f", 0.0))
    invalid_error = run_invalid_check(binary, model, invalid_path, invalid_ref)

    manifest = {
        "schema_version": 1,
        "fixture_kind": "xvoice.cpp.wav_frontend",
        "model": str(model),
        "threshold": float(args.threshold),
        "target_rms": float(args.target_rms),
        "cases": rows,
        "invalid_wav": {
            "path": str(invalid_path),
            "expected_error": invalid_error,
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wav_frontend_fixtures: ok cases={len(rows)} manifest={manifest_path}")


if __name__ == "__main__":
    main()
