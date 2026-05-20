#!/usr/bin/env python3
"""End-to-end X-Voice C++ synthesis regression check.

This is a product smoke, not a strict waveform parity test. It runs the public
CLI, reads the metadata sidecar and WAV, then checks stable contracts: sampler
mode, speed policy, generated duration, sample rate, RMS/peak/clip, and optional
high-frequency energy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import subprocess
import wave
from pathlib import Path
from typing import Any


DEFAULT_TEXT = "发帖人（弟弟）在详细描述其五口之家的现状后，寻求处理家庭问题的建议，并提出了自己的初步计划。"


def read_wav_mono_f32(path: Path) -> tuple[int, list[float]]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        raw = wav.readframes(frames)
    if sample_width != 2:
        raise RuntimeError(f"expected 16-bit PCM WAV, got sample_width={sample_width}: {path}")
    values = struct.unpack("<" + "h" * (len(raw) // 2), raw)
    if channels == 1:
        return sample_rate, [value / 32768.0 for value in values]
    mono: list[float] = []
    for offset in range(0, len(values), channels):
        mono.append(sum(values[offset:offset + channels]) / float(channels * 32768.0))
    return sample_rate, mono


def hf_ratio_gt_8k(sample_rate: int, samples: list[float]) -> float | None:
    try:
        import numpy as np  # type: ignore
    except ImportError:
        return None
    if not samples:
        return 0.0
    audio = np.asarray(samples, dtype=np.float32)
    if audio.size < 1024:
        return 0.0
    n_fft = 2048
    hop = 1024
    window = np.hanning(n_fft).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate))
    high_mask = freqs > 8000.0
    high = 0.0
    total = 0.0
    for start in range(0, max(1, audio.size - n_fft + 1), hop):
        frame = audio[start:start + n_fft]
        if frame.size < n_fft:
            frame = np.pad(frame, (0, n_fft - frame.size))
        power = np.abs(np.fft.rfft(frame * window)) ** 2
        total += float(np.sum(power))
        high += float(np.sum(power[high_mask]))
    return 0.0 if total <= 0.0 else high / total


def audio_metrics(path: Path) -> dict[str, Any]:
    sample_rate, samples = read_wav_mono_f32(path)
    if not samples:
        return {
            "sample_rate": sample_rate,
            "samples": 0,
            "duration_seconds": 0.0,
            "rms": 0.0,
            "peak": 0.0,
            "clip_ratio": 0.0,
            "hf_ratio_gt_8k": 0.0,
        }
    peak = max(abs(sample) for sample in samples)
    rms = math.sqrt(sum(sample * sample for sample in samples) / float(len(samples)))
    clipped = sum(1 for sample in samples if abs(sample) >= 0.999)
    return {
        "sample_rate": sample_rate,
        "samples": len(samples),
        "duration_seconds": len(samples) / float(sample_rate),
        "rms": rms,
        "peak": peak,
        "clip_ratio": clipped / float(len(samples)),
        "hf_ratio_gt_8k": hf_ratio_gt_8k(sample_rate, samples),
    }


def profile_seconds(metadata: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in metadata.get("profile", []):
        phase = str(item.get("phase", ""))
        if phase:
            out[phase] = float(item.get("elapsed_seconds", 0.0))
    return out


def assert_range(name: str, value: float, lo: float, hi: float) -> None:
    if not (lo <= value <= hi):
        raise AssertionError(f"{name}={value} outside [{lo}, {hi}]")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--cli", type=Path, default=repo_root / "projects/x-voice-ggml-cpp/build-cuda/x-voice-cli")
    parser.add_argument("--model", type=Path, default=repo_root / "models/x-voice-f32.gguf")
    parser.add_argument("--ref-wav", type=Path, default=repo_root / "models/test.wav")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/xvoice-cpp-regression"))
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--text-kind", default="plain")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--preset", default="product")
    parser.add_argument("--speed-value", type=float, default=0.0)
    parser.add_argument("--min-speed", type=float, default=8.0)
    parser.add_argument("--max-speed", type=float, default=12.0)
    parser.add_argument("--min-duration", type=float, default=5.0)
    parser.add_argument("--max-duration", type=float, default=16.0)
    parser.add_argument("--min-rms", type=float, default=0.001)
    parser.add_argument("--max-rms", type=float, default=0.35)
    parser.add_argument("--min-peak", type=float, default=0.01)
    parser.add_argument("--max-clip-ratio", type=float, default=0.05)
    parser.add_argument("--max-hf-ratio-gt-8k", type=float, default=0.55)
    parser.add_argument("--disable-tf32", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = args.output_dir / "xvoice_regression.wav"
    metadata_path = args.output_dir / "xvoice_regression.json"
    summary_path = args.summary_json or (args.output_dir / "xvoice_regression_summary.json")

    cmd = [
        str(args.cli),
        "--model", str(args.model),
        "--load-tensors",
        "--synthesize",
        "--text", args.text,
        "--text-kind", args.text_kind,
        "--language", args.language,
        "--ref-wav", str(args.ref_wav),
        "--output-wav", str(wav_path),
        "--metadata-json", str(metadata_path),
        "--no-progress",
        "-b", args.backend,
        "-t", str(args.threads),
    ]
    if args.preset:
        cmd.extend(["--preset", args.preset])
    if args.speed_value > 0.0:
        cmd.extend(["--speed-value", str(args.speed_value)])

    env = os.environ.copy()
    if args.disable_tf32:
        env["NVIDIA_TF32_OVERRIDE"] = "0"
    proc = subprocess.run(cmd, cwd=args.repo_root, env=env, text=True, capture_output=True)
    if proc.returncode != 0:
        raise SystemExit(proc.stderr or proc.stdout)

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    metrics = audio_metrics(wav_path)
    sampler = metadata.get("sampler", {})
    audio = metadata.get("audio", {})
    reference = metadata.get("reference", {})

    if metadata.get("project") != "x-voice-ggml-cpp":
        raise AssertionError("metadata project mismatch")
    if metadata.get("text_kind") != args.text_kind:
        raise AssertionError("metadata text_kind mismatch")
    if metadata.get("language") != args.language:
        raise AssertionError("metadata language mismatch")
    if int(audio.get("sample_rate", 0)) != 24000 or int(metrics["sample_rate"]) != 24000:
        raise AssertionError("expected 24 kHz output")
    if args.preset == "product" and sampler.get("mode") != "cfg_nonlayered":
        raise AssertionError("product preset must use cfg_nonlayered sampler")
    if int(metadata.get("unit_count", 0)) < 20:
        raise AssertionError("frontend produced too few spoken units")
    if int(audio.get("generated_frames", 0)) <= 0 or int(audio.get("decode_frames", 0)) <= 0:
        raise AssertionError("generated/decode frames must be positive")
    if args.speed_value <= 0.0:
        assert_range("sampler.speed_value", float(sampler.get("speed_value", 0.0)), args.min_speed, args.max_speed)
        if sampler.get("speed_policy") not in {"auto", "auto_in_range", "auto_clamped"}:
            raise AssertionError(f"unexpected speed_policy={sampler.get('speed_policy')!r}")
    assert_range("audio.duration_seconds", float(metrics["duration_seconds"]), args.min_duration, args.max_duration)
    assert_range("audio.rms", float(metrics["rms"]), args.min_rms, args.max_rms)
    assert_range("audio.peak", float(metrics["peak"]), args.min_peak, 1.0)
    if float(metrics["clip_ratio"]) > args.max_clip_ratio:
        raise AssertionError(f"clip_ratio too high: {metrics['clip_ratio']}")
    if metrics["hf_ratio_gt_8k"] is not None and float(metrics["hf_ratio_gt_8k"]) > args.max_hf_ratio_gt_8k:
        raise AssertionError(f"hf_ratio_gt_8k too high: {metrics['hf_ratio_gt_8k']}")

    summary = {
        "ok": True,
        "command": cmd,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "metadata": str(metadata_path),
        "wav": str(wav_path),
        "sampler": sampler,
        "audio": audio,
        "reference": reference,
        "audio_metrics": metrics,
        "profile_seconds": profile_seconds(metadata),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
