#!/usr/bin/env python3
"""Run X-Voice C++ synthesis benchmark sweeps and write JSON metrics.

The harness intentionally drives the public CLI instead of linking internal C++
symbols. This keeps it aligned with the x-voice-ggml-cpp product boundary:
GGUF load, reference frontend, Stage-2 sampler, Vocos decode, and WAV write are
measured exactly as users run them.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import subprocess
import time
import wave
from pathlib import Path
from typing import Any


DEFAULT_TEXT = "发帖人（弟弟）在详细描述其五口之家的现状后，寻求处理家庭问题的建议，并提出了自己的初步计划。"


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_wav_mono_f32(path: Path) -> tuple[int, list[float]]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        raw = wav.readframes(frames)
    if sample_width != 2:
        raise RuntimeError(f"expected 16-bit PCM WAV, got sample width {sample_width}: {path}")
    samples = struct.unpack("<" + "h" * (len(raw) // 2), raw)
    if channels == 1:
        mono = [float(sample) / 32768.0 for sample in samples]
    else:
        mono = []
        for index in range(0, len(samples), channels):
            mono.append(sum(samples[index:index + channels]) / float(channels * 32768.0))
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
    high = 0.0
    total = 0.0
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate))
    high_mask = freqs > 8000.0
    for start in range(0, max(1, audio.size - n_fft + 1), hop):
        frame = audio[start:start + n_fft]
        if frame.size < n_fft:
            frame = np.pad(frame, (0, n_fft - frame.size))
        spectrum = np.abs(np.fft.rfft(frame * window)) ** 2
        total += float(np.sum(spectrum))
        high += float(np.sum(spectrum[high_mask]))
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
    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
    clip_ratio = sum(1 for sample in samples if abs(sample) >= 0.999) / len(samples)
    return {
        "sample_rate": sample_rate,
        "samples": len(samples),
        "duration_seconds": len(samples) / float(sample_rate),
        "rms": rms,
        "peak": peak,
        "clip_ratio": clip_ratio,
        "hf_ratio_gt_8k": hf_ratio_gt_8k(sample_rate, samples),
    }


def profile_by_phase(metadata: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in metadata.get("profile", []):
        phase = str(item.get("phase", ""))
        if phase:
            out[phase] = float(item.get("elapsed_seconds", 0.0))
    return out


def ref_args(ref_spec: str) -> tuple[list[str], str]:
    if ref_spec == "full":
        return [], "full"
    if ref_spec.startswith("auto:"):
        max_frames = int(ref_spec.split(":", 1)[1])
        return ["--ref-auto-trim", "--ref-max-frames", str(max_frames)], f"auto_{max_frames}"
    start, frames = ref_spec.split(":", 1)
    return ["--ref-start-frame", start, "--ref-max-frames", frames], f"{start}_{frames}"


def run_one(args: argparse.Namespace, mode: str, steps: int, ref_spec: str, index: int) -> dict[str, Any]:
    extra_ref_args, ref_label = ref_args(ref_spec)
    stem = f"{index:03d}_{mode}_s{steps}_ref{ref_label}"
    wav_path = args.output_dir / f"{stem}.wav"
    metadata_path = args.output_dir / f"{stem}.json"
    cmd = [
        str(args.cli),
        "--model", str(args.model),
        "--load-tensors",
        "--synthesize",
        "--text", args.text,
        "--text-kind", args.text_kind,
        "--language", args.language,
        "--ref-wav", str(args.ref_wav),
        "--sampler-mode", mode,
        "--step-count", str(steps),
        "--speed-value", str(args.speed_value),
        "--output-wav", str(wav_path),
        "--metadata-json", str(metadata_path),
        "--no-progress",
        "-b", args.backend,
        "-t", str(args.threads),
    ]
    cmd.extend(extra_ref_args)
    if args.profile_stage2_branches:
        cmd.append("--profile-stage2-branches")
    if args.stage2_batch_dit_forward:
        cmd.append("--stage2-batch-dit-forward")
    env = os.environ.copy()
    if args.disable_tf32:
        env["NVIDIA_TF32_OVERRIDE"] = "0"
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=args.repo_root, env=env, text=True, capture_output=True)
    wall_seconds = time.perf_counter() - started
    row: dict[str, Any] = {
        "mode": mode,
        "step_count": steps,
        "ref_spec": ref_spec,
        "wall_seconds": wall_seconds,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "wav": str(wav_path),
        "metadata": str(metadata_path),
        "command": cmd,
    }
    if proc.returncode != 0:
        row["ok"] = False
        return row
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    row["ok"] = True
    row["profile_seconds"] = profile_by_phase(metadata)
    row["audio"] = metadata.get("audio", {})
    row["reference"] = metadata.get("reference", {})
    row["sampler"] = metadata.get("sampler", {})
    row["audio_metrics"] = audio_metrics(wav_path)
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("ok")]
    failed_rows = [row for row in rows if not row.get("ok")]
    def best_by(key: str) -> dict[str, Any] | None:
        candidates = [row for row in ok_rows if row.get("audio_metrics", {}).get(key) is not None]
        return min(candidates, key=lambda row: row["audio_metrics"][key]) if candidates else None
    return {
        "rows": len(rows),
        "ok": len(ok_rows),
        "failed": len(failed_rows),
        "best_hf_ratio_gt_8k": best_by("hf_ratio_gt_8k"),
        "fastest": min(ok_rows, key=lambda row: row["wall_seconds"]) if ok_rows else None,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--cli", type=Path, default=repo_root / "projects/x-voice-ggml-cpp/build-cuda/x-voice-cli")
    parser.add_argument("--model", type=Path, default=repo_root / "models/x-voice-f32.gguf")
    parser.add_argument("--ref-wav", type=Path, default=repo_root / "models/test.wav")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "projects/x-voice-ggml-cpp/benchmarks/synthesis")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--text-kind", default="plain")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--speed-value", type=float, default=10.0)
    parser.add_argument("--modes", default="no_cfg,cfg_nonlayered,cfg_layered")
    parser.add_argument("--steps", default="24,32,36,40")
    parser.add_argument("--refs", default="40:256,40:384,full")
    parser.add_argument("--quick", action="store_true", help="Run a small smoke matrix.")
    parser.add_argument("--profile-stage2-branches", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stage2-batch-dit-forward", action="store_true")
    parser.add_argument("--disable-tf32", action="store_true", default=True)
    ns = parser.parse_args()

    if ns.quick:
        modes = ["cfg_nonlayered"]
        steps = [24, 32]
        refs = ["40:384", "auto:384"]
    else:
        modes = parse_csv_strings(ns.modes)
        steps = parse_csv_ints(ns.steps)
        refs = parse_csv_strings(ns.refs)

    ns.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = ns.output_dir / "bench_xvoice_synthesis.jsonl"
    jsonl_path.write_text("", encoding="utf-8")
    rows: list[dict[str, Any]] = []
    index = 0
    for mode in modes:
        for step_count in steps:
            for ref_spec in refs:
                index += 1
                print(f"[{index}/{len(modes) * len(steps) * len(refs)}] {mode} steps={step_count} ref={ref_spec}", flush=True)
                row = run_one(ns, mode, step_count, ref_spec, index)
                rows.append(row)
                with jsonl_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                if not row.get("ok"):
                    print(f"  failed: {row.get('stderr', '')[:500]}", flush=True)
                else:
                    metrics = row["audio_metrics"]
                    print(
                        "  wall={:.3f}s sampler={:.3f}s hf>8k={} rms={:.5f} peak={:.5f}".format(
                            row["wall_seconds"],
                            row["profile_seconds"].get("stage2 sampler", 0.0),
                            "n/a" if metrics["hf_ratio_gt_8k"] is None else f"{metrics['hf_ratio_gt_8k']:.5f}",
                            metrics["rms"],
                            metrics["peak"],
                        ),
                        flush=True,
                    )
    summary = summarize(rows)
    with (ns.output_dir / "bench_xvoice_synthesis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"summary: {ns.output_dir / 'bench_xvoice_synthesis_summary.json'}")


if __name__ == "__main__":
    main()
