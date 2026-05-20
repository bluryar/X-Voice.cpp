#!/usr/bin/env python3
"""Export Python Stage-2 host sampler NPZ fixtures to raw C++ parity files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write raw Stage-2 host sampler fixture files consumed by x-voice-ggml-cpp",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fixture", default="/root/code/ggbond/models/x-voice-stage2-host-fixture.npz")
    parser.add_argument("--torch-ref", default="/root/code/ggbond/models/x-voice-stage2-host-torch-ref.npz")
    parser.add_argument("--output-dir", default="/root/code/ggbond/models/x-voice-cpp-fixtures/stage2-host")
    return parser


def write_f32(path: Path, array: np.ndarray) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(array, dtype=np.float32)
    arr.tofile(path)
    return {"path": str(path), "shape": list(arr.shape), "dtype": "float32", "bytes": path.stat().st_size}


def write_i32(path: Path, array: np.ndarray) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(array, dtype=np.int32)
    arr.tofile(path)
    return {"path": str(path), "shape": list(arr.shape), "dtype": "int32", "bytes": path.stat().st_size}


def main() -> None:
    args = get_parser().parse_args()
    fixture = np.load(args.fixture)
    torch_ref = np.load(args.torch_ref)
    out_dir = Path(args.output_dir)

    manifest: dict[str, object] = {
        "stage2_text_ids_plus_one": write_i32(out_dir / "stage2_text_ids_plus_one.i32", fixture["stage2_text_ids_plus_one"]),
        "stage2_text_language_ids": write_i32(out_dir / "stage2_text_language_ids.i32", fixture["stage2_text_language_ids"]),
        "stage2_text_no_lang_fusion_mask": write_f32(out_dir / "stage2_text_no_lang_fusion_mask.f32", fixture["stage2_text_no_lang_fusion_mask"]),
        "stage2_text_keep_mask": write_f32(out_dir / "stage2_text_keep_mask.f32", fixture["stage2_text_keep_mask"]),
        "stage2_text_pos_embed": write_f32(out_dir / "stage2_text_pos_embed.f32", fixture["stage2_text_pos_embed"]),
        "stage2_time_language_ids": write_i32(out_dir / "stage2_time_language_ids.i32", fixture["stage2_time_language_ids"]),
        "stage2_cond": write_f32(out_dir / "stage2_cond.f32", fixture["stage2_cond"]),
        "stage2_cond_mask": write_f32(out_dir / "stage2_cond_mask.f32", fixture["stage2_cond_mask"]),
        "stage2_fixed_noise": write_f32(out_dir / "stage2_fixed_noise.f32", fixture["stage2_fixed_noise"]),
        "stage2_sampler_timesteps": write_f32(out_dir / "stage2_sampler_timesteps.f32", fixture["stage2_sampler_timesteps"]),
        "stage2_host_sampler_out": write_f32(out_dir / "stage2_host_sampler_out.f32", torch_ref["stage2_host_sampler_out"]),
        "stage2_host_cfg_nonlayered_out": write_f32(
            out_dir / "stage2_host_cfg_nonlayered_out.f32",
            torch_ref["stage2_host_cfg_nonlayered_out"],
        ),
        "stage2_host_cfg_layered_out": write_f32(
            out_dir / "stage2_host_cfg_layered_out.f32",
            torch_ref["stage2_host_cfg_layered_out"],
        ),
        "seq_len": int(fixture["stage2_cond"].shape[0]),
        "mel_channel_count": int(fixture["stage2_cond"].shape[1]),
        "step_count": int(fixture["stage2_sampler_timesteps"].shape[0] - 1),
    }
    for step in range(int(manifest["step_count"]) + 1):
        manifest[f"stage2_host_sampler_y{step}"] = write_f32(
            out_dir / f"stage2_host_sampler_y{step}.f32",
            torch_ref[f"stage2_host_sampler_y{step}"],
        )
        key = f"stage2_host_cfg_nonlayered_y{step}"
        if key in torch_ref:
            manifest[key] = write_f32(out_dir / f"{key}.f32", torch_ref[key])
        key = f"stage2_host_cfg_layered_y{step}"
        if key in torch_ref:
            manifest[key] = write_f32(out_dir / f"{key}.f32", torch_ref[key])

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"fixture_raw_dir: {out_dir}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
