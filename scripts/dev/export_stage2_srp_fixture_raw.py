#!/usr/bin/env python3
"""Export Python NPZ Stage-2/SRP fixtures to raw files for the C++ parity CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write raw F32 fixture files consumed by x-voice-ggml-cpp",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fixture", default="/root/code/ggbond/models/x-voice-stage2-srp-fixture.npz")
    parser.add_argument("--torch-ref", default="/root/code/ggbond/models/x-voice-stage2-srp-torch-ref.npz")
    parser.add_argument("--output-dir", default="/root/code/ggbond/models/x-voice-cpp-fixtures/stage2-srp")
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

    manifest = {
        "stage2_time_hidden": write_f32(out_dir / "stage2_time_hidden.f32", fixture["stage2_time_hidden"]),
        "stage2_time_language_ids": write_i32(
            out_dir / "stage2_time_language_ids.i32",
            fixture["stage2_time_language_ids"],
        ),
        "stage2_time_embed": write_f32(out_dir / "stage2_time_embed.f32", torch_ref["stage2_time_embed"]),
        "stage2_text_ids_plus_one": write_i32(
            out_dir / "stage2_text_ids_plus_one.i32",
            fixture["stage2_text_ids_plus_one"],
        ),
        "stage2_text_language_ids": write_i32(
            out_dir / "stage2_text_language_ids.i32",
            fixture["stage2_text_language_ids"],
        ),
        "stage2_text_no_lang_fusion_mask": write_f32(
            out_dir / "stage2_text_no_lang_fusion_mask.f32",
            fixture["stage2_text_no_lang_fusion_mask"],
        ),
        "stage2_text_keep_mask": write_f32(
            out_dir / "stage2_text_keep_mask.f32",
            fixture["stage2_text_keep_mask"],
        ),
        "stage2_text_pos_embed": write_f32(
            out_dir / "stage2_text_pos_embed.f32",
            fixture["stage2_text_pos_embed"],
        ),
        "stage2_text_embed": write_f32(out_dir / "stage2_text_embed.f32", torch_ref["stage2_text_embed"]),
        "stage2_x": write_f32(out_dir / "stage2_x.f32", fixture["stage2_x"]),
        "stage2_cond": write_f32(out_dir / "stage2_cond.f32", fixture["stage2_cond"]),
        "stage2_input_text_embed": write_f32(
            out_dir / "stage2_input_text_embed.f32",
            fixture["stage2_input_text_embed"],
        ),
        "stage2_input_embed": write_f32(out_dir / "stage2_input_embed.f32", torch_ref["stage2_input_embed"]),
        "stage2_stack22": write_f32(out_dir / "stage2_stack22.f32", torch_ref["stage2_stack22"]),
        "stage2_final_norm": write_f32(out_dir / "stage2_final_norm.f32", torch_ref["stage2_final_norm"]),
        "stage2_output": write_f32(out_dir / "stage2_output.f32", torch_ref["stage2_output"]),
        "srp_mel": write_f32(out_dir / "srp_mel.f32", fixture["srp_mel"]),
        "srp_input_embed": write_f32(out_dir / "srp_input_embed.f32", torch_ref["srp_input_embed"]),
        "srp_logits": write_f32(out_dir / "srp_logits.f32", torch_ref["srp_logits"]),
    }
    for block_index in (0, 1, 7, 15, 21):
        manifest[f"stage2_block{block_index}_input"] = write_f32(
            out_dir / f"stage2_block{block_index}_input.f32",
            torch_ref[f"stage2_block{block_index}_input"],
        )
        manifest[f"stage2_block{block_index}"] = write_f32(
            out_dir / f"stage2_block{block_index}.f32",
            torch_ref[f"stage2_block{block_index}"],
        )
    manifest["seq_len"] = int(fixture["srp_mel"].shape[0])
    manifest["mel_channel_count"] = int(fixture["srp_mel"].shape[1])
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"fixture_raw_dir: {out_dir}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
