#!/usr/bin/env python3
"""Check whether x-voice-ggml-cpp is ready to publish as source.

The default mode matches development inside the parent ggbond workbench, where
`vendor/ggml` may be supplied by `../../vendor/ggml`. Use `--standalone` before
publishing the extracted repository; that mode requires a local vendored or
submodule GGML tree.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_PATHS = [
    "CMakeLists.txt",
    "README.md",
    "AGENTS.md",
    ".gitignore",
    "include/x_voice.h",
    "src/model.cpp",
    "src/stage2.cpp",
    "src/srp.cpp",
    "src/vocos.cpp",
    "src/synthesis.cpp",
    "src/frontend.cpp",
    "src/audio.cpp",
    "src/x_voice_cli.cpp",
    "resources/zh_frontend_overrides.tsv",
    "resources/zh_pinyin_dict/manifest.json",
    "scripts/dev/build_xvoice_cuda.sh",
    "scripts/dev/synthesize_product_cuda.sh",
    "scripts/dev/quantize_xvoice_gguf.sh",
    "scripts/dev/check_xvoice_synthesis_regression.py",
    "scripts/dev/check_xvoice_cpp_v0_smoke.sh",
    "scripts/dev/bench_xvoice_synthesis.py",
    "tools/quantize_gguf.cpp",
    "docs/status.md",
    "docs/roadmap.md",
    "docs/release.md",
    "docs/quantization.md",
    "vendor/VENDOR_VERSIONS.md",
    "vendor/cpp-pinyin/CMakeLists.txt",
    "vendor/cppjieba/CMakeLists.txt",
    "vendor/nlohmann-json/single_include/nlohmann/json.hpp",
]

FORBIDDEN_SUFFIXES = {
    ".gguf",
    ".safetensors",
    ".pt",
    ".pth",
    ".onnx",
    ".wav",
    ".mp3",
    ".flac",
}

IGNORED_DIR_NAMES = {
    ".git",
}

GENERATED_DIR_NAMES = {
    "build",
    "build-cuda",
    "build-zh",
    "benchmarks",
    "out",
}


def iter_release_files(root: Path, skip_generated: bool):
    for path in root.rglob("*"):
        rel_parts = path.relative_to(root).parts
        if any(part in IGNORED_DIR_NAMES for part in rel_parts):
            continue
        if skip_generated and any(part in GENERATED_DIR_NAMES for part in rel_parts):
            continue
        yield path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    default_project = repo_root / "projects/x-voice-ggml-cpp"
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=default_project)
    parser.add_argument("--standalone", action="store_true", help="Require vendor/ggml for a standalone source repository.")
    parser.add_argument("--strict-clean", action="store_true", help="Fail if build/output directories are present.")
    parser.add_argument("--json", type=Path, help="Write machine-readable report.")
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    errors: list[str] = []
    warnings: list[str] = []

    for rel in REQUIRED_PATHS:
        if not (project_dir / rel).exists():
            errors.append(f"missing required path: {rel}")

    local_ggml = project_dir / "vendor/ggml/CMakeLists.txt"
    workbench_ggml = project_dir / "../../vendor/ggml/CMakeLists.txt"
    if args.standalone:
        if not local_ggml.exists():
            errors.append("standalone release requires vendor/ggml/CMakeLists.txt")
    elif not local_ggml.exists():
        if workbench_ggml.resolve().exists():
            warnings.append("vendor/ggml missing; workbench fallback is available")
        else:
            errors.append("vendor/ggml missing and workbench fallback is unavailable")

    if not (project_dir / "LICENSE").exists():
        warnings.append("top-level LICENSE is missing; choose one before publishing")

    for path in iter_release_files(project_dir, skip_generated=not args.strict_clean):
        if path.is_file() and (path.suffix.lower() == ".pyc" or "__pycache__" in path.parts):
            errors.append(f"release tree contains Python cache artifact: {path.relative_to(project_dir)}")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"release tree contains generated/model/audio artifact: {path.relative_to(project_dir)}")

    generated_dirs = []
    for rel in sorted(GENERATED_DIR_NAMES):
        if (project_dir / rel).exists():
            generated_dirs.append(rel)
    if generated_dirs:
        message = "generated directories present: " + ", ".join(generated_dirs)
        if args.strict_clean:
            errors.append(message)
        else:
            warnings.append(message)

    report = {
        "ok": not errors,
        "mode": "standalone" if args.standalone else "workbench",
        "project_dir": str(project_dir),
        "errors": errors,
        "warnings": warnings,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
