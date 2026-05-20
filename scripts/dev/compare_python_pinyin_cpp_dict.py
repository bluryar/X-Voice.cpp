#!/usr/bin/env python3
"""Compare vendored python-pinyin loaded data with cpp-pinyin dictionaries.

This script intentionally imports pypinyin from the vendored python-pinyin
submodule so the comparison follows python-pinyin's own data loading path:

    pypinyin.constants -> pinyin_dict.py / phrases_dict.py JSON loaders

For cpp-pinyin, the script mirrors the upstream dictionary load order:

    phrases_dict.txt -> user_dict.txt override

Outputs are JSON/JSONL audit files, not runtime frontend inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[4]
PROJECT = ROOT / "projects" / "x-voice-ggml-cpp"
DEFAULT_PYTHON_PINYIN = PROJECT / "vendor" / "python-pinyin"
DEFAULT_CPP_PINYIN_DICT = PROJECT / "resources" / "zh_pinyin_dict"
DEFAULT_OUT = PROJECT / "build" / "python_pinyin_cpp_alignment"


@dataclass
class Reading:
    reading: List[str]
    normalized: List[str]
    source: str


def load_python_pinyin(repo: Path):
    sys.path.insert(0, str(repo))
    from pypinyin.constants import PHRASES_DICT, PINYIN_DICT  # type: ignore
    from pypinyin.style._tone_convert import to_tone3  # type: ignore

    return PHRASES_DICT, PINYIN_DICT, to_tone3


def normalize_syllable(value: str, to_tone3) -> str:
    value = value.strip()
    if not value:
        return ""
    value = value.replace("5", "")
    return to_tone3(value, v_to_u=False, neutral_tone_with_five=False).strip()


def normalize_reading(reading: Sequence[str], to_tone3) -> List[str]:
    return [x for x in (normalize_syllable(item, to_tone3) for item in reading) if x]


def read_cpp_phrase_file(path: Path, source_name: str, to_tone3, sep: str) -> Dict[str, Reading]:
    out: Dict[str, Reading] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            word, raw = line.split(":", 1)
            word = word.strip()
            if sep == ",":
                reading = [item.strip() for item in raw.split(",") if item.strip()]
            else:
                reading = [item.strip() for item in raw.split() if item.strip()]
            if not word or not reading:
                continue
            out[word] = Reading(
                reading=reading,
                normalized=normalize_reading(reading, to_tone3),
                source=f"{source_name}:{line_no}",
            )
    return out


def load_cpp_phrase_dict(cpp_dict: Path, to_tone3) -> Dict[str, Reading]:
    mandarin = cpp_dict / "mandarin"
    phrases = read_cpp_phrase_file(mandarin / "phrases_dict.txt", "cpp:phrases_dict.txt", to_tone3, ",")
    user = read_cpp_phrase_file(mandarin / "user_dict.txt", "cpp:user_dict.txt", to_tone3, " ")
    phrases.update(user)
    return phrases


def load_cpp_word_dict(cpp_dict: Path, to_tone3) -> Dict[str, Reading]:
    out: Dict[str, Reading] = {}
    path = cpp_dict / "mandarin" / "word.txt"
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            word, raw = line.split(":", 1)
            word = word.strip()
            reading = [item.strip() for item in raw.split(",") if item.strip()]
            if not word or not reading:
                continue
            out[word] = Reading(
                reading=reading,
                normalized=normalize_reading(reading, to_tone3),
                source=f"cpp:word.txt:{line_no}",
            )
    return out


def load_python_phrase_dict(repo: Path, to_tone3) -> Dict[str, Reading]:
    phrases, _single, _ = load_python_pinyin(repo)
    out: Dict[str, Reading] = {}
    for word, items in phrases.items():
        reading: List[str] = []
        for pys in items:
            if pys:
                reading.append(str(pys[0]))
        if not reading:
            continue
        out[str(word)] = Reading(
            reading=reading,
            normalized=normalize_reading(reading, to_tone3),
            source="python-pinyin:PHRASES_DICT",
        )
    return out


def load_python_word_dict(repo: Path, to_tone3) -> Dict[str, Reading]:
    _phrases, single, _ = load_python_pinyin(repo)
    out: Dict[str, Reading] = {}
    for codepoint, raw in single.items():
        try:
            word = chr(int(codepoint))
        except (TypeError, ValueError):
            word = chr(codepoint)
        reading = [item.strip() for item in str(raw).split(",") if item.strip()]
        if not reading:
            continue
        out[word] = Reading(
            reading=reading,
            normalized=normalize_reading(reading, to_tone3),
            source="python-pinyin:PINYIN_DICT",
        )
    return out


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def reading_json(value: Reading) -> dict:
    return {
        "reading": value.reading,
        "normalized": value.normalized,
        "source": value.source,
    }


def compare_dicts(kind: str, python: Dict[str, Reading], cpp: Dict[str, Reading], out_dir: Path) -> dict:
    python_words = set(python)
    cpp_words = set(cpp)
    shared = sorted(python_words & cpp_words)
    only_python = sorted(python_words - cpp_words)
    only_cpp = sorted(cpp_words - python_words)

    conflicts: List[dict] = []
    aligned = 0
    for word in shared:
        py = python[word]
        cp = cpp[word]
        if py.normalized == cp.normalized:
            aligned += 1
        else:
            conflicts.append({
                "word": word,
                "python": reading_json(py),
                "cpp": reading_json(cp),
            })

    only_python_count = write_jsonl(
        out_dir / f"{kind}_only_python.jsonl",
        ({"word": word, "python": reading_json(python[word])} for word in only_python),
    )
    only_cpp_count = write_jsonl(
        out_dir / f"{kind}_only_cpp.jsonl",
        ({"word": word, "cpp": reading_json(cpp[word])} for word in only_cpp),
    )
    conflict_count = write_jsonl(out_dir / f"{kind}_conflicts.jsonl", conflicts)
    sample_count = write_jsonl(
        out_dir / f"{kind}_aligned_sample.jsonl",
        (
            {"word": word, "python": reading_json(python[word]), "cpp": reading_json(cpp[word])}
            for word in shared[:200]
            if python[word].normalized == cpp[word].normalized
        ),
    )

    return {
        "kind": kind,
        "python_entries": len(python),
        "cpp_entries": len(cpp),
        "shared": len(shared),
        "aligned": aligned,
        "conflicts": conflict_count,
        "only_python": only_python_count,
        "only_cpp": only_cpp_count,
        "aligned_sample": sample_count,
        "conflicts_path": str(out_dir / f"{kind}_conflicts.jsonl"),
        "only_python_path": str(out_dir / f"{kind}_only_python.jsonl"),
        "only_cpp_path": str(out_dir / f"{kind}_only_cpp.jsonl"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-pinyin", type=Path, default=DEFAULT_PYTHON_PINYIN)
    parser.add_argument("--cpp-pinyin-dict", type=Path, default=DEFAULT_CPP_PINYIN_DICT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    python_repo = args.python_pinyin.resolve()
    cpp_dict = args.cpp_pinyin_dict.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _phrases, _single, to_tone3 = load_python_pinyin(python_repo)

    python_phrases = load_python_phrase_dict(python_repo, to_tone3)
    cpp_phrases = load_cpp_phrase_dict(cpp_dict, to_tone3)
    python_words = load_python_word_dict(python_repo, to_tone3)
    cpp_words = load_cpp_word_dict(cpp_dict, to_tone3)

    phrase_summary = compare_dicts("phrase", python_phrases, cpp_phrases, out_dir)
    word_summary = compare_dicts("word", python_words, cpp_words, out_dir)

    summary = {
        "schema_version": 1,
        "python_pinyin": str(python_repo),
        "cpp_pinyin_dict": str(cpp_dict),
        "output": str(out_dir),
        "python_loading": "import pypinyin.constants.PHRASES_DICT/PINYIN_DICT from vendored python-pinyin",
        "cpp_loading": "phrases_dict.txt then user_dict.txt override for phrases; word.txt for single chars",
        "phrase": phrase_summary,
        "word": word_summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "python_pinyin_cpp_alignment: ok "
        f"phrase_conflicts={phrase_summary['conflicts']} "
        f"word_conflicts={word_summary['conflicts']} "
        f"out={out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
