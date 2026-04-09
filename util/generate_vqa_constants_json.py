#!/usr/bin/env python3
"""Generate inspectable runtime constants for the ME-VQA requirements module."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def _build_config() -> dict:
    return {
        "meta": {
            "generator": "util/generate_vqa_constants_json.py",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "notes": [
                "This file stores static parsing/template constants for vqa_requirements.py.",
                "Procedural normalization and parsing logic intentionally remains in Python code.",
            ],
        },
        "question_templates": {
            "coarse": "What is the coarse expression class?",
            "fine": "What is the fine-grained expression class?",
            "single_au": "What is the action unit?",
            "multi_au": "What are the action units?",
        },
        "analysis_questions": [
            "Please analyse the micro-expression shown in this video in detail.",
            "Provide a detailed analysis of the micro-expression observed in this clip.",
            "Provide a comprehensive analysis of the facial micro-expression shown in this clip.",
        ],
        "combined_reference_questions": {
            "multi_au_fine_coarse": "What are the action units present, and based on them, what is the fine-grained expression and the coarse expression class?",
            "single_au_fine_coarse": "What is the action unit present, and based on it, what is the fine-grained expression and the coarse expression class?",
        },
        "regex": {
            "bool_au": {
                "pattern": "^Is the action unit (.+?) shown on the face\\?$",
                "flags": ["IGNORECASE"],
            }
        },
        "datasets": {
            "target": ["casme3", "samm"],
            "reference": ["casme2", "samm", "smic"],
            "from_video_patterns": [
                {"dataset": "casme3", "video_regex": "^CAS-\\d+$", "regex_flags": ["IGNORECASE"]},
                {"dataset": "samm", "video_regex": "^SAMM-\\d+$", "regex_flags": ["IGNORECASE"]},
            ],
            "fallback_dataset": "unknown",
        },
        "valid_labels": {
            "coarse": ["positive", "negative", "surprise"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate me_vqa_util/vqa_config_constants.json")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent.parent / "me_vqa_util" / "vqa_config_constants.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    cfg = _build_config()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
