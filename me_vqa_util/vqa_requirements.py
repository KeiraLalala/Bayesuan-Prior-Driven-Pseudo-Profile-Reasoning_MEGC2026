"""Stage 3: target requirement extraction and shared schema utilities."""

import hashlib
import json
import random
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional


CONFIG_JSON_PATH = Path(__file__).with_name("vqa_config_constants.json")


def _load_config():
    if not CONFIG_JSON_PATH.exists():
        raise FileNotFoundError(
            f"Missing generated constants file: {CONFIG_JSON_PATH}. "
            "Run `python util/generate_vqa_constants_json.py` first."
        )
    with CONFIG_JSON_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _compile_regex(pattern, flags):
    re_flags = 0
    for flag_name in flags:
        re_flags |= getattr(re, flag_name)
    return re.compile(pattern, re_flags)


_CONFIG = _load_config()
_Q = _CONFIG["question_templates"]
COARSE_Q = _Q["coarse"]
FINE_Q = _Q["fine"]
SINGLE_AU_Q = _Q["single_au"]
MULTI_AU_Q = _Q["multi_au"]

ANALYSIS_QUESTIONS = set(_CONFIG["analysis_questions"])

_BOOL_AU = _CONFIG["regex"]["bool_au"]
BOOL_AU_RE = _compile_regex(_BOOL_AU["pattern"], _BOOL_AU.get("flags", []))

_COMBINED = _CONFIG["combined_reference_questions"]
COMBINED_AU_FINE_COARSE_Q = _COMBINED["multi_au_fine_coarse"]
COMBINED_SINGLE_AU_FINE_COARSE_Q = _COMBINED["single_au_fine_coarse"]

_DATASETS = _CONFIG["datasets"]
TARGET_DATASETS = tuple(_DATASETS["target"])
REFERENCE_DATASETS = tuple(_DATASETS["reference"])
DATASET_FROM_VIDEO_PATTERNS = tuple(
    (item["dataset"], _compile_regex(item["video_regex"], item.get("regex_flags", [])))
    for item in _DATASETS["from_video_patterns"]
)
FALLBACK_DATASET = _DATASETS.get("fallback_dataset", "unknown")

VALID_COARSE_LABELS = set(_CONFIG["valid_labels"]["coarse"])


def normalize_space(text):
    return re.sub(r"\s+", " ", str(text or "").strip())


def is_blank_answer(value):
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def stable_rng(seed, *parts):
    key = "|".join([str(seed)] + [str(p) for p in parts])
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def weighted_choice(counter, rng, default=None):
    if not counter:
        return default
    items = list(counter.items())
    total = 0.0
    for _, w in items:
        try:
            ww = float(w)
        except (TypeError, ValueError):
            ww = 0.0
        if ww > 0:
            total += ww
    if total <= 0:
        return items[0][0]
    point = rng.uniform(0.0, total)
    acc = 0.0
    for key, weight in items:
        try:
            ww = float(weight)
        except (TypeError, ValueError):
            ww = 0.0
        if ww <= 0:
            continue
        acc += ww
        if point <= acc:
            return key
    return items[-1][0]


def choose_random(items, rng):
    if not items:
        return None
    return items[rng.randrange(len(items))]


def split_au_tokens(value):
    if value is None:
        return []
    s = normalize_space(str(value).lower())
    if not s:
        return []

    # Remove known scaffolding fragments if they are embedded in answers.
    s = re.sub(r"\bthe action units present are\s*:\s*", "", s)
    s = re.sub(r"\bthe action unit present is\s*:\s*", "", s)
    s = re.sub(r"\bobserved action unit\s*:\s*", "", s)
    s = re.sub(r"\baction unit[s]?\s*:\s*", "", s)

    s = s.replace("|", ",").replace(";", ",")
    s = re.sub(r"\band\b", ",", s)
    s = re.sub(r"\s*/\s*", ",", s)
    parts = [normalize_space(tok) for tok in s.split(",")]

    out = []
    seen = set()
    for tok in parts:
        tok = re.sub(r"^au\s*\d+\s*[:\-]?\s*", "", tok)
        tok = tok.strip(" .-_")
        if not tok:
            continue
        if tok in {"n/a", "na", "none", "null", "no", "unknown"}:
            continue
        if len(tok) > 64:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


def canonicalize_au_tokens(tokens):
    out = []
    seen = set()
    for tok in tokens or []:
        t = normalize_space(str(tok).lower())
        if not t:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def canonicalize_au_answer(tokens):
    vals = canonicalize_au_tokens(tokens)
    return ", ".join(vals)


def looks_like_analysis_text(value):
    s = normalize_space(value).lower()
    if not s:
        return False
    if len(s) > 40:
        return True
    markers = ("therefore", "observed", "action unit", "coarse expression class", "fine-grained expression")
    return any(m in s for m in markers)


def normalize_label(value):
    s = normalize_space(value).lower()
    s = s.strip(" .,:;!?\"'")
    return s


def detect_dataset_from_video(video):
    v = str(video or "").strip()
    if not v:
        return None
    for ds, pat in DATASET_FROM_VIDEO_PATTERNS:
        if pat.fullmatch(v):
            return ds
    return None


def detect_dataset(row, fallback=None):
    ds = str(row.get("dataset", "")).strip().lower()
    if ds:
        return ds

    merged = " ".join(
        [
            str(row.get("video", "")),
            str(row.get("video_id", "")),
            str(row.get("image_id", "")),
            str(row.get("id", "")),
        ]
    ).lower()

    if "casme3" in merged:
        return "casme3"
    if "casme2" in merged:
        return "casme2"
    if "samm" in merged:
        return "samm"
    if "smic" in merged:
        return "smic"

    by_video = detect_dataset_from_video(row.get("video", ""))
    if by_video:
        return by_video
    if fallback:
        return fallback
    return FALLBACK_DATASET


def get_reference_key(row):
    ds = detect_dataset(row)
    image_id = str(row.get("image_id") or row.get("video") or row.get("video_id") or "").strip()
    return ds, image_id


def get_target_key(row, dataset_hint=None):
    ds = detect_dataset(row, fallback=dataset_hint)
    video = str(row.get("video") or row.get("video_id") or "").strip()
    return ds, video


def group_by_video(rows, dataset_hint=None):
    grouped = OrderedDict()
    for idx, row in enumerate(rows):
        key = get_target_key(row, dataset_hint=dataset_hint)
        grouped.setdefault(key, []).append(idx)
    return grouped


def parse_question(question):
    q = normalize_space(question)
    if q == COARSE_Q:
        return {"kind": "coarse"}
    if q == FINE_Q:
        return {"kind": "fine"}
    if q == SINGLE_AU_Q:
        return {"kind": "single_au"}
    if q == MULTI_AU_Q:
        return {"kind": "multi_au"}
    if q in ANALYSIS_QUESTIONS:
        return {"kind": "analysis"}

    m = BOOL_AU_RE.match(q)
    if m:
        return {"kind": "bool_au", "au": normalize_space(m.group(1)).lower()}

    q_low = q.lower()
    if "analyse the micro-expression" in q_low or "analysis of the micro-expression" in q_low:
        return {"kind": "analysis"}
    return {"kind": "unknown"}


def parse_question_target(question: str, qwen_bridge: Optional[Any] = None) -> Dict[str, Any]:
    """Target-side parser with optional Qwen assist and rule fallback."""
    if qwen_bridge is None:
        return parse_question(question)
    return qwen_bridge.parse_mevqa_question(question, fallback_parser=parse_question)
