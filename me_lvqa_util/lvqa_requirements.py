"""Stage 3: target requirement extraction and shared schema utilities."""

import hashlib
import json
import random
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional


CONFIG_JSON_PATH = Path(__file__).with_name("config_constants.json")


def _load_config():
    if not CONFIG_JSON_PATH.exists():
        raise FileNotFoundError(
            f"Missing generated constants file: {CONFIG_JSON_PATH}. "
            "Run `python util/generate_lvqa_constants_json.py` first."
        )
    with CONFIG_JSON_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _compile_regex(pattern, flags):
    re_flags = 0
    for flag_name in flags:
        re_flags |= getattr(re, flag_name)
    return re.compile(pattern, re_flags)


_CONFIG = _load_config()

MICRO_LABEL = _CONFIG["labels"]["micro_expression"]
MACRO_LABEL = _CONFIG["labels"]["macro_expression"]
DATASETS = tuple(_CONFIG["datasets"])

_RULES = _CONFIG["dataset_detection"]["rules"]
FALLBACK_DATASET = _CONFIG["dataset_detection"].get("fallback_dataset", "unknown")
SAMM_VIDEO_RE = _compile_regex(_RULES["samm"]["video_regex"], _RULES["samm"].get("regex_flags", []))
CASME3_VIDEO_RE = _compile_regex(_RULES["casme3"]["video_regex"], _RULES["casme3"].get("regex_flags", []))
CASME3_VIDEO_PREFIXES = tuple(_RULES["casme3"].get("video_prefixes", []))

ORDINAL_WORD_TO_INT = {str(k): int(v) for k, v in _CONFIG["ordinal_word_to_int"]["mapping"].items()}


def normalize_space(text):
    return re.sub(r"\s+", " ", str(text or "").strip())


def is_blank_answer(value):
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def parse_int(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if re.fullmatch(r"-?\d+", s):
            return int(s)
        m = re.search(r"-?\d+", s)
        if m:
            return int(m.group(0))
    return None


def normalize_event_label(value):
    if value is None:
        return None
    s = normalize_space(str(value).lower())
    if not s:
        return None
    if "micro" in s:
        return MICRO_LABEL
    if "macro" in s:
        return MACRO_LABEL
    return None


def split_au_tokens(value):
    if value is None:
        return []
    s = normalize_space(str(value).lower())
    if not s:
        return []
    s = s.replace("|", ",").replace(";", ",")
    s = re.sub(r"\band\b", ",", s)
    s = re.sub(r"\s*/\s*", ",", s)
    raw = [normalize_space(tok) for tok in s.split(",")]
    cleaned = []
    seen = set()
    for tok in raw:
        tok = re.sub(r"^au\s*\d+\s*[:\-]?\s*", "", tok)
        tok = tok.strip(" .-_")
        if not tok:
            continue
        if tok in {"none", "n/a", "na", "null", "no action units"}:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        cleaned.append(tok)
    return cleaned


def canonicalize_au_answer(value):
    tokens = split_au_tokens(value)
    return ", ".join(tokens)


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


def stable_rng(seed, *parts):
    key = "|".join([str(seed)] + [str(p) for p in parts])
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def detect_dataset_from_video(video):
    v = str(video or "").strip()
    if not v:
        return None
    if SAMM_VIDEO_RE.fullmatch(v):
        return "samm"
    v_low = v.lower()
    if CASME3_VIDEO_RE.fullmatch(v) or any(v_low.startswith(prefix.lower()) for prefix in CASME3_VIDEO_PREFIXES):
        return "casme3"
    return None


def detect_dataset(row, fallback=None):
    parts = [
        str(row.get("dataset", "")),
        str(row.get("video_id", "")),
        str(row.get("video", "")),
        str(row.get("source", "")),
    ]
    merged = " ".join(parts).lower()
    if "casme3" in merged or re.search(r"\bcasme\s*3\b", merged):
        return "casme3"
    if "samm" in merged:
        return "samm"

    by_video = detect_dataset_from_video(row.get("video", ""))
    if by_video:
        return by_video

    if fallback:
        return fallback
    return FALLBACK_DATASET


def get_video_key(row, dataset_hint=None):
    dataset = detect_dataset(row, fallback=dataset_hint)
    video = str(row.get("video") or row.get("video_id") or "").strip()
    return dataset, video


def group_by_video(rows, dataset_hint=None):
    grouped = OrderedDict()
    for idx, row in enumerate(rows):
        key = get_video_key(row, dataset_hint=dataset_hint)
        grouped.setdefault(key, []).append(idx)
    return grouped


def extract_ordinal_index(question):
    q = normalize_space(str(question).lower())
    patterns = [
        r"\bthe\s+(\d+)\s*-\s*(?:st|nd|rd|th)\b",
        r"\b(\d+)\s*-\s*(?:st|nd|rd|th)\b",
        r"\bthe\s+(\d+)(?:st|nd|rd|th)\b",
        r"\b(\d+)(?:st|nd|rd|th)\b",
        r"\bexpression\s+event\s+(\d+)\b",
        r"\bevent\s+number\s+(\d+)\b",
        r"\bno\.?\s*(\d+)\b",
    ]
    for pat in patterns:
        m = re.search(pat, q)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                continue
    for word, idx in ORDINAL_WORD_TO_INT.items():
        if re.search(rf"\b{re.escape(word)}\b", q):
            return idx
    return None


def parse_question_type(question):
    q = normalize_space(str(question).lower())
    q_clean = re.sub(r"[^a-z0-9\-\s]", " ", q)
    q_clean = normalize_space(q_clean)

    if ("action unit" in q_clean) or re.search(r"\b(?:aus|au)\b", q_clean):
        if any(t in q_clean for t in ("distinct", "appear", "list", "which", "what")):
            return {"kind": "au"}

    idx = extract_ordinal_index(q_clean)
    type_markers = (
        "expression type",
        "what type",
        "whether",
        "is the",
        "which type",
    )
    if idx is not None and "expression" in q_clean:
        if any(m in q_clean for m in type_markers) or ("micro" in q_clean and "macro" in q_clean):
            return {"kind": "nth_type", "index": idx}

    count_markers = ("how many", "number of", "count of", "total number")
    if "expression" in q_clean and any(m in q_clean for m in count_markers):
        if "micro" in q_clean:
            return {"kind": "micro"}
        if "macro" in q_clean:
            return {"kind": "macro"}
        return {"kind": "total"}

    if "total" in q_clean and "expression" in q_clean and "micro" not in q_clean and "macro" not in q_clean:
        return {"kind": "total"}

    return {"kind": "unknown"}


def parse_question_type_target(question: str, qwen_bridge: Optional[Any] = None) -> Dict[str, Any]:
    """Target-side parser with optional Qwen assist and rule fallback."""
    if qwen_bridge is None:
        return parse_question_type(question)
    return qwen_bridge.parse_lvqa_question(question, fallback_parser=parse_question_type)
