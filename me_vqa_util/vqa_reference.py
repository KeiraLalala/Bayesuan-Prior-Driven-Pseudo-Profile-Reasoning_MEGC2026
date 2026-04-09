"""Stage 1: reference profile construction from labeled rows."""

import re
from collections import Counter, defaultdict

from .vqa_requirements import (
    ANALYSIS_QUESTIONS,
    BOOL_AU_RE,
    COARSE_Q,
    COMBINED_AU_FINE_COARSE_Q,
    COMBINED_SINGLE_AU_FINE_COARSE_Q,
    FINE_Q,
    MULTI_AU_Q,
    SINGLE_AU_Q,
    VALID_COARSE_LABELS,
    canonicalize_au_tokens,
    get_reference_key,
    is_blank_answer,
    looks_like_analysis_text,
    normalize_label,
    split_au_tokens,
)


def _is_clean_short_label(value, max_words=3):
    s = normalize_label(value)
    if not s:
        return False
    if len(s) > 32:
        return False
    if looks_like_analysis_text(s):
        return False
    words = s.replace("-", " ").split()
    if len(words) > max_words:
        return False
    if any(any(ch.isdigit() for ch in w) for w in words):
        return False
    return True


def _extract_from_text(answer_text):
    s = str(answer_text or "").strip()
    if not s:
        return {"coarse": None, "fine": None, "aus": []}
    low = s.lower()

    fine = None
    coarse = None
    aus = []

    fine_pats = (
        r"fine-grained expression class is ([a-z\- ]+)",
        r"fine-grained label is ([a-z\- ]+)",
        r"suggests ([a-z\- ]+) and places",
        r"consistent with ([a-z\- ]+) and",
    )
    coarse_pats = (
        r"coarse expression class is ([a-z\- ]+)",
        r"coarse label is ([a-z\- ]+)",
        r"\b(positive|negative|surprise)\s+category",
    )
    for pat in fine_pats:
        m = re.search(pat, low)
        if m:
            fine = normalize_label(m.group(1))
            break
    for pat in coarse_pats:
        m = re.search(pat, low)
        if m:
            coarse = normalize_label(m.group(1))
            break

    m = re.search(r"action units present are:\s*(.+?)\.", low)
    if m:
        aus.extend(split_au_tokens(m.group(1)))
    m = re.search(r"action unit present is:\s*(.+?)\.", low)
    if m:
        aus.extend(split_au_tokens(m.group(1)))
    m = re.search(r"observed action unit:\s*(.+?)\.", low)
    if m:
        aus.extend(split_au_tokens(m.group(1)))
    m = re.search(r"^([a-z ,\-]+)\s+is observed", low)
    if m:
        aus.extend(split_au_tokens(m.group(1)))

    return {"coarse": coarse, "fine": fine, "aus": canonicalize_au_tokens(aus)}


def _collect_allowed_labels(train_rows):
    coarse_counter = Counter()
    fine_counter = Counter()
    for row in train_rows:
        q = row.get("question")
        ans = row.get("answer")
        if is_blank_answer(ans):
            continue
        if q == COARSE_Q:
            v = normalize_label(ans)
            if _is_clean_short_label(v, max_words=2) and v in VALID_COARSE_LABELS:
                coarse_counter[v] += 1
        elif q == FINE_Q:
            v = normalize_label(ans)
            if _is_clean_short_label(v, max_words=3):
                fine_counter[v] += 1
    coarse_allowed = set(coarse_counter)
    if not coarse_allowed:
        coarse_allowed = set(VALID_COARSE_LABELS)
    fine_allowed = set(fine_counter)
    return coarse_allowed, fine_allowed


def _choose_label(candidates, allowed_set=None):
    cnt = Counter()
    for value in candidates:
        v = normalize_label(value)
        if not v:
            continue
        if allowed_set is not None and v not in allowed_set:
            continue
        cnt[v] += 1
    if not cnt:
        return None
    return cnt.most_common(1)[0][0]


def build_reference_profiles(train_rows):
    coarse_allowed, fine_allowed = _collect_allowed_labels(train_rows)

    grouped = defaultdict(list)
    for row in train_rows:
        grouped[get_reference_key(row)].append(row)

    profiles = defaultdict(dict)
    for (dataset, image_id), rows in grouped.items():
        coarse_candidates = []
        fine_candidates = []
        au_tokens = []
        bool_answers = {}
        analysis_texts = []

        for row in rows:
            q = str(row.get("question", "")).strip()
            ans = row.get("answer")
            if is_blank_answer(ans):
                continue
            ans_s = str(ans).strip()

            if q == COARSE_Q:
                v = normalize_label(ans_s)
                if _is_clean_short_label(v, max_words=2):
                    coarse_candidates.append(v)
                continue

            if q == FINE_Q:
                v = normalize_label(ans_s)
                if _is_clean_short_label(v, max_words=3):
                    fine_candidates.append(v)
                continue

            if q == SINGLE_AU_Q or q == MULTI_AU_Q:
                au_tokens.extend(split_au_tokens(ans_s))
                continue

            m = BOOL_AU_RE.match(q)
            if m:
                au_q = normalize_label(m.group(1))
                a = normalize_label(ans_s)
                if a in {"yes", "no"}:
                    bool_answers[au_q] = a
                    if a == "yes":
                        au_tokens.append(au_q)
                continue

            if q in ANALYSIS_QUESTIONS and normalize_label(ans_s) not in {"no", "unknown", "n/a"}:
                analysis_texts.append(ans_s)

            if q == COMBINED_AU_FINE_COARSE_Q or q == COMBINED_SINGLE_AU_FINE_COARSE_Q or q in ANALYSIS_QUESTIONS:
                parsed = _extract_from_text(ans_s)
                if parsed["coarse"]:
                    coarse_candidates.append(parsed["coarse"])
                if parsed["fine"]:
                    fine_candidates.append(parsed["fine"])
                if parsed["aus"]:
                    au_tokens.extend(parsed["aus"])

        coarse = _choose_label(coarse_candidates, allowed_set=coarse_allowed)
        fine = _choose_label(fine_candidates, allowed_set=fine_allowed)
        aus = canonicalize_au_tokens(au_tokens)

        profiles[dataset][image_id] = {
            "dataset": dataset,
            "image_id": image_id,
            "coarse": coarse,
            "fine": fine,
            "aus": aus,
            "bool_answers": dict(bool_answers),
            "analysis_examples": analysis_texts,
            "has_core_labels": coarse is not None and fine is not None,
        }
    return profiles

