"""Stage 1: reference profile construction from answered rows."""

from collections import Counter, defaultdict

from .lvqa_requirements import (
    DATASETS,
    canonicalize_au_answer,
    get_video_key,
    is_blank_answer,
    normalize_event_label,
    parse_int,
    parse_question_type,
)


def extract_video_profile_from_answers(rows, dataset_hint=None):
    if not rows:
        return {
            "dataset": dataset_hint or "unknown",
            "video": "",
            "total": None,
            "micro": None,
            "macro": None,
            "sequence": {},
            "au": "",
        }

    dataset, video = get_video_key(rows[0], dataset_hint=dataset_hint)
    total = None
    micro = None
    macro = None
    sequence = {}
    au_candidates = Counter()

    for row in rows:
        q_info = parse_question_type(row.get("question", ""))
        kind = q_info.get("kind")
        answer = row.get("answer")
        if is_blank_answer(answer):
            continue

        if kind == "total":
            val = parse_int(answer)
            if val is not None:
                total = val
        elif kind == "micro":
            val = parse_int(answer)
            if val is not None:
                micro = val
        elif kind == "macro":
            val = parse_int(answer)
            if val is not None:
                macro = val
        elif kind == "nth_type":
            idx = q_info.get("index")
            lbl = normalize_event_label(answer)
            if idx and lbl:
                sequence[int(idx)] = lbl
        elif kind == "au":
            au = canonicalize_au_answer(answer)
            if au:
                au_candidates[au] += 1

    if total is None and micro is not None and macro is not None:
        total = micro + macro
    if total is None and sequence:
        total = max(sequence.keys())
    if total is not None:
        total = max(1, int(total))

    if micro is None and total is not None and macro is not None:
        micro = max(0, total - int(macro))
    if macro is None and total is not None and micro is not None:
        macro = max(0, total - int(micro))

    au = ""
    if au_candidates:
        au = au_candidates.most_common(1)[0][0]

    return {
        "dataset": dataset,
        "video": video,
        "total": total,
        "micro": micro if micro is None else int(micro),
        "macro": macro if macro is None else int(macro),
        "sequence": dict(sequence),
        "au": au,
    }


def build_reference_profiles(train_rows):
    grouped = defaultdict(list)
    for row in train_rows:
        ds, video = get_video_key(row)
        grouped[(ds, video)].append(row)

    profiles = {ds: {} for ds in DATASETS}
    for (dataset, video), rows in grouped.items():
        if dataset not in DATASETS:
            continue
        profiles[dataset][video] = extract_video_profile_from_answers(rows, dataset_hint=dataset)
    return profiles

