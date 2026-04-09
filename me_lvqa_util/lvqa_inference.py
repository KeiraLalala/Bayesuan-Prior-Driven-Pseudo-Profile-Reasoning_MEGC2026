"""Stage 4: pseudo-profile construction and answer decoding."""

from collections import Counter

from util.io_utils import load_jsonl, save_jsonl
from .lvqa_priors import choose_stats, sample_micro_given_total, sample_total
from .lvqa_requirements import (
    MACRO_LABEL,
    MICRO_LABEL,
    canonicalize_au_answer,
    choose_random,
    get_video_key,
    group_by_video,
    is_blank_answer,
    normalize_event_label,
    parse_question_type,
    stable_rng,
    weighted_choice,
)


def enforce_sequence_counts(sequence, total, target_micro, protected_indices=None):
    protected = set(protected_indices or [])
    total = max(1, int(total))
    target_micro = max(0, min(int(target_micro), total))

    for idx in range(1, total + 1):
        lbl = normalize_event_label(sequence.get(idx))
        if lbl is None:
            sequence[idx] = MACRO_LABEL
        else:
            sequence[idx] = lbl

    micro_idx = [i for i in range(1, total + 1) if sequence[i] == MICRO_LABEL]
    macro_idx = [i for i in range(1, total + 1) if sequence[i] == MACRO_LABEL]

    if len(micro_idx) > target_micro:
        flips = len(micro_idx) - target_micro
        ordered = [i for i in reversed(micro_idx) if i not in protected] + [i for i in reversed(micro_idx) if i in protected]
        for idx in ordered[:flips]:
            sequence[idx] = MACRO_LABEL
    elif len(micro_idx) < target_micro:
        flips = target_micro - len(micro_idx)
        ordered = [i for i in reversed(macro_idx) if i not in protected] + [i for i in reversed(macro_idx) if i in protected]
        for idx in ordered[:flips]:
            sequence[idx] = MICRO_LABEL

    return sequence


def choose_au(stats, total, micro, rng):
    cand = None
    if total in stats["au_by_total"] and stats["au_by_total"][total]:
        cand = weighted_choice(stats["au_by_total"][total], rng, default=None)
    if not cand and micro in stats["au_by_micro"] and stats["au_by_micro"][micro]:
        cand = weighted_choice(stats["au_by_micro"][micro], rng, default=None)
    if not cand and stats["au_counter"]:
        cand = weighted_choice(stats["au_counter"], rng, default=None)

    if cand:
        return canonicalize_au_answer(cand)

    target_len = weighted_choice(stats["au_len_counter"], rng, default=3)
    try:
        target_len = int(target_len)
    except (TypeError, ValueError):
        target_len = 3
    target_len = max(1, min(target_len, 12))

    tokens = []
    token_counter = Counter(stats["au_token_counter"])
    while token_counter and len(tokens) < target_len:
        tok = weighted_choice(token_counter, rng, default=None)
        if tok is None:
            break
        tokens.append(tok)
        token_counter.pop(tok, None)
    return ", ".join(tokens)


def build_video_profile(dataset, video, video_rows, reference_profiles, stats_by_dataset, seed, question_parser=None):
    stats = choose_stats(dataset, stats_by_dataset)
    rng = stable_rng(seed, dataset, video)
    parser = question_parser or parse_question_type

    q_needed = []
    for row in video_rows:
        q_info = parser(row.get("question", ""))
        if q_info.get("kind") == "nth_type" and q_info.get("index"):
            q_needed.append(int(q_info["index"]))
    max_needed = max(q_needed) if q_needed else 1

    exact = reference_profiles.get(dataset, {}).get(video)

    reference_candidates = [p for p in stats["profiles"] if p.get("total") is not None]
    large = [p for p in reference_candidates if int(p.get("total", 0)) >= max_needed]
    if large:
        reference_candidates = large
    reference = exact if exact else choose_random(reference_candidates, rng)

    total = None
    micro = None
    macro = None
    sequence = {}
    if reference:
        total = reference.get("total")
        micro = reference.get("micro")
        macro = reference.get("macro")
        sequence.update(reference.get("sequence", {}) or {})

    if total is None and micro is not None and macro is not None:
        total = int(micro) + int(macro)
    if total is None:
        total = sample_total(stats, rng, minimum_total=max_needed)
    total = max(int(total), int(max_needed), 1)

    if micro is None and macro is None:
        ratio_hint = None
        if reference and reference.get("total") and reference.get("micro") is not None:
            t0 = int(reference["total"])
            if t0 > 0:
                ratio_hint = float(reference["micro"]) / float(t0)
        micro = sample_micro_given_total(stats, total, rng, ratio_hint=ratio_hint)
        macro = total - micro
    elif micro is None:
        macro = max(0, min(int(macro), total))
        micro = total - macro
    elif macro is None:
        micro = max(0, min(int(micro), total))
        macro = total - micro
    else:
        micro = max(0, min(int(micro), total))
        macro = max(0, min(int(macro), total))
        if micro + macro != total:
            macro = max(0, total - micro)
            if micro + macro != total:
                micro = max(0, total - macro)

    for i in range(1, total + 1):
        if i in sequence and normalize_event_label(sequence[i]) is not None:
            sequence[i] = normalize_event_label(sequence[i])
            continue
        pri = stats["type_by_index"].get(i) or stats["global_type"]
        sequence[i] = weighted_choice(pri, rng, default=MACRO_LABEL) or MACRO_LABEL
        if sequence[i] not in (MICRO_LABEL, MACRO_LABEL):
            sequence[i] = MACRO_LABEL

    sequence = enforce_sequence_counts(sequence, total, micro, protected_indices=q_needed)

    au_answer = ""
    if reference and reference.get("au"):
        au_answer = canonicalize_au_answer(reference.get("au"))
    if not au_answer:
        au_answer = choose_au(stats, total, micro, rng)

    return {
        "dataset": dataset,
        "video": video,
        "total": int(total),
        "micro": int(micro),
        "macro": int(total - micro),
        "sequence": sequence,
        "au": au_answer,
    }


def answer_question(row, profile, stats, question_parser=None):
    parser = question_parser or parse_question_type
    q_info = parser(row.get("question", ""))
    kind = q_info.get("kind")

    as_string = stats.get("count_as_string", False)
    if kind == "total":
        val = int(profile.get("total", 0))
        return str(val) if as_string else val
    if kind == "micro":
        val = int(profile.get("micro", 0))
        return str(val) if as_string else val
    if kind == "macro":
        val = int(profile.get("macro", 0))
        return str(val) if as_string else val
    if kind == "nth_type":
        idx = q_info.get("index")
        if idx is None:
            return MACRO_LABEL
        return profile.get("sequence", {}).get(int(idx), MACRO_LABEL)
    if kind == "au":
        return canonicalize_au_answer(profile.get("au", ""))

    original = row.get("answer")
    if not is_blank_answer(original):
        return original
    return ""


def fill_target_file(target_path, output_path, dataset_hint, reference_profiles, stats_by_dataset, seed, question_parser=None):
    rows = load_jsonl(target_path)
    grouped = group_by_video(rows, dataset_hint=dataset_hint)

    profile_cache = {}
    for (dataset, video), idxs in grouped.items():
        video_rows = [rows[i] for i in idxs]
        profile_cache[(dataset, video)] = build_video_profile(
            dataset=dataset,
            video=video,
            video_rows=video_rows,
            reference_profiles=reference_profiles,
            stats_by_dataset=stats_by_dataset,
            seed=seed,
            question_parser=question_parser,
        )

    out_rows = []
    for row in rows:
        dataset, video = get_video_key(row, dataset_hint=dataset_hint)
        stats = choose_stats(dataset, stats_by_dataset)
        profile = profile_cache.get((dataset, video), {"total": 1, "micro": 0, "macro": 1, "sequence": {}, "au": ""})
        out = dict(row)
        out["answer"] = answer_question(row=row, profile=profile, stats=stats, question_parser=question_parser)
        out_rows.append(out)

    save_jsonl(output_path, out_rows)
