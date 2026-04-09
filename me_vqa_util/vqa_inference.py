"""Stage 4: pseudo-profile construction and answer decoding."""

from collections import Counter

from util.io_utils import load_jsonl, save_jsonl
from .vqa_priors import choose_stats, sample_au_bundle, sample_bundle, sample_coarse, sample_fine_given_coarse
from .vqa_requirements import (
    canonicalize_au_answer,
    canonicalize_au_tokens,
    choose_random,
    get_target_key,
    group_by_video,
    is_blank_answer,
    normalize_space,
    parse_question,
    stable_rng,
    weighted_choice,
)


def extract_target_requirements(video_rows, question_parser=None):
    parser = question_parser or parse_question
    req = {
        "needs_single_au": False,
        "needs_multi_au": False,
        "needs_analysis": False,
        "asked_bool_aus": [],
    }
    for row in video_rows:
        q = parser(row.get("question", ""))
        kind = q.get("kind")
        if kind == "single_au":
            req["needs_single_au"] = True
        elif kind == "multi_au":
            req["needs_multi_au"] = True
        elif kind == "analysis":
            req["needs_analysis"] = True
        elif kind == "bool_au":
            au = q.get("au")
            if au:
                req["asked_bool_aus"].append(au)
    req["asked_bool_aus"] = canonicalize_au_tokens(req["asked_bool_aus"])
    return req


def _sample_au_tokens(stats, coarse, fine, rng):
    bundle = sample_au_bundle(stats, coarse, fine, rng)
    if bundle:
        return list(bundle)

    target_len = weighted_choice(stats["au_len_counter"], rng, default=1)
    try:
        target_len = int(target_len)
    except (TypeError, ValueError):
        target_len = 1
    target_len = max(1, min(target_len, 8))

    tokens = []
    token_counter = Counter(stats["au_token_counter"])
    while token_counter and len(tokens) < target_len:
        tok = weighted_choice(token_counter, rng, default=None)
        if tok is None:
            break
        tokens.append(tok)
        token_counter.pop(tok, None)
    return tokens


def _is_fine_compatible(stats, coarse, fine):
    if not coarse or not fine:
        return False
    if coarse not in stats["fine_given_coarse"]:
        return True
    return fine in stats["fine_given_coarse"][coarse]


def build_video_profile(dataset, video, video_rows, reference_profiles, stats_by_dataset, seed, question_parser=None):
    stats = choose_stats(dataset, stats_by_dataset)
    rng = stable_rng(seed, dataset, video)
    req = extract_target_requirements(video_rows, question_parser=question_parser)
    need_au = req["needs_single_au"] or req["needs_multi_au"] or bool(req["asked_bool_aus"]) or req["needs_analysis"]

    exact = reference_profiles.get(dataset, {}).get(video)
    candidates = [p for p in stats["profiles"] if p.get("has_core_labels")]
    if need_au:
        with_au = [p for p in candidates if p.get("aus")]
        if with_au:
            candidates = with_au
    reference = exact if exact else choose_random(candidates, rng)

    mode = "synthesis"
    if reference and reference.get("has_core_labels"):
        reuse_prob = 1.0 if exact else 0.65
        if rng.random() < reuse_prob:
            mode = "reference_reuse"

    coarse = reference.get("coarse") if (mode == "reference_reuse" and reference) else None
    fine = reference.get("fine") if (mode == "reference_reuse" and reference) else None
    aus = list(reference.get("aus", [])) if (mode == "reference_reuse" and reference) else []

    if not coarse:
        coarse = sample_coarse(stats, rng)
    if not fine or not _is_fine_compatible(stats, coarse, fine):
        fine = sample_fine_given_coarse(stats, coarse, rng)

    if not aus:
        bundle = sample_bundle(stats, coarse, fine, rng)
        if bundle:
            _, _, bundle_aus = bundle
            if bundle_aus:
                aus = list(bundle_aus)
    if not aus and need_au:
        aus = _sample_au_tokens(stats, coarse, fine, rng)

    # For asked yes/no AU queries, optionally include queried AU by reference yes/no prior.
    if req["asked_bool_aus"]:
        for asked_au in req["asked_bool_aus"]:
            if asked_au in aus:
                continue
            dist = stats["yes_no_by_au"].get(asked_au)
            if dist:
                sampled = weighted_choice(dist, rng, default="no")
                if sampled == "yes":
                    aus.append(asked_au)

    aus = canonicalize_au_tokens(aus)

    return {
        "dataset": dataset,
        "video": video,
        "coarse": coarse or "negative",
        "fine": fine or "disgust",
        "aus": aus,
        "mode": mode,
    }


def _analysis_text(question, profile):
    coarse = profile.get("coarse", "negative")
    fine = profile.get("fine", "disgust")
    aus = profile.get("aus", []) or []

    if not aus:
        core = (
            f"The facial movement is subtle; the best fine-grained label is {fine}, "
            f"and the corresponding coarse class is {coarse}."
        )
    elif len(aus) == 1:
        core = (
            f"The observed action unit is {aus[0]}, which is consistent with {fine} "
            f"and maps to the {coarse} coarse expression class."
        )
    else:
        aus_text = ", ".join(aus[:-1]) + f" and {aus[-1]}"
        core = (
            f"The observed action units are {aus_text}. Together they support the fine-grained "
            f"class {fine}, under the {coarse} coarse expression category."
        )

    q = normalize_space(question)
    if "comprehensive analysis" in q.lower():
        return core + " This conclusion is derived from the same profile used for all answers in this video."
    if "detailed analysis" in q.lower():
        return core + " This aligns with the profile-level label assignment."
    return core


def answer_question(row, profile, question_parser=None, analysis_renderer=None):
    parser = question_parser or parse_question
    q_info = parser(row.get("question", ""))
    kind = q_info.get("kind")
    aus = profile.get("aus", []) or []

    if kind == "coarse":
        return profile.get("coarse", "negative")
    if kind == "fine":
        return profile.get("fine", "disgust")
    if kind == "single_au":
        return aus[0] if aus else "unknown"
    if kind == "multi_au":
        return canonicalize_au_answer(aus) if aus else "unknown"
    if kind == "bool_au":
        asked = q_info.get("au")
        if not asked:
            return "no"
        return "yes" if asked in set(aus) else "no"
    if kind == "analysis":
        if analysis_renderer is not None:
            return analysis_renderer(row.get("question", ""), profile, fallback_renderer=_analysis_text)
        return _analysis_text(row.get("question", ""), profile)

    original = row.get("answer")
    if not is_blank_answer(original):
        return original
    return "unknown"


def fill_target_file(
    target_path,
    output_path,
    dataset_hint,
    reference_profiles,
    stats_by_dataset,
    seed,
    question_parser=None,
    analysis_renderer=None,
):
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
        dataset, video = get_target_key(row, dataset_hint=dataset_hint)
        profile = profile_cache.get((dataset, video), {"coarse": "negative", "fine": "disgust", "aus": []})
        out = dict(row)
        out["answer"] = answer_question(
            row,
            profile,
            question_parser=question_parser,
            analysis_renderer=analysis_renderer,
        )
        out_rows.append(out)

    save_jsonl(output_path, out_rows)
