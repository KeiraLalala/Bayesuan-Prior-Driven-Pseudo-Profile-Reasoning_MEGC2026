"""Build train-derived priors and provide sampling helpers."""

from collections import Counter, defaultdict

from .lvqa_requirements import (
    DATASETS,
    MACRO_LABEL,
    canonicalize_au_answer,
    detect_dataset,
    normalize_event_label,
    parse_question_type,
    split_au_tokens,
    weighted_choice,
)


def _init_dataset_stats():
    return {
        "total_counter": Counter(),
        "micro_given_total": defaultdict(Counter),
        "type_by_index": defaultdict(Counter),
        "global_type": Counter(),
        "au_counter": Counter(),
        "au_len_counter": Counter(),
        "au_token_counter": Counter(),
        "au_by_total": defaultdict(Counter),
        "au_by_micro": defaultdict(Counter),
        "profiles": [],
        "count_as_string": False,
    }


def _add_profile_to_stats(stats, profile):
    total = profile.get("total")
    micro = profile.get("micro")
    macro = profile.get("macro")
    seq = profile.get("sequence", {}) or {}
    au = canonicalize_au_answer(profile.get("au", ""))

    if total is not None:
        total = int(total)
        stats["total_counter"][total] += 1
        if micro is not None:
            m = max(0, min(int(micro), total))
            stats["micro_given_total"][total][m] += 1
        elif macro is not None:
            m = max(0, min(total - int(macro), total))
            stats["micro_given_total"][total][m] += 1

    if seq:
        for idx, lbl in seq.items():
            norm = normalize_event_label(lbl)
            if norm is None:
                continue
            ii = int(idx)
            stats["type_by_index"][ii][norm] += 1
            stats["global_type"][norm] += 1

    if au:
        stats["au_counter"][au] += 1
        tokens = split_au_tokens(au)
        if tokens:
            stats["au_len_counter"][len(tokens)] += 1
            for tok in tokens:
                stats["au_token_counter"][tok] += 1
        if total is not None:
            stats["au_by_total"][int(total)][au] += 1
        if micro is not None:
            stats["au_by_micro"][int(micro)][au] += 1

    stats["profiles"].append(profile)


def build_dataset_statistics(reference_profiles, train_rows):
    stats_by_dataset = {ds: _init_dataset_stats() for ds in DATASETS}
    all_stats = _init_dataset_stats()

    for ds in DATASETS:
        for profile in reference_profiles.get(ds, {}).values():
            _add_profile_to_stats(stats_by_dataset[ds], profile)
            _add_profile_to_stats(all_stats, profile)

    numeric_mode = {ds: Counter() for ds in DATASETS}
    for row in train_rows:
        ds = detect_dataset(row)
        if ds not in DATASETS:
            continue
        kind = parse_question_type(row.get("question", "")).get("kind")
        if kind not in {"total", "micro", "macro"}:
            continue
        ans = row.get("answer")
        if isinstance(ans, str):
            numeric_mode[ds]["str"] += 1
        elif isinstance(ans, (int, float)):
            numeric_mode[ds]["num"] += 1

    for ds in DATASETS:
        stats = stats_by_dataset[ds]
        stats["count_as_string"] = numeric_mode[ds]["str"] > numeric_mode[ds]["num"]
        if not stats["total_counter"]:
            stats["total_counter"][1] = 1
        if not stats["global_type"]:
            stats["global_type"][MACRO_LABEL] = 1
        if not stats["au_len_counter"]:
            stats["au_len_counter"][3] = 1

    if not all_stats["total_counter"]:
        all_stats["total_counter"][1] = 1
    if not all_stats["global_type"]:
        all_stats["global_type"][MACRO_LABEL] = 1
    if not all_stats["au_len_counter"]:
        all_stats["au_len_counter"][3] = 1

    stats_by_dataset["__all__"] = all_stats
    return stats_by_dataset


def choose_stats(dataset, stats_by_dataset):
    if dataset in stats_by_dataset:
        return stats_by_dataset[dataset]
    return stats_by_dataset["__all__"]


def sample_total(stats, rng, minimum_total=1):
    pool = Counter()
    for total, w in stats["total_counter"].items():
        if int(total) >= int(minimum_total):
            pool[int(total)] += w
    if not pool:
        chosen = weighted_choice(stats["total_counter"], rng, default=minimum_total)
        return max(int(minimum_total), int(chosen or minimum_total))
    chosen = weighted_choice(pool, rng, default=minimum_total)
    return max(int(minimum_total), int(chosen or minimum_total))


def sample_micro_given_total(stats, total, rng, ratio_hint=None):
    total = max(1, int(total))
    if total in stats["micro_given_total"] and stats["micro_given_total"][total]:
        micro = weighted_choice(stats["micro_given_total"][total], rng, default=0)
        return max(0, min(int(micro), total))

    if stats["micro_given_total"]:
        nearest = min(stats["micro_given_total"].keys(), key=lambda t: abs(int(t) - total))
        near_micro = weighted_choice(stats["micro_given_total"][nearest], rng, default=0)
        if int(nearest) > 0:
            ratio = float(near_micro) / float(nearest)
        else:
            ratio = ratio_hint if ratio_hint is not None else 0.2
    else:
        ratio = ratio_hint if ratio_hint is not None else 0.2

    micro = int(round(total * ratio))
    return max(0, min(micro, total))
