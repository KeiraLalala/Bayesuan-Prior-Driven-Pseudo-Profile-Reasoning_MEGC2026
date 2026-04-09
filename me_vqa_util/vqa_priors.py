"""Build reference priors/statistics for ME-VQA profile generation."""

from collections import Counter, defaultdict

from .vqa_requirements import weighted_choice


def _init_stats():
    return {
        "coarse_counter": Counter(),
        "fine_counter": Counter(),
        "fine_given_coarse": defaultdict(Counter),
        "bundle_counter": Counter(),  # (coarse, fine, tuple(aus))
        "bundle_by_coarse": defaultdict(Counter),
        "bundle_by_pair": defaultdict(Counter),  # (coarse, fine) -> Counter(bundle)
        "au_bundle_counter": Counter(),  # tuple(aus)
        "au_bundle_by_pair": defaultdict(Counter),
        "au_bundle_by_coarse": defaultdict(Counter),
        "au_len_counter": Counter(),
        "au_token_counter": Counter(),
        "yes_no_by_au": defaultdict(Counter),
        "profiles": [],
    }


def _add_profile(stats, profile):
    coarse = profile.get("coarse")
    fine = profile.get("fine")
    aus = tuple(profile.get("aus", []) or [])
    bool_answers = profile.get("bool_answers", {}) or {}

    if coarse:
        stats["coarse_counter"][coarse] += 1
    if fine:
        stats["fine_counter"][fine] += 1
    if coarse and fine:
        stats["fine_given_coarse"][coarse][fine] += 1

    if coarse and fine:
        bundle = (coarse, fine, aus)
        stats["bundle_counter"][bundle] += 1
        stats["bundle_by_coarse"][coarse][bundle] += 1
        stats["bundle_by_pair"][(coarse, fine)][bundle] += 1

    if aus:
        stats["au_bundle_counter"][aus] += 1
        if coarse:
            stats["au_bundle_by_coarse"][coarse][aus] += 1
        if coarse and fine:
            stats["au_bundle_by_pair"][(coarse, fine)][aus] += 1
        stats["au_len_counter"][len(aus)] += 1
        for tok in aus:
            stats["au_token_counter"][tok] += 1

    for au, ans in bool_answers.items():
        if ans in {"yes", "no"}:
            stats["yes_no_by_au"][au][ans] += 1

    stats["profiles"].append(profile)


def build_priors(reference_profiles):
    stats_by_dataset = {}
    for dataset in reference_profiles.keys():
        stats_by_dataset[dataset] = _init_stats()
    all_stats = _init_stats()

    for dataset, ds_profiles in reference_profiles.items():
        for profile in ds_profiles.values():
            _add_profile(all_stats, profile)
            _add_profile(stats_by_dataset[dataset], profile)

    stats_by_dataset["__all__"] = all_stats
    return stats_by_dataset


def choose_stats(dataset, stats_by_dataset):
    if dataset in stats_by_dataset and stats_by_dataset[dataset]["profiles"]:
        return stats_by_dataset[dataset]
    return stats_by_dataset["__all__"]


def sample_coarse(stats, rng):
    return weighted_choice(stats["coarse_counter"], rng, default="negative")


def sample_fine_given_coarse(stats, coarse, rng):
    if coarse in stats["fine_given_coarse"] and stats["fine_given_coarse"][coarse]:
        return weighted_choice(stats["fine_given_coarse"][coarse], rng, default=None)
    return weighted_choice(stats["fine_counter"], rng, default="disgust")


def sample_bundle(stats, coarse, fine, rng):
    if (coarse, fine) in stats["bundle_by_pair"] and stats["bundle_by_pair"][(coarse, fine)]:
        return weighted_choice(stats["bundle_by_pair"][(coarse, fine)], rng, default=None)
    if coarse in stats["bundle_by_coarse"] and stats["bundle_by_coarse"][coarse]:
        return weighted_choice(stats["bundle_by_coarse"][coarse], rng, default=None)
    return weighted_choice(stats["bundle_counter"], rng, default=None)


def sample_au_bundle(stats, coarse, fine, rng):
    if (coarse, fine) in stats["au_bundle_by_pair"] and stats["au_bundle_by_pair"][(coarse, fine)]:
        return weighted_choice(stats["au_bundle_by_pair"][(coarse, fine)], rng, default=None)
    if coarse in stats["au_bundle_by_coarse"] and stats["au_bundle_by_coarse"][coarse]:
        return weighted_choice(stats["au_bundle_by_coarse"][coarse], rng, default=None)
    return weighted_choice(stats["au_bundle_counter"], rng, default=None)
