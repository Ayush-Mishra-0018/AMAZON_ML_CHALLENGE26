"""Interpretable pair features and the pre-model candidate ranking heuristic."""

from __future__ import annotations

from difflib import SequenceMatcher

from .blocking import (
    ADDRESS_LAST_NUMBER,
    ADDRESS_LSH,
    ADDRESS_NUMBER_TAIL,
    EXACT_ADDRESS,
    EXACT_CORE_NAME,
    EXACT_FOLDED_NAME,
    EXACT_NAME,
    NAME_LSH,
)
from .records import PreparedRecord

FEATURE_NAMES = (
    "name_exact",
    "name_fold_exact",
    "name_core_exact",
    "name_ratio",
    "name_core_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_token_jaccard",
    "name_token_containment",
    "name_length_ratio",
    "address_exact",
    "address_fold_exact",
    "address_ratio",
    "address_token_sort_ratio",
    "address_token_set_ratio",
    "address_token_jaccard",
    "address_token_containment",
    "address_length_ratio",
    "address_missing_left",
    "address_missing_right",
    "name_digit_exact",
    "name_digit_overlap",
    "address_digit_exact",
    "address_digit_overlap",
    "address_digit_conflict",
    "script_overlap",
    "same_source_s2",
    "channel_exact_name",
    "channel_exact_address",
    "channel_name_lsh",
    "channel_address_lsh",
    "channel_number_block",
    "channel_count",
)


def _ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    try:
        from rapidfuzz import fuzz

        return fuzz.ratio(left, right) / 100.0
    except ImportError:
        return SequenceMatcher(None, left, right).ratio()


def _token_sort_ratio(left: str, right: str) -> float:
    return _ratio(" ".join(sorted(left.split())), " ".join(sorted(right.split())))


def _token_set_ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    try:
        from rapidfuzz import fuzz

        return fuzz.token_set_ratio(left, right) / 100.0
    except ImportError:
        left_tokens, right_tokens = set(left.split()), set(right.split())
        shared = left_tokens & right_tokens
        if not shared:
            return 0.0
        return _ratio(
            " ".join(sorted(shared | (left_tokens - right_tokens))),
            " ".join(sorted(shared | (right_tokens - left_tokens))),
        )


def _set_scores(left: tuple[str, ...] | set[str], right: tuple[str, ...] | set[str]) -> tuple[float, float]:
    left_set, right_set = set(left), set(right)
    if not left_set or not right_set:
        return 0.0, 0.0
    intersection = len(left_set & right_set)
    jaccard = intersection / len(left_set | right_set)
    containment = intersection / min(len(left_set), len(right_set))
    return jaccard, containment


def _length_ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return min(len(left), len(right)) / max(len(left), len(right))


def _digit_features(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[float, float, float]:
    left_set, right_set = set(left), set(right)
    if not left_set or not right_set:
        return 0.0, 0.0, 0.0
    exact = float(left_set == right_set)
    overlap = len(left_set & right_set) / len(left_set | right_set)
    conflict = float(not bool(left_set & right_set))
    return exact, overlap, conflict


def pair_features(query: PreparedRecord, target: PreparedRecord, channel_mask: int) -> tuple[float, ...]:
    name_tokens_left = tuple(query.name_norm.split())
    name_tokens_right = tuple(target.name_norm.split())
    address_tokens_left = tuple(query.address_canon.split())
    address_tokens_right = tuple(target.address_canon.split())
    name_jaccard, name_containment = _set_scores(name_tokens_left, name_tokens_right)
    address_jaccard, address_containment = _set_scores(
        address_tokens_left,
        address_tokens_right,
    )
    name_digit_exact, name_digit_overlap, _ = _digit_features(
        query.name_digits,
        target.name_digits,
    )
    address_digit_exact, address_digit_overlap, address_digit_conflict = _digit_features(
        query.address_digits,
        target.address_digits,
    )
    script_jaccard, _ = _set_scores(query.scripts, target.scripts)
    number_channels = ADDRESS_NUMBER_TAIL | ADDRESS_LAST_NUMBER
    return (
        float(bool(query.name_norm) and query.name_norm == target.name_norm),
        float(bool(query.name_fold) and query.name_fold == target.name_fold),
        float(bool(query.name_core) and query.name_core == target.name_core),
        _ratio(query.name_fold, target.name_fold),
        _ratio(query.name_core, target.name_core),
        _token_sort_ratio(query.name_fold, target.name_fold),
        _token_set_ratio(query.name_fold, target.name_fold),
        name_jaccard,
        name_containment,
        _length_ratio(query.name_norm, target.name_norm),
        float(bool(query.address_canon) and query.address_canon == target.address_canon),
        float(bool(query.address_fold) and query.address_fold == target.address_fold),
        _ratio(query.address_fold, target.address_fold),
        _token_sort_ratio(query.address_fold, target.address_fold),
        _token_set_ratio(query.address_fold, target.address_fold),
        address_jaccard,
        address_containment,
        _length_ratio(query.address_canon, target.address_canon),
        float(not query.address_canon),
        float(not target.address_canon),
        name_digit_exact,
        name_digit_overlap,
        address_digit_exact,
        address_digit_overlap,
        address_digit_conflict,
        script_jaccard,
        float(target.source == "S2"),
        float(bool(channel_mask & (EXACT_NAME | EXACT_FOLDED_NAME | EXACT_CORE_NAME))),
        float(bool(channel_mask & EXACT_ADDRESS)),
        float(bool(channel_mask & NAME_LSH)),
        float(bool(channel_mask & ADDRESS_LSH)),
        float(bool(channel_mask & number_channels)),
        float(channel_mask.bit_count()),
    )


def heuristic_score(features: tuple[float, ...]) -> float:
    """Rank a broad blocking pool before the learned model sees final candidates."""

    values = dict(zip(FEATURE_NAMES, features, strict=True))
    name_score = max(
        values["name_ratio"],
        values["name_core_ratio"],
        values["name_token_set_ratio"],
    )
    address_score = max(
        values["address_ratio"],
        values["address_token_set_ratio"],
    )
    address_available = not (
        values["address_missing_left"] or values["address_missing_right"]
    )
    if address_available:
        score = 0.57 * name_score + 0.33 * address_score
    else:
        score = 0.82 * name_score
    score += 0.06 * values["channel_exact_name"]
    score += 0.04 * values["channel_exact_address"]
    score += 0.03 * min(values["channel_count"], 3.0) / 3.0
    score += 0.04 * values["address_digit_exact"]
    score -= 0.16 * values["address_digit_conflict"]
    return max(0.0, min(1.0, score))
