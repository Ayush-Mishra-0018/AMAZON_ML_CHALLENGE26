"""Deterministic approximate blocking keys used by the SQLite candidate index."""

from __future__ import annotations

import hashlib

EXACT_NAME = 1 << 0
EXACT_FOLDED_NAME = 1 << 1
EXACT_CORE_NAME = 1 << 2
EXACT_ADDRESS = 1 << 3
NAME_LSH = 1 << 4
ADDRESS_LSH = 1 << 5
ADDRESS_NUMBER_TAIL = 1 << 6
ADDRESS_LAST_NUMBER = 1 << 7
POSITIVE_INJECTION = 1 << 8

CHANNEL_NAMES = {
    EXACT_NAME: "exact_name",
    EXACT_FOLDED_NAME: "exact_folded_name",
    EXACT_CORE_NAME: "exact_core_name",
    EXACT_ADDRESS: "exact_address",
    NAME_LSH: "name_lsh",
    ADDRESS_LSH: "address_lsh",
    ADDRESS_NUMBER_TAIL: "address_number_tail",
    ADDRESS_LAST_NUMBER: "address_last_number",
    POSITIVE_INJECTION: "positive_injection",
}


def character_ngrams(value: str, size: int = 3) -> tuple[str, ...]:
    """Return unique padded character n-grams for approximate text blocking."""

    compact = " ".join(value.split())
    if not compact:
        return ()
    padded = f"^{compact}$"
    if len(padded) <= size:
        return (padded,)
    return tuple(sorted({padded[index : index + size] for index in range(len(padded) - size + 1)}))


def _hash64(value: str) -> int:
    digest = hashlib.blake2b(
        value.encode("utf-8"),
        digest_size=8,
        person=b"amazonER",
    ).digest()
    return int.from_bytes(digest, "big", signed=False)


def simhash64(value: str) -> int:
    """Create a 64-bit locality-sensitive fingerprint from character trigrams."""

    grams = character_ngrams(value)
    if not grams:
        return 0
    accumulators = [0] * 64
    for gram in grams:
        hashed = _hash64(gram)
        for bit in range(64):
            accumulators[bit] += 1 if hashed & (1 << bit) else -1
    fingerprint = 0
    for bit, value_at_bit in enumerate(accumulators):
        if value_at_bit >= 0:
            fingerprint |= 1 << bit
    return fingerprint


def lsh_bands(value: str) -> tuple[int, int, int, int]:
    """Split SimHash into four indexed 16-bit bands."""

    fingerprint = simhash64(value)
    return tuple((fingerprint >> shift) & 0xFFFF for shift in (0, 16, 32, 48))  # type: ignore[return-value]


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def channel_names(mask: int) -> tuple[str, ...]:
    return tuple(name for bit, name in CHANNEL_NAMES.items() if mask & bit)
