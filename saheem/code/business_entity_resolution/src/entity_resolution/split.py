"""Stable entity-level split assignment."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SplitFractions:
    train: float = 0.8
    calibration: float = 0.1
    holdout: float = 0.1

    def __post_init__(self) -> None:
        values = (self.train, self.calibration, self.holdout)
        if any(value < 0 for value in values):
            raise ValueError("split fractions cannot be negative")
        if abs(sum(values) - 1.0) > 1e-12:
            raise ValueError("split fractions must sum to 1")


def match_count_bin(match_count: int) -> str:
    if match_count < 0:
        raise ValueError("match count cannot be negative")
    if match_count == 0:
        return "0"
    if match_count == 1:
        return "1"
    if match_count <= 3:
        return "2-3"
    if match_count <= 5:
        return "4-5"
    return "6+"


def assign_split(
    source1_entity_id: str,
    *,
    seed: int = 2026,
    fractions: SplitFractions = SplitFractions(),
    group_key: str | None = None,
) -> str:
    """Assign an entity or near-duplicate group reproducibly without RNG state."""

    key = group_key or source1_entity_id
    payload = f"{seed}:{key}".encode("utf-8")
    integer = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    value = integer / 2**64
    if value < fractions.train:
        return "train"
    if value < fractions.train + fractions.calibration:
        return "calibration"
    return "holdout"
