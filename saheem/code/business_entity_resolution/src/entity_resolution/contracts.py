"""File schemas and immutable row types used throughout the pipeline."""

from __future__ import annotations

from dataclasses import dataclass

SOURCE_COLUMNS = (
    "entity_id",
    "business_name",
    "business_address",
    "country",
)
GROUND_TRUTH_COLUMNS = ("source1_entity_id", "matched_entity_ids")
MATCHING_COLUMNS = ("source1_entity_id", "matched_entity_ids")
CANDIDATE_COLUMNS = ("source1_entity_id", "candidate_entity_ids")


@dataclass(frozen=True, slots=True)
class SourceRecord:
    entity_id: str
    business_name: str
    business_address: str
    country: str


@dataclass(frozen=True, slots=True)
class IdListRow:
    source1_entity_id: str
    entity_ids: frozenset[str]
