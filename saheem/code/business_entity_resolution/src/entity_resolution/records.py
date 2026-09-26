"""Compact normalized records shared by blocking, features, and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .blocking import lsh_bands
from .contracts import SourceRecord
from .normalization import address_views, name_views

_LIST_SEPARATOR = "\x1f"


def _pack(values: tuple[str, ...]) -> str:
    return _LIST_SEPARATOR.join(values)


def _unpack(value: str) -> tuple[str, ...]:
    return tuple(item for item in value.split(_LIST_SEPARATOR) if item) if value else ()


def _last_alpha_token(value: str) -> str:
    for token in reversed(value.split()):
        if len(token) >= 3 and any(character.isalpha() for character in token):
            return token
    return ""


@dataclass(frozen=True, slots=True)
class PreparedRecord:
    entity_id: str
    source: str
    country: str
    name_norm: str
    name_fold: str
    name_core: str
    address_norm: str
    address_fold: str
    address_canon: str
    name_digits: tuple[str, ...]
    address_digits: tuple[str, ...]
    scripts: tuple[str, ...]
    name_bands: tuple[int, int, int, int]
    address_bands: tuple[int, int, int, int]
    address_first_number: str
    address_last_number: str
    address_tail: str

    @classmethod
    def from_source(cls, record: SourceRecord) -> "PreparedRecord":
        names = name_views(record.business_name)
        addresses = address_views(record.business_address)
        source = record.entity_id.split("-", 1)[0]
        return cls(
            entity_id=record.entity_id,
            source=source,
            country=record.country,
            name_norm=names.normalized,
            name_fold=names.latin_folded,
            name_core=names.core,
            address_norm=addresses.normalized,
            address_fold=addresses.latin_folded,
            address_canon=addresses.canonical_tokens,
            name_digits=names.numeric_tokens,
            address_digits=addresses.numeric_tokens,
            scripts=names.scripts,
            name_bands=lsh_bands(names.core or names.normalized),
            address_bands=lsh_bands(addresses.canonical_tokens),
            address_first_number=addresses.numeric_tokens[0] if addresses.numeric_tokens else "",
            address_last_number=addresses.numeric_tokens[-1] if addresses.numeric_tokens else "",
            address_tail=_last_alpha_token(addresses.canonical_tokens),
        )

    @classmethod
    def from_sqlite_row(cls, row: Any) -> "PreparedRecord":
        return cls(
            entity_id=row["entity_id"],
            source=row["source"],
            country=row["country"],
            name_norm=row["name_norm"],
            name_fold=row["name_fold"],
            name_core=row["name_core"],
            address_norm=row["address_norm"],
            address_fold=row["address_fold"],
            address_canon=row["address_canon"],
            name_digits=_unpack(row["name_digits"]),
            address_digits=_unpack(row["address_digits"]),
            scripts=_unpack(row["scripts"]),
            name_bands=tuple(row[f"name_lsh{index}"] for index in range(4)),  # type: ignore[arg-type]
            address_bands=tuple(row[f"address_lsh{index}"] for index in range(4)),  # type: ignore[arg-type]
            address_first_number=row["address_first_number"],
            address_last_number=row["address_last_number"],
            address_tail=row["address_tail"],
        )

    def sqlite_values(self) -> tuple[object, ...]:
        return (
            self.entity_id,
            self.source,
            self.country,
            self.name_norm,
            self.name_fold,
            self.name_core,
            self.address_norm,
            self.address_fold,
            self.address_canon,
            _pack(self.name_digits),
            _pack(self.address_digits),
            _pack(self.scripts),
            *self.name_bands,
            *self.address_bands,
            self.address_first_number,
            self.address_last_number,
            self.address_tail,
        )
