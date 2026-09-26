"""Information-preserving text views for names and addresses."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass

_SPACE_RE = re.compile(r"\s+")
_NUMBER_RE = re.compile(r"(?<!\w)\d+[a-z]*(?!\w)", flags=re.IGNORECASE)

# Only trailing phrases are removed, and the full normalized name is always retained.
_LEGAL_SUFFIXES = tuple(
    sorted(
        {
            ("and", "co"),
            ("and", "company"),
            ("private", "limited"),
            ("pvt", "ltd"),
            ("co",),
            ("company",),
            ("corp",),
            ("corporation",),
            ("inc",),
            ("incorporated",),
            ("limited",),
            ("llc",),
            ("llp",),
            ("ltd",),
            ("plc",),
            ("pvt",),
            ("sas",),
            ("sarl",),
        },
        key=len,
        reverse=True,
    )
)

_ADDRESS_CANONICAL = {
    "apt": "apartment",
    "apartment": "apartment",
    "ave": "avenue",
    "avenue": "avenue",
    "bldg": "building",
    "building": "building",
    "blvd": "boulevard",
    "boulevard": "boulevard",
    "dr": "drive",
    "drive": "drive",
    "fl": "floor",
    "floor": "floor",
    "hwy": "highway",
    "highway": "highway",
    "ln": "lane",
    "lane": "lane",
    "no": "number",
    "rd": "road",
    "road": "road",
    "ste": "suite",
    "suite": "suite",
    "st": "street",
    "street": "street",
}

_KNOWN_SCRIPTS = (
    "ARABIC",
    "BENGALI",
    "CYRILLIC",
    "DEVANAGARI",
    "GUJARATI",
    "GURMUKHI",
    "KANNADA",
    "LATIN",
    "MALAYALAM",
    "TAMIL",
    "TELUGU",
)


def _letters_numbers_marks_or_space(text: str) -> str:
    characters: list[str] = []
    for character in text:
        category = unicodedata.category(character)
        if category[0] in {"L", "N", "M"}:
            characters.append(character)
        else:
            characters.append(" ")
    return _SPACE_RE.sub(" ", "".join(characters)).strip()


def normalize_text(value: str, *, ampersand_as_and: bool = True) -> str:
    """Apply reversible-enough Unicode and separator normalization.

    NFKC unifies compatibility characters and ``casefold`` handles Unicode case.
    Letters, numbers, and combining marks from every script are retained.
    """

    value = unicodedata.normalize("NFKC", value or "").casefold()
    if ampersand_as_and:
        value = value.replace("&", " and ")
    return _letters_numbers_marks_or_space(value)


def fold_latin_diacritics(value: str) -> str:
    """Remove accents from Latin letters without stripping marks from other scripts."""

    output: list[str] = []
    for character in value:
        decomposed = unicodedata.normalize("NFD", character)
        base = decomposed[0] if decomposed else character
        if "LATIN" in unicodedata.name(base, ""):
            output.extend(part for part in decomposed if not unicodedata.combining(part))
        else:
            output.append(character)
    return unicodedata.normalize("NFC", "".join(output))


def strip_legal_suffix(normalized_name: str) -> str:
    tokens = normalized_name.split()
    changed = True
    while tokens and changed:
        changed = False
        for suffix in _LEGAL_SUFFIXES:
            length = len(suffix)
            if tuple(tokens[-length:]) == suffix:
                del tokens[-length:]
                changed = True
                break
    core = " ".join(tokens).strip()
    return core or normalized_name


def script_flags(value: str) -> tuple[str, ...]:
    scripts: set[str] = set()
    for character in value:
        if not character.isalpha():
            continue
        name = unicodedata.name(character, "")
        matched = next((script for script in _KNOWN_SCRIPTS if script in name), None)
        scripts.add(matched or "OTHER")
    return tuple(sorted(scripts))


def numeric_tokens(normalized_value: str) -> tuple[str, ...]:
    return tuple(_NUMBER_RE.findall(normalized_value))


@dataclass(frozen=True, slots=True)
class NameViews:
    normalized: str
    latin_folded: str
    core: str
    token_sorted: str
    numeric_tokens: tuple[str, ...]
    scripts: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AddressViews:
    normalized: str
    latin_folded: str
    canonical_tokens: str
    token_sorted: str
    numeric_tokens: tuple[str, ...]
    scripts: tuple[str, ...]
    missing: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def name_views(value: str) -> NameViews:
    normalized = normalize_text(value)
    return NameViews(
        normalized=normalized,
        latin_folded=fold_latin_diacritics(normalized),
        core=strip_legal_suffix(normalized),
        token_sorted=" ".join(sorted(normalized.split())),
        numeric_tokens=numeric_tokens(normalized),
        scripts=script_flags(value),
    )


def address_views(value: str) -> AddressViews:
    normalized = normalize_text(value)
    canonical = " ".join(_ADDRESS_CANONICAL.get(token, token) for token in normalized.split())
    return AddressViews(
        normalized=normalized,
        latin_folded=fold_latin_diacritics(normalized),
        canonical_tokens=canonical,
        token_sorted=" ".join(sorted(canonical.split())),
        numeric_tokens=numeric_tokens(normalized),
        scripts=script_flags(value),
        missing=not bool(normalized),
    )
