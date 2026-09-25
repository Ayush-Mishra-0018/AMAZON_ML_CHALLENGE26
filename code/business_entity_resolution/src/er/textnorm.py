"""Text normalisation. Pure functions, no external data; only the installed ``text-unidecode`` table."""
import re
import zlib
from text_unidecode import unidecode

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_DIGITS = re.compile(r"\d+")
_DOMAIN = re.compile(r"\.(?:com|net|org|in|fr|co|io)\b|^[#@]")


def norm(s: str) -> str:
    """Lower-case, transliterate to ASCII (also romanises Indic scripts), drop punctuation."""
    return _NON_ALNUM.sub(" ", unidecode(s).lower()).strip()


def is_non_latin(s: str) -> bool:
    return any(ord(c) > 0x24F and c.isalpha() for c in s)


def looks_like_domain(s: str) -> bool:
    return bool(_DOMAIN.search(s.lower()))


def skeleton_token(tok: str) -> str:
    """Phonetic skeleton: measured AUC 0.994 (vs 0.970 raw) separating true from random non-Latin names."""
    s = tok.replace("ph", "f").replace("w", "v").replace("z", "j").replace("x", "ks").replace("c", "k").replace("q", "k")
    t = s[0] + re.sub(r"[aeiouyh]", "", s[1:])
    return re.sub(r"(.)\1+", r"\1", t)


def skeleton(normed: str) -> str:
    return " ".join(skeleton_token(t) for t in normed.split() if t)


def name_tokens(name: str) -> list:
    """Word tokens + joined-prefix token (catches 'prairiefive.com' vs 'Prairie Five') + phonetic tokens."""
    n = norm(name)
    t = [x for x in n.split() if len(x) > 1 or x.isdigit()]
    out = set(t)
    out.add("~" + "".join(n.split())[:8])
    for x in t:
        if x.isalpha() and len(x) >= 3:
            out.add("sk" + skeleton_token(x))
    return sorted(out)


def addr_tokens(addr: str) -> list:
    """Word tokens; numbers lose leading zeros (0011118 == 11118) and also emit a 2-digit suffix token
    (5844 ~ 844 ~ 44: house-number corruption is a measured miss category)."""
    out = set()
    for x in norm(addr).split():
        if len(x) < 2 and not x.isdigit():
            continue
        if x.isdigit():
            x = x.lstrip("0") or "0"
            out.add(x)
            if len(x) >= 2:
                out.add("s2" + x[-2:])
        else:
            out.add(x)
    return sorted(out)


def number_set(normed_addr: str) -> frozenset:
    return frozenset((x.lstrip("0") or "0") for x in _DIGITS.findall(normed_addr))


def h64(s: str) -> int:
    """Process-independent 64-bit hash from two seeded crc32 (python's hash() is salted per process)."""
    b = s.encode()
    return zlib.crc32(b) | (zlib.crc32(b, 0x9E3779B9) << 32)
