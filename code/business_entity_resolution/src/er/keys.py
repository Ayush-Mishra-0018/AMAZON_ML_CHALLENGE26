"""Conjunctive blocking keys. Identity lives in *conjunctions* (measured: single tokens <=34% recall,
token-pair keys 98% at ~300 candidates/S1). Types: A = address-token pair, X = name-token x address-token,
N = name-token pair."""
import itertools
import numpy as np
from .textnorm import name_tokens, addr_tokens, h64

A, X, N = 0, 1, 2


def record_keys(name: str, addr: str):
    nt = name_tokens(name)
    at = addr_tokens(addr)
    hs, ts = [], []
    for a, b in itertools.combinations(at, 2):
        hs.append(h64("A|" + a + "|" + b)); ts.append(A)
    for a in nt:
        for b in at:
            hs.append(h64("X|" + a + "|" + b)); ts.append(X)
    for a, b in itertools.combinations(nt, 2):
        hs.append(h64("N|" + a + "|" + b)); ts.append(N)
    return hs, ts


def keys_for_records(names, addrs):
    """Return (hash uint64[], type int8[], record_local_index int32[]) for a batch of records."""
    H, T, R = [], [], []
    for i, (n, a) in enumerate(zip(names, addrs)):
        hs, ts = record_keys(n, a)
        H.extend(hs); T.extend(ts); R.extend([i] * len(hs))
    return np.array(H, dtype=np.uint64), np.array(T, dtype=np.int8), np.array(R, dtype=np.int32)
