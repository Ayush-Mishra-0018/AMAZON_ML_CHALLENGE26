"""Pair features. Every feature is computable at test time from the two records + retrieval statistics only.

Deliberately EXCLUDED: absolute S1-name-frequency features. They helped India (+0.009 F0.5 in a leave-name-group-out
CV) but the test S1 is a different-sized sample (US dup-name rate 0.29 vs 0.36 in train), i.e. a measured shift risk.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from .textnorm import norm, skeleton, is_non_latin, looks_like_domain, number_set

NAME_F = ["nr", "nts", "npr", "nso", "nsk", "nskts", "lnn", "nonlat", "dom"]
ADDR_F = ["ats", "ar", "apr", "numj", "numeq", "aempty", "lna"]
KEY_F = ["sA", "sX", "sN", "nA", "nX", "nN", "ksum", "kshare", "kgap", "krank", "ncand_rec",
         "mindf_rel", "mdA_rel", "mdX_rel", "mdN_rel"]
STAGE1 = NAME_F + ADDR_F + ["isS2"] + KEY_F


class Records:
    """Normalised, array-backed view of one side (S1 or pool) so pair features are pure array lookups."""

    def __init__(self, df: pd.DataFrame):
        self.ids = df.entity_id.values
        self.nn = [norm(x) for x in df.business_name]
        self.na = [norm(x) for x in df.business_address]
        self.sk = [skeleton(x) for x in self.nn]
        self.nums = [number_set(x) for x in self.na]
        self.nonlat = np.array([is_non_latin(x) for x in df.business_name])
        self.dom = np.array([looks_like_domain(x) for x in df.business_name])
        self.lnn = np.array([len(x) for x in self.nn], dtype=np.float32)
        self.lna = np.array([len(x) for x in self.na], dtype=np.float32)
        self.empty_a = self.lna == 0
        self.isS2 = (df.src.values == "S2") if "src" in df else np.zeros(len(df), bool)


def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.uint8).astype(np.float32)


def _take(lst, idx):
    return [lst[i] for i in idx]


def pair_features(pairs: pd.DataFrame, S1: Records, P: Records, cap: int) -> pd.DataFrame:
    a, b = pairs.s1_i.values, pairs.pool_i.values
    an, bn, aa, ba = _take(S1.nn, a), _take(P.nn, b), _take(S1.na, a), _take(P.na, b)
    f = pd.DataFrame(index=pairs.index)
    f["nr"] = _cp(an, bn, fuzz.ratio)
    f["nts"] = _cp(an, bn, fuzz.token_set_ratio)
    f["npr"] = _cp(an, bn, fuzz.partial_ratio)
    f["nso"] = _cp(an, bn, fuzz.token_sort_ratio)
    ask, bsk = _take(S1.sk, a), _take(P.sk, b)
    f["nsk"] = _cp(ask, bsk, fuzz.ratio)
    f["nskts"] = _cp(ask, bsk, fuzz.token_set_ratio)
    f["lnn"] = P.lnn[b]
    f["nonlat"] = P.nonlat[b].astype(np.float32)
    f["dom"] = P.dom[b].astype(np.float32)
    empty = P.empty_a[b]
    for nm, sc in (("ats", fuzz.token_set_ratio), ("ar", fuzz.ratio), ("apr", fuzz.partial_ratio)):
        v = _cp(aa, ba, sc)
        v[empty] = -1
        f[nm] = v
    nj = np.full(len(a), -1, np.float32)
    ne = np.full(len(a), -1, np.float32)
    for i, (x, y) in enumerate(zip(_take(S1.nums, a), _take(P.nums, b))):
        if x and y:
            nj[i] = len(x & y) / len(x | y)
            ne[i] = float(x == y)
    f["numj"], f["numeq"] = nj, ne
    f["aempty"] = empty.astype(np.float32)
    f["lna"] = P.lna[b]
    f["isS2"] = P.isS2[b].astype(np.float32)
    for c in ["sA", "sX", "sN", "nA", "nX", "nN", "ksum", "kshare", "kgap", "krank", "ncand_rec"]:
        f[c] = pairs[c].values.astype(np.float32)
    f["mindf_rel"] = np.minimum(pairs.mindf.values, 9999) / cap
    for t in "AXN":
        f[f"md{t}_rel"] = np.minimum(pairs[f"md{t}"].values, 9999) / cap
    return f[STAGE1]
