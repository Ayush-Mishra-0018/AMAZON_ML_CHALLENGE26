"""Stage-2 relational (S2<->S3 consensus) features.

Measured on held-out S1 groups: adding them to stage-1 probabilities raised F0.5 by +0.007 (US) and +0.019 (India);
listwise-only features (rank/max/sum of p within an S1) gave ~0; one-owner argmax gave ~0 (74% of FPs are orphans).
Idea: a true member's record agrees with OTHER confident members (independent noise), a decoy is a deterministic
near-copy of the S1 record and does not.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from .features import Records, _cp, _take

REL_F = ["sup", "cross", "bestn", "besta", "numagree", "nconf_others"]


def relational_features(rows: pd.DataFrame, P: Records, conf_p: float = 0.5) -> pd.DataFrame:
    """rows: columns s1_i, pool_i, p1, isS2 (index must be unique). Returns REL_F aligned to rows.index."""
    out = pd.DataFrame(0.0, index=rows.index, columns=REL_F, dtype=np.float32)
    conf = rows.loc[rows.p1 >= conf_p, ["s1_i", "pool_i", "p1", "isS2"]]
    if conf.empty:
        return out
    r = rows[["s1_i", "pool_i", "isS2"]].copy()
    r["row"] = rows.index
    m = r.merge(conf.rename(columns={"pool_i": "pool_m", "p1": "p_m", "isS2": "isS2_m"}), on="s1_i")
    nconf = conf.groupby("s1_i").size()
    out["nconf_others"] = (rows.s1_i.map(nconf).fillna(0).values - (rows.p1.values >= conf_p)).astype(np.float32)
    m = m[m.pool_i.values != m.pool_m.values]
    if m.empty:
        return out
    bi, bm = m.pool_i.values, m.pool_m.values
    sn = _cp(_take(P.nn, bi), _take(P.nn, bm), fuzz.token_set_ratio)
    sa = _cp(_take(P.na, bi), _take(P.na, bm), fuzz.token_set_ratio)
    both = ~(P.empty_a[bi] | P.empty_a[bm])
    sa = np.where(both, sa, 0.0)
    na = np.array([float(bool(x) and bool(y) and x == y) for x, y in zip(_take(P.nums, bi), _take(P.nums, bm))], np.float32)
    sup = (sa >= 90) & (sn >= 80)
    m = m.assign(bestn=sn * m.p_m.values, besta=sa * m.p_m.values, sup=sup.astype(np.float32),
                 cross=(sup & (m.isS2.values != m.isS2_m.values)).astype(np.float32), numagree=na)
    g = m.groupby("row").agg(sup=("sup", "sum"), cross=("cross", "sum"), bestn=("bestn", "max"),
                             besta=("besta", "max"), numagree=("numagree", "sum"))
    out.loc[g.index, g.columns] = g.values.astype(np.float32)
    return out
