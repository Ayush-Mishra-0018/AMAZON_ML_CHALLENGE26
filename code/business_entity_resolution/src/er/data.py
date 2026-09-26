"""Loading, caching, and building self-consistent training universes."""
import numpy as np
import pandas as pd
from .config import WORK_DIR, find_data_dir
from .progress import stage, log

COLS = ["entity_id", "business_name", "business_address", "country"]


def _read(path):
    # explicit tab separator and no quote handling (names may contain quotes); empty strings stay empty
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)


def load_split(split: str):
    """Return (s1, pool, gt). pool = S2 + S3 with a ``src`` column ('S2'/'S3'). gt is None for test."""
    cache = WORK_DIR / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    d = find_data_dir() / split
    out = {}
    names = ("source1", "source2", "source3") + (("ground_truth",) if split == "train" else ())
    for k, n in enumerate(names, 1):
        pq = cache / f"{split}_{n}.parquet"
        tsv = d / f"{split}_{n}.tsv"
        if not pq.exists():
            with stage(f"[{split} {k}/{len(names)}] first run: converting {tsv.name} ({tsv.stat().st_size / 1e6:.0f} MB) to parquet cache"):
                _read(tsv).to_parquet(pq)
        with stage(f"[{split} {k}/{len(names)}] loading {pq.name}", 30):
            out[n] = pd.read_parquet(pq)
        log(f"   {n}: {len(out[n]):,} rows")
    s2, s3 = out["source2"].assign(src="S2"), out["source3"].assign(src="S3")
    pool = pd.concat([s2, s3], ignore_index=True)
    return out["source1"], pool, out.get("ground_truth")


def owner_map(gt: pd.DataFrame) -> pd.Series:
    """pool entity_id -> owning S1 id. (Observed in train: every matched S2/S3 id has exactly one owner.)"""
    g = gt[gt.matched_entity_ids != ""]
    ex = pd.DataFrame({"s1": g.source1_entity_id.values, "m": g.matched_entity_ids.str.split(",").values}).explode("m")
    assert not ex.m.duplicated().any(), "one-owner property violated in labels"
    return pd.Series(ex.s1.values, index=ex.m.values)


def make_universe(s1, pool, gt, country: str, n_s1: int, seed: int = 0):
    """Sample ``n_s1`` S1 entities of a country plus ALL their true S2/S3 records plus a proportional share of
    orphan records. Cluster-closed, so labels stay complete. (Caveat: same-name confounders are diluted vs full scale.)"""
    log(f"   building {country} training universe (cluster-closed sample of {n_s1:,} S1 + all their records + proportional orphans)")
    rng = np.random.RandomState(seed)
    s1c = s1[s1.country == country]
    n_s1 = min(n_s1, len(s1c))
    pick = s1c.iloc[rng.choice(len(s1c), n_s1, replace=False)].reset_index(drop=True)
    own = owner_map(gt)
    pc = pool[pool.country == country]
    owner = pc.entity_id.map(own)
    keep_matched = owner.isin(set(pick.entity_id)).values
    orphan = owner.isna().values
    frac = n_s1 / len(s1c)
    keep_orphan = orphan & (rng.rand(len(pc)) < frac)
    upool = pc[keep_matched | keep_orphan].reset_index(drop=True)
    truth = {e: set() for e in pick.entity_id}
    for m, s in own[own.isin(set(pick.entity_id))].items():
        truth[s].add(m)
    return pick, upool, truth
