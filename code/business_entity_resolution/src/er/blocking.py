"""Record-centric blocking.

Index the (smaller, deduplicated) S1 side once as sorted 64-bit key hashes; stream every S2/S3 record against it.
Each S2/S3 record keeps its top-K S1 candidates ranked by conjunctive-key score. The per-key S1 document
frequency is measured exactly (no sampling); keys shared by more than ``cap`` S1 records are ignored.
"""
import multiprocessing as mp
import numpy as np
import pandas as pd
from .config import Params, WORK_DIR
from .progress import Progress, stage, log
from .keys import keys_for_records

_H = _I = None


def _init(hpath, ipath):
    global _H, _I
    _H = np.load(hpath, mmap_mode="r")
    _I = np.load(ipath, mmap_mode="r")


def _index_task(args):
    off, names, addrs = args
    H, T, R = keys_for_records(names, addrs)
    return H, (R + off).astype(np.int32)


def _scan_task(args):
    off, names, addrs, cap, topk = args
    H, T, R = keys_for_records(names, addrs)
    if len(H) == 0:
        return None
    lo = np.searchsorted(_H, H, "left")
    hi = np.searchsorted(_H, H, "right")
    df = hi - lo
    m = (df >= 1) & (df <= cap)
    if not m.any():
        return None
    lo, df, T, R = lo[m], df[m], T[m], R[m]
    tot = int(df.sum())
    rep_r = np.repeat(R, df)
    rep_t = np.repeat(T, df)
    rep_df = np.repeat(df, df)
    pos = np.repeat(lo, df) + (np.arange(tot) - np.repeat(np.cumsum(df) - df, df))
    s1i = np.asarray(_I[pos])
    d = pd.DataFrame({"r": rep_r, "s": s1i, "t": rep_t, "df": rep_df})
    w = 1.0 / d.df.values
    for t, nm in enumerate("AXN"):
        is_t = d.t.values == t
        d["s" + nm] = np.where(is_t, w, 0.0)
        d["n" + nm] = is_t.astype(np.int16)
        d["md" + nm] = np.where(is_t, d.df.values, 9999)
    g = d.groupby(["r", "s"], sort=False).agg(
        sA=("sA", "sum"), sX=("sX", "sum"), sN=("sN", "sum"), nA=("nA", "sum"), nX=("nX", "sum"), nN=("nN", "sum"),
        mdA=("mdA", "min"), mdX=("mdX", "min"), mdN=("mdN", "min")).reset_index()
    g["ksum"] = g.sA + g.sX + g.sN
    g["mindf"] = g[["mdA", "mdX", "mdN"]].min(axis=1)
    g["ncand_rec"] = g.groupby("r").s.transform("size")
    g = g.sort_values(["r", "ksum"], ascending=[True, False])
    g["krank"] = g.groupby("r").cumcount()
    best = g.r.map(g.loc[g.krank == 0].set_index("r").ksum)
    second = g.r.map(g.loc[g.krank == 1].set_index("r").ksum).fillna(0.0)
    g["kshare"] = g.ksum / best
    g["kgap"] = (best - second) / best
    g = g[g.krank < topk].copy()
    g["pool_i"] = (g.r + off).astype(np.int64)
    g = g.rename(columns={"s": "s1_i"}).drop(columns="r")
    f32 = [c for c in g.columns if c.startswith(("s", "k")) and c not in ("s1_i",)]
    g[f32] = g[f32].astype(np.float32)
    return g


class BlockingEngine:
    """Usage: ``with BlockingEngine(s1_names, s1_addrs, params) as eng: for pairs in eng.stream(names, addrs): ...``"""

    def __init__(self, s1_names, s1_addrs, params: Params, tag: str = "idx"):
        self.p = params
        self.n_s1 = len(s1_names)
        self.cap = max(params.cap_min, int(round(params.cap_frac * self.n_s1)))
        d = WORK_DIR / "index"
        d.mkdir(parents=True, exist_ok=True)
        self.hpath, self.ipath = d / f"{tag}_H.npy", d / f"{tag}_I.npy"
        self._build(list(s1_names), list(s1_addrs))
        self.pool = mp.Pool(params.n_workers, initializer=_init, initargs=(str(self.hpath), str(self.ipath)))

    def _build(self, names, addrs):
        T = self.p.task_records
        tasks = [(i, names[i:i + T], addrs[i:i + T]) for i in range(0, len(names), T)]
        log(f"   building S1 key index: {len(names):,} S1 records in {len(tasks)} tasks on {self.p.n_workers} workers "
            f"(key generation is the slow part)")
        prog = Progress(len(tasks), "index build", unit=" tasks")
        parts = []
        with mp.Pool(self.p.n_workers) as pl:      # temporary pool: scan workers need the finished index files
            for r in pl.imap(_index_task, tasks):
                parts.append(r)
                prog.update(1, f"{sum(len(p[0]) for p in parts) / 1e6:.1f}M keys so far")
        with stage("sorting keys and writing memory-mapped index"):
            H = np.concatenate([p[0] for p in parts]); I = np.concatenate([p[1] for p in parts]); del parts
            o = np.argsort(H, kind="stable")
            np.save(self.hpath, H[o]); np.save(self.ipath, I[o])
        self.n_keys = len(H)

    def stream(self, names, addrs):
        """Yield one DataFrame of candidate pairs per ``chunk_records`` pool records (columns: pool_i, s1_i, key stats)."""
        names, addrs = list(names), list(addrs)
        C, T = self.p.chunk_records, self.p.task_records
        n_chunks = (len(names) + C - 1) // C
        for ci, c0 in enumerate(range(0, len(names), C), 1):
            tasks = [(i, names[i:min(i + T, c0 + C)], addrs[i:min(i + T, c0 + C)], self.cap, self.p.top_k_per_record)
                     for i in range(c0, min(c0 + C, len(names)), T)]
            prog = Progress(len(tasks), f"blocking chunk {ci}/{n_chunks}", unit=" tasks")
            parts, npairs = [], 0
            for g in self.pool.imap(_scan_task, tasks):
                if g is not None:
                    parts.append(g); npairs += len(g)
                prog.update(1, f"{npairs:,} candidate pairs")
            yield pd.concat(parts, ignore_index=True) if parts else None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.pool.terminate()
