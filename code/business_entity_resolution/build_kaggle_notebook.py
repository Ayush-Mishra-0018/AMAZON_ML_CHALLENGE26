"""Generates kaggle_end_to_end.ipynb: ONE self-contained notebook (no src/ imports) for Kaggle."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))

# ----------------------------------------------------------------------------------------------------------------
md(r'''
# Business Entity Resolution: end-to-end on Kaggle
Single notebook: data -> normalisation -> record-centric blocking -> stage-1 LightGBM -> stage-2 S2<->S3 consensus -> accuracy gate -> `matching_results.tsv` + `candidate_pairs.tsv`.

## How to run on Kaggle
1. **Add the data:** *Add Input -> Upload* the challenge folder (the one containing `train/` and `test/`) as a Kaggle Dataset and attach it. The notebook finds it automatically under `/kaggle/input`.
2. **Settings:** Accelerator **None (CPU)** (no GPU is used). **Internet: On** (only needed if `rapidfuzz`/`text-unidecode` are not pre-installed).
3. **First run `RUN_MODE = "smoke"`** (next cell): a few minutes, exercises every code path including France. Then set `RUN_MODE = "full"` and use **Save Version -> Save & Run All (Commit)** so it runs in the background (up to 12 h). Outputs land in `/kaggle/working/output/`.
4. If a session dies, just run again: trained models and finished countries are **checkpointed** and skipped.

## Honest status
* The design comes from measured experiments on the training data (conjunctive-key blocking, decoy-aware second stage). Prototype numbers: F0.5 about 0.94 (US) / 0.92-0.93 (India) on held-out S1 groups.
* **This exact notebook code has not been run end-to-end by me.** Use smoke mode first; the accuracy gate protects you: **no submission file is written unless validation macro F0.5 >= `MIN_VAL_F05` for every country.**
* **France has no labels**, so its accuracy cannot be validated; check the printed per-country diagnostics against US/India.
''')

code(r'''
# ============================== CONFIG (edit here) ==============================
RUN_MODE = "smoke"          # "smoke" = few-minute code-path test  |  "full" = real run
MIN_VAL_F05 = 0.90          # ACCURACY GATE: refuse to write submission unless validation F0.5 >= this (per country)
N_TRAIN_S1 = {"US": 150_000, "India": 150_000}   # S1 entities per country in the training universe (full mode)
RESUME = True               # reuse saved models / finished-country checkpoints if present
SEED = 0
''')

code(r'''
import os, sys, json, time, zlib, gc, itertools, re, csv, threading, subprocess, shutil, multiprocessing as mp
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np, pandas as pd

ON_KAGGLE = Path("/kaggle").exists()
BASE = Path("/kaggle/working") if ON_KAGGLE else Path.cwd()
SUFFIX = "_smoke" if RUN_MODE == "smoke" else ""
WORK, OUT = BASE / f"work{SUFFIX}", BASE / f"output{SUFFIX}"
for d in (WORK, OUT, WORK / "ckpt", WORK / "models", WORK / "cache", WORK / "index"):
    d.mkdir(parents=True, exist_ok=True)

@dataclass
class Params:
    cap_frac: float = 3e-5          # a key is usable if shared by <= max(cap_min, cap_frac * n_S1) S1 records
    cap_min: int = 8
    top_k_per_record: int = 8       # S1 candidates kept per S2/S3 record (ranked by key score)
    chunk_records: int = 400_000    # pool records per pass (bounds RAM)
    task_records: int = 8_000       # pool records per worker task
    n_workers: int = max(1, os.cpu_count() or 2)
    stage1_min_p: float = 0.05
    lgb1: dict = field(default_factory=lambda: dict(n_estimators=250, learning_rate=0.06, num_leaves=63, min_child_samples=40,
                                                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1))
    lgb2: dict = field(default_factory=lambda: dict(n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=30, verbose=-1))
    n_folds: int = 3
    plateau_tol: float = 0.001      # choose the HIGHEST threshold within tol of the best OOF F0.5 (precision insurance)
    one_owner: bool = False         # measured ~neutral in prototype -> off

P = Params()
if RUN_MODE == "smoke":
    N_TRAIN_S1 = {"US": 4000, "India": 4000}
    SMOKE_POOL_HEAD, SMOKE_S1_HEAD = 300_000, 6000    # inference on a tiny slice, just to exercise the code
    MIN_VAL_F05 = 0.0                                  # tiny data: do not gate in smoke mode
    P.chunk_records = 150_000
print(f"mode={RUN_MODE} | kaggle={ON_KAGGLE} | cpus={os.cpu_count()} | work={WORK} | out={OUT}")
''')

code(r'''
# ============================== ENVIRONMENT ==============================
def _need(mod, pip):
    try:
        __import__(mod)
    except ImportError:
        print(f"installing {pip} (needs Internet ON) ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip])
for m, p in (("rapidfuzz", "rapidfuzz"), ("text_unidecode", "text-unidecode"), ("lightgbm", "lightgbm"), ("pyarrow", "pyarrow")):
    _need(m, p)
import lightgbm as lgb
from rapidfuzz import fuzz, process
try:
    import psutil
except ImportError:
    psutil = None
print("lightgbm", lgb.__version__, "| pandas", pd.__version__, "| numpy", np.__version__)
''')

# ----------------------------------------------------------------------------------------------------------------
md(r'''
## 1. Worker module
Multiprocessing uses the **spawn** context (safe after LightGBM/OpenMP threads exist; forking then can hang). Spawned workers cannot import functions defined inside a notebook, so the small worker-side code (text normalisation, key generation, index/scan tasks) is written to `er_workers.py` by this cell and imported. Everything else lives in the notebook.
''')

code(r'''
WORKER_SRC = r"""
import re, zlib, itertools
import numpy as np, pandas as pd
from text_unidecode import unidecode

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_DIGITS = re.compile(r"\d+")
_DOMAIN = re.compile(r"\.(?:com|net|org|in|fr|co|io)\b|^[#@]")

def norm(s):
    return _NON_ALNUM.sub(" ", unidecode(s).lower()).strip()

def is_non_latin(s):
    return any(ord(c) > 0x24F and c.isalpha() for c in s)

def looks_like_domain(s):
    return bool(_DOMAIN.search(s.lower()))

def skeleton_token(tok):
    s = tok.replace("ph", "f").replace("w", "v").replace("z", "j").replace("x", "ks").replace("c", "k").replace("q", "k")
    t = s[0] + re.sub(r"[aeiouyh]", "", s[1:])
    return re.sub(r"(.)\1+", r"\1", t)

def skeleton(normed):
    return " ".join(skeleton_token(t) for t in normed.split() if t)

def number_set(normed_addr):
    return frozenset((x.lstrip("0") or "0") for x in _DIGITS.findall(normed_addr))

def name_tokens(name):
    n = norm(name)
    t = [x for x in n.split() if len(x) > 1 or x.isdigit()]
    out = set(t)
    out.add("~" + "".join(n.split())[:8])                      # joined prefix: 'prairiefive.com' ~ 'Prairie Five'
    for x in t:
        if x.isalpha() and len(x) >= 3:
            out.add("sk" + skeleton_token(x))                  # phonetic token (transliterated names)
    return sorted(out)

def addr_tokens(addr):
    out = set()
    for x in norm(addr).split():
        if len(x) < 2 and not x.isdigit():
            continue
        if x.isdigit():
            x = x.lstrip("0") or "0"                           # 0011118 == 11118
            out.add(x)
            if len(x) >= 2:
                out.add("s2" + x[-2:])                         # 5844 ~ 844 ~ 44 (house-number corruption)
        else:
            out.add(x)
    return sorted(out)

def h64(s):
    b = s.encode()
    return zlib.crc32(b) | (zlib.crc32(b, 0x9E3779B9) << 32)

def keys_for_records(names, addrs):
    # conjunctive keys: A = address-token pair, X = name-token x address-token, N = name-token pair
    H, T, R = [], [], []
    for i, (n, a) in enumerate(zip(names, addrs)):
        nt, at = name_tokens(n), addr_tokens(a)
        hs, ts = [], []
        for x, y in itertools.combinations(at, 2):
            hs.append(h64("A|" + x + "|" + y)); ts.append(0)
        for x in nt:
            for y in at:
                hs.append(h64("X|" + x + "|" + y)); ts.append(1)
        for x, y in itertools.combinations(nt, 2):
            hs.append(h64("N|" + x + "|" + y)); ts.append(2)
        H.extend(hs); T.extend(ts); R.extend([i] * len(hs))
    return np.array(H, dtype=np.uint64), np.array(T, dtype=np.int8), np.array(R, dtype=np.int32)

_H = _I = None

def init(hpath, ipath):
    global _H, _I
    _H = np.load(hpath, mmap_mode="r")
    _I = np.load(ipath, mmap_mode="r")

def index_task(args):
    off, names, addrs = args
    H, T, R = keys_for_records(names, addrs)
    return H, (R + off).astype(np.int32)

def scan_task(args):
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
    rep_r, rep_t, rep_df = np.repeat(R, df), np.repeat(T, df), np.repeat(df, df)
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
    f32 = [c for c in g.columns if c.startswith(("s", "k")) and c != "s1_i"]
    g[f32] = g[f32].astype(np.float32)
    return g
"""
(WORK / "er_workers.py").write_text(WORKER_SRC, encoding="utf-8")
if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))
import importlib, er_workers as W
importlib.reload(W)
CTX = mp.get_context("spawn")
print("worker module written:", WORK / "er_workers.py")
print(W.name_tokens("prairiefive.com")[:6], W.addr_tokens("0011118 Main St")[:6])
''')

# ----------------------------------------------------------------------------------------------------------------
md("## 2. Progress / logging helpers\nTimestamped lines with elapsed time and RAM, stage durations, heartbeat for long silent steps, progress with ETA. Also tee'd to `work/run.log`.")
code(r'''
_T0 = time.time()
_LOG = open(WORK / "run.log", "a", encoding="utf-8")

def _fmt(sec):
    sec = int(max(0, sec)); h, r = divmod(sec, 3600); m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

def log(*a):
    ram = ""
    if psutil:
        v = psutil.virtual_memory(); ram = f" | RAM {v.used / 1e9:.1f}/{v.total / 1e9:.0f}GB"
    line = f"[{time.strftime('%H:%M:%S')} +{_fmt(time.time() - _T0)}{ram}] " + " ".join(str(x) for x in a)
    print(line, flush=True); _LOG.write(line + "\n"); _LOG.flush()

def banner(t):
    log("\n" + "=" * 78 + f"\n  {t}\n" + "=" * 78)

@contextmanager
def stage(title, every=30.0):
    log(f">> {title}"); t0 = time.time(); stop = threading.Event()
    def beat():
        while not stop.wait(every):
            log(f"   ... still {title} ({_fmt(time.time() - t0)} so far)")
    th = threading.Thread(target=beat, daemon=True); th.start()
    try:
        yield
    finally:
        stop.set()
    log(f"OK {title}  (took {_fmt(time.time() - t0)})")

class Progress:
    def __init__(self, total, desc, every=10.0, unit=""):
        self.total, self.desc, self.every, self.unit = max(int(total), 1), desc, every, unit
        self.n, self.t0, self.last = 0, time.time(), 0.0
    def update(self, n=1, extra=""):
        self.n += n; now = time.time()
        if now - self.last >= self.every or self.n >= self.total:
            self.last = now; el = now - self.t0; rate = self.n / el if el > 0 else 0
            eta = (self.total - self.n) / rate if rate > 0 else 0
            log(f"   {self.desc}: {self.n:,}/{self.total:,} ({100 * self.n / self.total:.1f}%) | {rate:,.1f}{self.unit}/s | "
                f"elapsed {_fmt(el)} | ETA {_fmt(eta)}" + (f" | {extra}" if extra else ""))

def table(title, rows):
    log(title + "\n" + "\n".join(f"      {k!s:>6} : {v:.4f}" for k, v in rows.items()))
log("helpers ready")
''')

# ----------------------------------------------------------------------------------------------------------------
md("## 3. Data loading\nFinds the data under `/kaggle/input` (or a local `Data/` folder). TSVs are read with an explicit tab separator and no quote handling, then cached as parquet in `/kaggle/working`.")
code(r'''
def find_data_dir():
    roots = [Path("/kaggle/input"), Path("Data"), Path("../Data"), Path("../../Data"), Path("../../../Data"), Path(".")]
    for r in roots:
        if r.exists():
            for p in r.rglob("train_source1.tsv"):
                return p.parents[1]
    raise FileNotFoundError("Attach the challenge dataset (folder containing train/ and test/) or place it under ./Data")

DATA = find_data_dir()
print("data dir:", DATA)

def load_split(split):
    names = ("source1", "source2", "source3") + (("ground_truth",) if split == "train" else ())
    out = {}
    for k, n in enumerate(names, 1):
        pq, tsv = WORK / "cache" / f"{split}_{n}.parquet", DATA / split / f"{split}_{n}.tsv"
        if not pq.exists():
            with stage(f"[{split} {k}/{len(names)}] converting {tsv.name} ({tsv.stat().st_size / 1e6:.0f} MB) to parquet (first run only)"):
                pd.read_csv(tsv, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE).to_parquet(pq)
        with stage(f"[{split} {k}/{len(names)}] loading {pq.name}"):
            out[n] = pd.read_parquet(pq)
        log(f"   {n}: {len(out[n]):,} rows")
    pool = pd.concat([out["source2"].assign(src="S2"), out["source3"].assign(src="S3")], ignore_index=True)
    return out["source1"], pool, out.get("ground_truth")

def owner_map(gt):
    """pool id -> owning S1 id (asserts the one-owner property observed in the training labels)."""
    g = gt[gt.matched_entity_ids != ""]
    ex = pd.DataFrame({"s1": g.source1_entity_id.values, "m": g.matched_entity_ids.str.split(",").values}).explode("m")
    assert not ex.m.duplicated().any(), "one-owner property violated in labels"
    return pd.Series(ex.s1.values, index=ex.m.values)

def make_universe(s1, pool, own, country, n_s1, seed=0):
    """Cluster-closed sample: n_s1 S1 + ALL their true S2/S3 records + a proportional share of orphans.
    (Caveat: same-name confounders are diluted vs full scale, so real F0.5 can be a bit lower.)"""
    log(f"   building {country} training universe ({n_s1:,} S1 + their records + proportional orphans)")
    rng = np.random.RandomState(seed)
    s1c = s1[s1.country == country]; n_s1 = min(n_s1, len(s1c))
    pick = s1c.iloc[rng.choice(len(s1c), n_s1, replace=False)].reset_index(drop=True)
    pc = pool[pool.country == country]
    owner = pc.entity_id.map(own)
    keep = owner.isin(set(pick.entity_id)).values | (owner.isna().values & (rng.rand(len(pc)) < n_s1 / len(s1c)))
    upool = pc[keep].reset_index(drop=True)
    truth = {e: set() for e in pick.entity_id}
    sub = own[own.isin(set(pick.entity_id))]
    for m, s in zip(sub.index, sub.values):
        truth[s].add(m)
    return pick, upool, truth
''')

# ----------------------------------------------------------------------------------------------------------------
md("""
## 4. Records + pair features
Vectorised with `rapidfuzz.process.cpdist`. Every feature is computable at test time. **Deliberately excluded:** absolute S1-name-frequency (helped India in CV but test S1 is a differently sized sample: US duplicate-name rate 0.29 vs 0.36 in train).
""")
code(r'''
NAME_F = ["nr", "nts", "npr", "nso", "nsk", "nskts", "lnn", "nonlat", "dom"]
ADDR_F = ["ats", "ar", "apr", "numj", "numeq", "aempty", "lna"]
KEY_F = ["sA", "sX", "sN", "nA", "nX", "nN", "ksum", "kshare", "kgap", "krank", "ncand_rec", "mindf_rel", "mdA_rel", "mdX_rel", "mdN_rel"]
STAGE1 = NAME_F + ADDR_F + ["isS2"] + KEY_F
REL_F = ["sup", "cross", "bestn", "besta", "numagree", "nconf_others"]
STAGE2 = ["p1"] + STAGE1 + REL_F

class Records:
    """Normalised array-backed view of one side (S1 or pool)."""
    def __init__(self, df):
        self.ids = df.entity_id.values
        self.nn = [W.norm(x) for x in df.business_name]
        self.na = [W.norm(x) for x in df.business_address]
        self.sk = [W.skeleton(x) for x in self.nn]
        self.nums = [W.number_set(x) for x in self.na]
        self.nonlat = np.array([W.is_non_latin(x) for x in df.business_name])
        self.dom = np.array([W.looks_like_domain(x) for x in df.business_name])
        self.lnn = np.array([len(x) for x in self.nn], dtype=np.float32)
        self.lna = np.array([len(x) for x in self.na], dtype=np.float32)
        self.empty_a = self.lna == 0
        self.isS2 = (df.src.values == "S2") if "src" in df else np.zeros(len(df), bool)

def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.uint8).astype(np.float32)

def _take(lst, idx):
    return [lst[i] for i in idx]

def pair_features(pairs, S1, Pl, cap):
    a, b = pairs.s1_i.values, pairs.pool_i.values
    an, bn, aa, ba = _take(S1.nn, a), _take(Pl.nn, b), _take(S1.na, a), _take(Pl.na, b)
    f = pd.DataFrame(index=pairs.index)
    f["nr"], f["nts"] = _cp(an, bn, fuzz.ratio), _cp(an, bn, fuzz.token_set_ratio)
    f["npr"], f["nso"] = _cp(an, bn, fuzz.partial_ratio), _cp(an, bn, fuzz.token_sort_ratio)
    ask, bsk = _take(S1.sk, a), _take(Pl.sk, b)
    f["nsk"], f["nskts"] = _cp(ask, bsk, fuzz.ratio), _cp(ask, bsk, fuzz.token_set_ratio)
    f["lnn"], f["nonlat"], f["dom"] = Pl.lnn[b], Pl.nonlat[b].astype(np.float32), Pl.dom[b].astype(np.float32)
    empty = Pl.empty_a[b]
    for nm, sc in (("ats", fuzz.token_set_ratio), ("ar", fuzz.ratio), ("apr", fuzz.partial_ratio)):
        v = _cp(aa, ba, sc); v[empty] = -1; f[nm] = v
    nj, ne = np.full(len(a), -1, np.float32), np.full(len(a), -1, np.float32)
    for i, (x, y) in enumerate(zip(_take(S1.nums, a), _take(Pl.nums, b))):
        if x and y:
            nj[i] = len(x & y) / len(x | y); ne[i] = float(x == y)     # decoys differ by a house number
    f["numj"], f["numeq"] = nj, ne
    f["aempty"], f["lna"], f["isS2"] = empty.astype(np.float32), Pl.lna[b], Pl.isS2[b].astype(np.float32)
    for c in ["sA", "sX", "sN", "nA", "nX", "nN", "ksum", "kshare", "kgap", "krank", "ncand_rec"]:
        f[c] = pairs[c].values.astype(np.float32)
    f["mindf_rel"] = np.minimum(pairs.mindf.values, 9999) / cap
    for t in "AXN":
        f[f"md{t}_rel"] = np.minimum(pairs[f"md{t}"].values, 9999) / cap
    return f[STAGE1]
log("features ready:", len(STAGE1), "stage-1 features")
''')

# ----------------------------------------------------------------------------------------------------------------
md("""
## 5. Record-centric blocking
Index the small, deduplicated **S1** side once (sorted 64-bit key hashes, memory-mapped so all workers share it), then stream every S2/S3 record against it. Each pool record keeps its top-K S1 candidates ranked by conjunctive-key score. `candidate_pairs.tsv` is exactly this set.
Prototype evidence (S1-centric, full US pool): single-token blocking <=34% recall; conjunctive keys **98.2% at ~308 candidates/S1**; India harder (95.0% at 515), mostly non-Latin names and address fragments.
""")
code(r'''
class BlockingEngine:
    def __init__(self, s1_names, s1_addrs, params, tag="idx"):
        self.p, self.n_s1 = params, len(s1_names)
        self.cap = max(params.cap_min, int(round(params.cap_frac * self.n_s1)))
        self.hpath, self.ipath = WORK / "index" / f"{tag}_H.npy", WORK / "index" / f"{tag}_I.npy"
        self._build(list(s1_names), list(s1_addrs))
        self.pool = CTX.Pool(params.n_workers, initializer=W.init, initargs=(str(self.hpath), str(self.ipath)))

    def _build(self, names, addrs):
        T = self.p.task_records
        tasks = [(i, names[i:i + T], addrs[i:i + T]) for i in range(0, len(names), T)]
        log(f"   building S1 key index: {len(names):,} S1 in {len(tasks)} tasks on {self.p.n_workers} workers")
        prog, parts = Progress(len(tasks), "index build", unit=" tasks"), []
        with CTX.Pool(self.p.n_workers) as pl:
            for r in pl.imap(W.index_task, tasks):
                parts.append(r); prog.update(1, f"{sum(len(x[0]) for x in parts) / 1e6:.1f}M keys")
        with stage("sorting keys and writing memory-mapped index"):
            H = np.concatenate([x[0] for x in parts]); I = np.concatenate([x[1] for x in parts]); del parts
            o = np.argsort(H, kind="stable"); np.save(self.hpath, H[o]); np.save(self.ipath, I[o])
        self.n_keys = len(H); del H, I, o; gc.collect()

    def stream(self, names, addrs):
        names, addrs = list(names), list(addrs)
        C, T = self.p.chunk_records, self.p.task_records
        n_chunks = (len(names) + C - 1) // C
        for ci, c0 in enumerate(range(0, len(names), C), 1):
            tasks = [(i, names[i:min(i + T, c0 + C)], addrs[i:min(i + T, c0 + C)], self.cap, self.p.top_k_per_record)
                     for i in range(c0, min(c0 + C, len(names)), T)]
            prog, parts, npairs = Progress(len(tasks), f"blocking chunk {ci}/{n_chunks}", unit=" tasks"), [], 0
            for g in self.pool.imap(W.scan_task, tasks):
                if g is not None:
                    parts.append(g); npairs += len(g)
                prog.update(1, f"{npairs:,} candidate pairs")
            yield pd.concat(parts, ignore_index=True) if parts else None

    def close(self):
        self.pool.terminate()

def candidate_pairs_for(S1, Pl, params, pool_df, s1_df, on_chunk):
    """Blocking + stage-1 features per chunk of pool records; on_chunk(pairs, feats) consumes each chunk."""
    eng = BlockingEngine(s1_df.business_name, s1_df.business_address, params)
    try:
        log(f"   index ready: {eng.n_s1:,} S1, {eng.n_keys / 1e6:.1f}M keys, key-frequency cap = {eng.cap}")
        n_chunks = (len(pool_df) + params.chunk_records - 1) // params.chunk_records
        total, t0 = 0, time.time()
        for ci, pairs in enumerate(eng.stream(pool_df.business_name, pool_df.business_address), 1):
            if pairs is None:
                log(f"   chunk {ci}/{n_chunks}: no candidates"); continue
            with stage(f"chunk {ci}/{n_chunks}: {len(pairs):,} pair features + scoring"):
                on_chunk(pairs, pair_features(pairs, S1, Pl, eng.cap))
            total += len(pairs); done = min(ci * params.chunk_records, len(pool_df)); el = time.time() - t0
            log(f"   chunks {ci}/{n_chunks} | {done:,}/{len(pool_df):,} pool records | {total:,} pairs | "
                f"ETA this country {_fmt(el / done * (len(pool_df) - done))}")
    finally:
        eng.close()
        for f in (eng.hpath, eng.ipath):
            try: f.unlink()
            except OSError: pass
''')

# ----------------------------------------------------------------------------------------------------------------
md("""
## 6. Stage-2 consensus, decision rule, metric
**Stage 2:** false positives are mostly near-duplicate *decoys* (same name, perturbed house number/suffix). A true member agrees with the *other* confident members (independent noise); a decoy does not. Prototype: F0.5 US 0.942 -> 0.949, India 0.916 -> 0.935. Listwise-only features ~0; one-owner assignment ~+0.001 (off).
**Decision:** threshold = the *highest* one within `plateau_tol` of the best out-of-fold F0.5 (F0.5 is flat near its optimum), as insurance because the test orphan rate is unknowable from unlabeled data.
""")
code(r'''
def relational_features(rows, Pl, conf_p=0.5):
    """rows: s1_i, pool_i, p1, isS2 (unique index). S2<->S3 consensus features aligned to rows.index."""
    out = pd.DataFrame(0.0, index=rows.index, columns=REL_F, dtype=np.float32)
    conf = rows.loc[rows.p1 >= conf_p, ["s1_i", "pool_i", "p1", "isS2"]]
    if conf.empty:
        return out
    r = rows[["s1_i", "pool_i", "isS2"]].copy(); r["row"] = rows.index
    m = r.merge(conf.rename(columns={"pool_i": "pool_m", "p1": "p_m", "isS2": "isS2_m"}), on="s1_i")
    nconf = conf.groupby("s1_i").size()
    out["nconf_others"] = (rows.s1_i.map(nconf).fillna(0).values - (rows.p1.values >= conf_p)).astype(np.float32)
    m = m[m.pool_i.values != m.pool_m.values]
    if m.empty:
        return out
    bi, bm = m.pool_i.values, m.pool_m.values
    sn = _cp(_take(Pl.nn, bi), _take(Pl.nn, bm), fuzz.token_set_ratio)
    sa = _cp(_take(Pl.na, bi), _take(Pl.na, bm), fuzz.token_set_ratio)
    sa = np.where(~(Pl.empty_a[bi] | Pl.empty_a[bm]), sa, 0.0)
    na = np.array([float(bool(x) and bool(y) and x == y) for x, y in zip(_take(Pl.nums, bi), _take(Pl.nums, bm))], np.float32)
    sup = (sa >= 90) & (sn >= 80)
    m = m.assign(bestn=sn * m.p_m.values, besta=sa * m.p_m.values, sup=sup.astype(np.float32),
                 cross=(sup & (m.isS2.values != m.isS2_m.values)).astype(np.float32), numagree=na)
    g = m.groupby("row").agg(sup=("sup", "sum"), cross=("cross", "sum"), bestn=("bestn", "max"),
                             besta=("besta", "max"), numagree=("numagree", "sum"))
    out.loc[g.index, g.columns] = g.values.astype(np.float32)
    return out

def f05_from_counts(tp, n, nt):
    tp, n, nt = (np.asarray(x, dtype=np.float64) for x in (tp, n, nt))
    with np.errstate(divide="ignore", invalid="ignore"):
        Pp, Rr = tp / n, tp / nt
        f = 1.25 * Pp * Rr / (0.25 * Pp + Rr)
    f = np.where((tp == 0) | (n == 0) | (nt == 0), 0.0, f)
    return np.where(nt == 0, (n == 0).astype(np.float64), f)          # singleton: 1.0 iff prediction is empty

def macro_f05(s1_idx, y, n_s1, ntrue):
    return f05_from_counts(np.bincount(s1_idx, weights=y, minlength=n_s1), np.bincount(s1_idx, minlength=n_s1), ntrue)

def scan_thresholds(s1_idx, p, y, n_s1, ntrue, grid=None):
    grid = np.round(np.arange(0.30, 0.96, 0.05), 2) if grid is None else grid
    return {float(t): float(macro_f05(s1_idx[p >= t], y[p >= t], n_s1, ntrue).mean()) for t in grid}

def choose_threshold(scores, tol):
    best = max(scores.values())
    return max(t for t, v in scores.items() if v >= best - tol), best

# --- instant self-test of the metric against a plain-python reference (no data needed) ---
def _ref(pred, truth):
    out = []
    for e, T in truth.items():
        Pp = set(pred.get(e, ()))
        if not T: out.append(1.0 if not Pp else 0.0); continue
        tp = len(Pp & T)
        if tp == 0: out.append(0.0); continue
        p_, r_ = tp / len(Pp), tp / len(T); out.append(1.25 * p_ * r_ / (0.25 * p_ + r_))
    return float(np.mean(out))
_t = {"a": {"x", "y"}, "b": set(), "c": {"z"}, "d": set()}; _p = {"a": {"x", "q"}, "b": set(), "c": {"z"}, "d": {"w"}}
assert abs(_ref(_p, _t) - macro_f05(np.array([0, 0, 2, 3]), np.array([1, 0, 1, 0]), 4, np.array([2, 0, 1, 0])).mean()) < 1e-9
print("metric self-test passed")
''')

# ----------------------------------------------------------------------------------------------------------------
md("""
## 7. Train + validate (accuracy-gated)
Builds cluster-closed universes, runs the real blocking + features, produces **grouped out-of-fold** predictions (grouped by S1 name, so same-name chains never straddle folds), and reports the decomposition: **candidate recall**, the **ceiling** (F0.5 with a perfect scorer inside the candidate set), stage-1, stage-2, per country, singletons vs non-singletons.
""")
code(r'''
def build_training_country(s1, pool, own, country, n_s1, params):
    banner(f"TRAIN DATA: {country}")
    pick, upool, truth = make_universe(s1, pool, own, country, n_s1, SEED)
    log(f"   universe: {len(pick):,} S1, {len(upool):,} S2/S3, {sum(len(t) for t in truth.values()):,} true pairs")
    with stage("normalising records"):
        S1, Pl = Records(pick), Records(upool)
    s1_index = {e: i for i, e in enumerate(pick.entity_id)}
    pid = {e: i for i, e in enumerate(upool.entity_id)}
    owner = np.full(len(upool), -1, np.int64)
    for e, T in truth.items():
        for m in T:
            owner[pid[m]] = s1_index[e]
    ntrue = np.array([len(truth[e]) for e in pick.entity_id])
    parts = []
    candidate_pairs_for(S1, Pl, params, upool, pick, lambda pr, f: parts.append(pd.concat([pr[["s1_i", "pool_i"]], f], axis=1)))
    D = pd.concat(parts, ignore_index=True)
    D["y"] = (owner[D.pool_i.values] == D.s1_i.values).astype(np.int8)
    fold = np.array([zlib.crc32(x.encode()) % params.n_folds for x in S1.nn])
    D["fold"], D["country"] = fold[D.s1_i.values], country
    log(f"   RESULT {country}: {len(D):,} candidate pairs ({len(D) / len(pick):.0f}/S1) | true pairs captured by blocking "
        f"{D.y.sum():,}/{ntrue.sum():,} = {D.y.sum() / ntrue.sum():.4f}")
    return dict(country=country, P=Pl, D=D, ntrue=ntrue, n_s1=len(pick))

def _oof(X, y, fold, prm, n_folds, name):
    p = np.zeros(len(X), np.float32)
    for f in range(n_folds):
        tr = fold != f
        with stage(f"{name}: fold {f + 1}/{n_folds} (train {tr.sum():,} -> predict {(~tr).sum():,})"):
            m = lgb.LGBMClassifier(**prm, random_state=SEED, n_jobs=os.cpu_count()).fit(X[tr], y[tr])
            p[~tr] = m.predict_proba(X[~tr])[:, 1]
    return p

def train(params):
    t_all = time.time()
    banner("TRAINING + VALIDATION (1/5: load labelled data)")
    s1, pool, gt = load_split("train")
    own = owner_map(gt)
    banner("2/5: blocking + features per country")
    C = [build_training_country(s1, pool, own, c, n, params) for c, n in N_TRAIN_S1.items()]
    del s1, pool, gt, own; gc.collect()
    D = pd.concat([c["D"] for c in C], ignore_index=True)
    off = np.cumsum([0] + [c["n_s1"] for c in C])
    D["gs1"] = D.s1_i.values + np.repeat(off[:-1], [len(c["D"]) for c in C])
    ntrue_all, n_all = np.concatenate([c["ntrue"] for c in C]), int(off[-1])
    rep = {"candidate_recall_pairs": float(D.y.sum() / ntrue_all.sum())}
    pos = D[D.y == 1]
    rep["ceiling_F05"] = float(macro_f05(pos.gs1.values, np.ones(len(pos)), n_all, ntrue_all).mean())
    log(f"   COMBINED: {len(D):,} pairs | candidate recall {rep['candidate_recall_pairs']:.4f} | ceiling F0.5 (perfect scorer) {rep['ceiling_F05']:.4f}")

    banner("3/5: stage-1 pair scorer (grouped out-of-fold)")
    X = D[STAGE1].to_numpy(np.float32)
    D["p1"] = _oof(X, D.y.values, D.fold.values, params.lgb1, params.n_folds, "stage 1")
    sc1 = scan_thresholds(D.gs1.values, D.p1.values, D.y.values, n_all, ntrue_all); rep["stage1_by_threshold"] = sc1
    table("   stage-1 macro F0.5 by threshold:", sc1); log(f"   stage-1 best F0.5 = {max(sc1.values()):.4f}")

    banner("4/5: stage-2 S2<->S3 consensus")
    S = D[D.p1 >= params.stage1_min_p].copy()
    log(f"   {len(S):,} of {len(D):,} pairs survive stage 1")
    rel = []
    for c in C:
        with stage(f"relational features {c['country']}"):
            idx = S.index[S.country.values == c["country"]]
            sub = S.loc[idx, ["s1_i", "pool_i", "p1", "isS2"]].reset_index(drop=True)
            r = relational_features(sub, c["P"]); r.index = idx; rel.append(r)
    S = S.join(pd.concat(rel))
    S["p2"] = _oof(S[STAGE2].to_numpy(np.float32), S.y.values, S.fold.values, params.lgb2, params.n_folds, "stage 2")
    sc2 = scan_thresholds(S.gs1.values, S.p2.values, S.y.values, n_all, ntrue_all)
    th, best = choose_threshold(sc2, params.plateau_tol)
    rep.update(stage2_by_threshold=sc2, threshold=th, stage2_at_threshold=sc2[th], stage2_best=best)
    table("   stage-2 macro F0.5 by threshold:", sc2); log(f"   chosen threshold {th} (highest within {params.plateau_tol} of best {best:.4f})")
    m = S.p2.values >= th; per = {}
    for c in C:
        mm = m & (S.country.values == c["country"])
        f = macro_f05(S.s1_i.values[mm], S.y.values[mm], c["n_s1"], c["ntrue"])
        per[c["country"]] = dict(F05=float(f.mean()), singleton=float(f[c["ntrue"] == 0].mean()), non_singleton=float(f[c["ntrue"] > 0].mean()))
    rep["per_country"] = per; tp = S.y.values[m].sum()
    rep["micro_precision"], rep["micro_recall"] = float(tp / m.sum()), float(tp / ntrue_all.sum())

    banner("5/5: accuracy gate + final models")
    rep["gate"] = MIN_VAL_F05
    rep["passes_gate"] = bool(sc2[th] >= MIN_VAL_F05 and all(v["F05"] >= MIN_VAL_F05 for v in per.values()))
    log(f"   ceiling F0.5 (blocking limit) : {rep['ceiling_F05']:.4f}")
    log(f"   OVERALL validation F0.5       : {sc2[th]:.4f}")
    for k, v in per.items():
        log(f"   {k:>6}: F0.5 {v['F05']:.4f} | singletons {v['singleton']:.4f} | non-singletons {v['non_singleton']:.4f}")
    log(f"   micro precision {rep['micro_precision']:.4f} | micro recall {rep['micro_recall']:.4f}")
    log(f"   ACCURACY GATE (>= {MIN_VAL_F05}): {'PASS' if rep['passes_gate'] else 'FAIL -> no submission will be written'}")
    with stage("fitting final stage-1 model"):
        m1 = lgb.LGBMClassifier(**params.lgb1, random_state=SEED, n_jobs=os.cpu_count()).fit(X, D.y.values)
    with stage("fitting final stage-2 model"):
        m2 = lgb.LGBMClassifier(**params.lgb2, random_state=SEED, n_jobs=os.cpu_count()).fit(S[STAGE2].to_numpy(np.float32), S.y.values)
    m1.booster_.save_model(str(WORK / "models" / "stage1.txt")); m2.booster_.save_model(str(WORK / "models" / "stage2.txt"))
    (WORK / "models" / "meta.json").write_text(json.dumps(dict(threshold=th, report=rep), indent=1))
    log(f"   models saved; training took {_fmt(time.time() - t_all)}")
    return rep

META = WORK / "models" / "meta.json"
if RESUME and META.exists():
    REPORT = json.loads(META.read_text())["report"]; log("RESUME: loaded saved models + validation report (delete work/models to retrain)")
else:
    REPORT = train(P)
''')

code(r'''
# Validation summary + threshold curves + gate
try:
    import matplotlib.pyplot as plt
    s1c = {float(k): v for k, v in REPORT["stage1_by_threshold"].items()}; s2c = {float(k): v for k, v in REPORT["stage2_by_threshold"].items()}
    plt.figure(figsize=(6, 3.5)); plt.plot(list(s1c), list(s1c.values()), "o-", label="stage 1")
    plt.plot(list(s2c), list(s2c.values()), "o-", label="stage 2 (+consensus)")
    plt.axvline(REPORT["threshold"], ls="--", c="gray"); plt.axhline(MIN_VAL_F05, ls=":", c="red")
    plt.xlabel("threshold"); plt.ylabel("grouped OOF macro F0.5"); plt.legend(); plt.grid(alpha=.3); plt.show()
except Exception as e:
    print("plot skipped:", e)
print(json.dumps({k: v for k, v in REPORT.items() if "by_threshold" not in k}, indent=1))
GATE_OK = REPORT["passes_gate"]
print("\nACCURACY GATE:", "PASS - inference will run" if GATE_OK else "FAIL - inference below will refuse to run")
''')

# ----------------------------------------------------------------------------------------------------------------
md("""
## 8. Inference on the test set (per country, checkpointed)
Every country label found in the test S1 is processed (**open set**, so France goes through the same generic code). Each finished country is saved to `work/ckpt/`; a restart skips it. Refuses to run if the gate failed.
France has no labels; compare its printed **predicted singleton rate** and **matches per S1** with the training values (~0.056 and ~3.46).
""")
code(r'''
def infer_country(country, s1, pool, b1, b2, th, params):
    ck_m, ck_c, ck_d = (WORK / "ckpt" / f"{country}_{k}" for k in ("matches.parquet", "cands.parquet", "diag.json"))
    if RESUME and ck_m.exists() and ck_c.exists() and ck_d.exists():
        log(f"[{country}] checkpoint found, skipping"); return json.loads(ck_d.read_text())
    s1c = s1[s1.country == country].reset_index(drop=True)
    pc = pool[pool.country == country].reset_index(drop=True)
    if RUN_MODE == "smoke":
        s1c, pc = s1c.head(SMOKE_S1_HEAD), pc.head(SMOKE_POOL_HEAD)
    banner(f"INFERENCE: {country}  ({len(s1c):,} S1, {len(pc):,} S2/S3)")
    with stage("normalising records"):
        S1, Pl = Records(s1c), Records(pc)
    cand_parts, surv = [], []
    def on_chunk(pairs, f):
        cand_parts.append(pairs[["s1_i", "pool_i"]].values.astype(np.int32))
        p1 = b1.predict(f.to_numpy(np.float32)); k = p1 >= params.stage1_min_p
        surv.append(pd.concat([pairs.loc[k, ["s1_i", "pool_i"]], f[k]], axis=1).assign(p1=p1[k]))
    candidate_pairs_for(S1, Pl, params, pc, s1c, on_chunk)
    Cn = np.concatenate(cand_parts) if cand_parts else np.zeros((0, 2), np.int32)
    R = pd.concat(surv, ignore_index=True) if surv else pd.DataFrame(columns=["s1_i", "pool_i", "p1"] + STAGE1)
    log(f"   {len(Cn):,} candidate pairs; {len(R):,} survive stage 1")
    if len(R):
        with stage("stage 2: consensus features + scoring"):
            R = R.join(relational_features(R[["s1_i", "pool_i", "p1", "isS2"]], Pl))
            R["p2"] = b2.predict(R[STAGE2].to_numpy(np.float32))
        sel = R[R.p2 >= th]
        if params.one_owner:
            sel = sel.loc[sel.groupby("pool_i").p2.idxmax()]
    else:
        sel = R.assign(p2=[])
    with stage("assembling per-S1 lists and saving checkpoint"):
        def lists(df):
            if not len(df): return pd.DataFrame({"s1_id": [], "ids": []})
            g = df.groupby("s1_i").pool_i.agg(lambda x: ",".join(Pl.ids[x.values]))
            return pd.DataFrame({"s1_id": S1.ids[g.index.values], "ids": g.values})
        lists(sel).to_parquet(ck_m)
        lists(pd.DataFrame(Cn, columns=["s1_i", "pool_i"])).to_parquet(ck_c)
    n_pred = np.bincount(sel.s1_i.values.astype(int), minlength=len(s1c)) if len(sel) else np.zeros(len(s1c), int)
    d = dict(n_s1=len(s1c), n_pool=len(pc), cand_pairs=int(len(Cn)), mean_cand_per_s1=len(Cn) / max(len(s1c), 1),
             pred_singleton_rate=float((n_pred == 0).mean()), mean_pred_matches=float(n_pred.mean()),
             pool_records_matched_frac=float(sel.pool_i.nunique() / max(len(pc), 1)) if len(sel) else 0.0)
    ck_d.write_text(json.dumps(d))
    log(f"   RESULT {country}: {d['mean_cand_per_s1']:.0f} candidates/S1 | predicted singletons {d['pred_singleton_rate']:.3f} (train truth ~0.056) | "
        f"{d['mean_pred_matches']:.2f} matches/S1 (train truth ~3.46) | pool matched {d['pool_records_matched_frac']:.3f}")
    del R, Cn, sel, S1, Pl; gc.collect()
    return d

if not GATE_OK:
    raise RuntimeError(f"Validation F0.5 {REPORT['stage2_at_threshold']:.4f} (or a country) is below the gate {MIN_VAL_F05}. "
                       "Not running inference. Improve the model (bigger N_TRAIN_S1, features) or lower MIN_VAL_F05 deliberately.")
banner("INFERENCE (1: load test data + models)")
TS1, TPOOL, _ = load_split("test")
B1 = lgb.Booster(model_file=str(WORK / "models" / "stage1.txt")); B2 = lgb.Booster(model_file=str(WORK / "models" / "stage2.txt"))
TH = json.loads(META.read_text())["threshold"]
COUNTRIES = sorted(TS1.country.unique()); log(f"countries in test (open set): {COUNTRIES} | threshold {TH}")
DIAG = {}
for k, c in enumerate(COUNTRIES, 1):
    log(f"--- country {k}/{len(COUNTRIES)}: {c}")
    DIAG[c] = infer_country(c, TS1, TPOOL, B1, B2, TH, P)
pd.DataFrame(DIAG).T
''')

# ----------------------------------------------------------------------------------------------------------------
md("## 9. Write the two TSV files + validate\nOne row per test S1 entity (empty list for singletons). Final matches are a subset of candidates by construction. Checks below mirror the official validator; the official script is also run if it is found in the dataset.")
code(r'''
banner("WRITING SUBMISSION FILES")
ids = TS1.entity_id.tolist()
def collect(kind):
    parts = [pd.read_parquet(WORK / "ckpt" / f"{c}_{kind}.parquet") for c in COUNTRIES]
    s = pd.concat(parts).set_index("s1_id").ids
    return s.reindex(ids).fillna("")
mt, cd = collect("matches"), collect("cands")
for name, hdr, ser in (("matching_results.tsv", "matched_entity_ids", mt), ("candidate_pairs.tsv", "candidate_entity_ids", cd)):
    with stage(f"writing {name}"):
        pd.DataFrame({"source1_entity_id": ids, hdr: ser.values}).to_csv(OUT / name, sep="\t", index=False, quoting=csv.QUOTE_NONE, lineterminator="\n")
    log(f"   wrote {OUT / name}")

# ---- format checks (subset of the official rules) ----
with stage("validating output format"):
    S2ids = set(TPOOL.entity_id)
    problems = []
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        df = pd.read_csv(OUT / name, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
        if list(df.columns)[0] != "source1_entity_id": problems.append(f"{name}: bad header")
        if len(df) != len(ids) or df.source1_entity_id.tolist() != ids: problems.append(f"{name}: rows != test S1 entities / order")
        if df.source1_entity_id.duplicated().any(): problems.append(f"{name}: duplicate S1 rows")
        for v in df.iloc[:, 1]:
            if v:
                parts = v.split(",")
                if len(parts) != len(set(parts)) or not all(p.startswith(("S2-", "S3-")) for p in parts) or not S2ids.issuperset(parts):
                    problems.append(f"{name}: bad id list {v[:60]}"); break
    cand_map = dict(zip(ids, cd.values)); bad = sum(1 for e, m in zip(ids, mt.values) if m and not set(m.split(",")) <= set(cand_map[e].split(",")))
    if bad: problems.append(f"{bad} matched ids not in candidates")
    log("FORMAT CHECK:", "PASS" if not problems else problems)

off = next(DATA.parent.rglob("validate_submission.py"), None)
if off:
    r = subprocess.run([sys.executable, str(off), "--matching", str(OUT / "matching_results.tsv"),
                        "--candidate", str(OUT / "candidate_pairs.tsv"), "--test-dir", str(DATA / "test")], capture_output=True, text=True)
    print("OFFICIAL VALIDATOR:\n", r.stdout[-1500:], r.stderr[-500:])
else:
    print("official validate_submission.py not found in the dataset; the built-in checks above were used")

# zip the deliverables (Kaggle only keeps /kaggle/working)
with stage("zipping outputs"):
    shutil.make_archive(str(BASE / f"submission_outputs{SUFFIX}"), "zip", OUT)
log("DONE. Upload ONLY matching_results.tsv to the portal. Also keep candidate_pairs.tsv for the final zip.")
log("Run summary:", {c: (round(d["pred_singleton_rate"], 3), round(d["mean_pred_matches"], 2)) for c, d in DIAG.items()}, "(singleton rate, matches/S1)")
''')

md(r"""
## 10. Notes and known limits
* **Unknown:** test orphan prevalence (test pool is larger per S1 than train; cause not identifiable from unlabeled data) and **France accuracy** (no labels). The threshold is therefore taken at the upper end of the flat F0.5 plateau (precision insurance).
* Real-scale F0.5 may be a bit below the validation number because sampled training universes dilute same-name confounders.
* **If the gate fails:** increase `N_TRAIN_S1`, then inspect the per-country report; do not lower `MIN_VAL_F05` just to get a file.
* **Kaggle:** CPU session, 4 cores / ~30 GB RAM. If RAM is tight, lower `P.chunk_records`. All outputs are in `/kaggle/working/output/` (and `submission_outputs.zip`).
* Final package for the organisers still needs `code/`, `README.md`, `requirements.txt` and `Documentation_template.md` (already in the repo) alongside `output/`.
""")

nb["cells"] = C
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "kaggle_end_to_end.ipynb")
print("wrote kaggle_end_to_end.ipynb with", len(C), "cells")
