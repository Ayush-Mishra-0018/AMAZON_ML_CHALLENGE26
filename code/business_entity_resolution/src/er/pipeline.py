"""End-to-end orchestration: ``train`` (validate + fit) and ``infer`` (test -> TSVs). Reports progress as it goes."""
import json
import time
import zlib
import numpy as np
import pandas as pd
import lightgbm as lgb
from .config import Params, WORK_DIR, OUT_DIR
from .data import load_split, make_universe
from .blocking import BlockingEngine
from .features import Records, pair_features, STAGE1
from .relational import relational_features, REL_F
from .decision import scan_thresholds, choose_threshold
from .evaluate import macro_f05
from .progress import log, stage, banner, Progress, table, _fmt

STAGE2 = ["p1"] + STAGE1 + REL_F
MODEL_DIR = WORK_DIR / "models"


def candidate_pairs_for(S1: Records, P: Records, params: Params, pool_df, s1_df, on_chunk):
    """Blocking + stage-1 features per chunk of pool records; ``on_chunk(pairs, feats)`` consumes each chunk."""
    with BlockingEngine(s1_df.business_name, s1_df.business_address, params) as eng:
        log(f"   index ready: {eng.n_s1:,} S1, {eng.n_keys / 1e6:.1f}M keys, key-frequency cap = {eng.cap}")
        n_chunks = (len(pool_df) + params.chunk_records - 1) // params.chunk_records
        total_pairs, t0 = 0, time.time()
        for ci, pairs in enumerate(eng.stream(pool_df.business_name, pool_df.business_address), 1):
            if pairs is None:
                log(f"   chunk {ci}/{n_chunks}: no candidates")
                continue
            with stage(f"chunk {ci}/{n_chunks}: computing {len(pairs):,} pair features + scoring"):
                on_chunk(pairs, pair_features(pairs, S1, P, eng.cap))
            total_pairs += len(pairs)
            done = min(ci * params.chunk_records, len(pool_df))
            el = time.time() - t0
            log(f"   chunks done {ci}/{n_chunks} | {done:,}/{len(pool_df):,} pool records | {total_pairs:,} pairs | "
                f"ETA for this country {_fmt(el / done * (len(pool_df) - done))}")


# ------------------------------------------------------------------ training / validation
def build_training_country(s1, pool, gt, country, n_s1, params):
    banner(f"TRAIN DATA: {country}")
    pick, upool, truth = make_universe(s1, pool, gt, country, n_s1, params.seed)
    log(f"   universe: {len(pick):,} S1, {len(upool):,} S2/S3 records, {sum(len(t) for t in truth.values()):,} true pairs")
    with stage("normalising records (transliteration, phonetic skeletons, number sets)"):
        S1, P = Records(pick), Records(upool)
    with stage("attaching ground-truth owners"):
        s1_index = {e: i for i, e in enumerate(pick.entity_id)}
        owner = np.full(len(upool), -1, np.int64)
        pid = {e: i for i, e in enumerate(upool.entity_id)}
        for e, T in truth.items():
            for m in T:
                owner[pid[m]] = s1_index[e]
        ntrue = np.array([len(truth[e]) for e in pick.entity_id])
    parts = []
    candidate_pairs_for(S1, P, params, upool, pick,
                        lambda pr, f: parts.append(pd.concat([pr[["s1_i", "pool_i"]], f], axis=1)))
    D = pd.concat(parts, ignore_index=True)
    D["y"] = (owner[D.pool_i.values] == D.s1_i.values).astype(np.int8)
    # folds are grouped by S1 name so same-name (chain) S1 records never straddle train/validation
    fold = np.array([zlib.crc32(x.encode()) % params.n_folds for x in S1.nn])
    D["fold"] = fold[D.s1_i.values]
    D["country"] = country
    log(f"   RESULT {country}: {len(D):,} candidate pairs ({len(D) / len(pick):.0f} per S1) | "
        f"true pairs captured by blocking: {D.y.sum():,}/{ntrue.sum():,} = {D.y.sum() / ntrue.sum():.4f}")
    return dict(country=country, S1=S1, P=P, D=D, ntrue=ntrue, n_s1=len(pick))


def _oof(X, y, fold, params_lgb, n_folds, seed, name):
    p = np.zeros(len(X), np.float32)
    for f in range(n_folds):
        tr = fold != f
        with stage(f"{name}: fold {f + 1}/{n_folds} (train {tr.sum():,} rows -> predict {(~tr).sum():,})", 30):
            m = lgb.LGBMClassifier(**params_lgb, random_state=seed, n_jobs=-1).fit(X[tr], y[tr])
            p[~tr] = m.predict_proba(X[~tr])[:, 1]
    return p


def train(params: Params = Params(), n_s1_per_country=None):
    """Fit both stages on cluster-closed universes; pick the threshold from grouped out-of-fold predictions."""
    t_all = time.time()
    n_s1_per_country = n_s1_per_country or {"US": 150_000, "India": 150_000}
    banner("TRAINING + VALIDATION  (step 1/5: load labelled data)")
    s1, pool, gt = load_split("train")
    log(f"   train totals: {len(s1):,} S1, {len(pool):,} pool records; universes: {n_s1_per_country}")
    banner("step 2/5: blocking + features per country")
    C = [build_training_country(s1, pool, gt, c, n, params) for c, n in n_s1_per_country.items()]
    D = pd.concat([c["D"] for c in C], ignore_index=True)
    off = np.cumsum([0] + [c["n_s1"] for c in C])
    D["gs1"] = D.s1_i.values + np.repeat(off[:-1], [len(c["D"]) for c in C])
    ntrue_all = np.concatenate([c["ntrue"] for c in C])
    n_all = int(off[-1])
    report = {"candidate_recall_pairs": float(D.y.sum() / ntrue_all.sum())}
    pos = D[D.y == 1]     # ceiling = F0.5 if scoring were perfect inside the candidate set
    report["ceiling_F05"] = float(macro_f05(pos.gs1.values, np.ones(len(pos)), n_all, ntrue_all).mean())
    log(f"   COMBINED: {len(D):,} candidate pairs | candidate recall {report['candidate_recall_pairs']:.4f} | "
        f"ceiling F0.5 (perfect scorer) {report['ceiling_F05']:.4f}")

    banner("step 3/5: stage-1 pair scorer (grouped out-of-fold)")
    X = D[STAGE1].values
    D["p1"] = _oof(X, D.y.values, D.fold.values, params.lgb_stage1, params.n_folds, params.seed, "stage 1")
    s1_scores = scan_thresholds(D.gs1.values, D.p1.values, D.y.values, n_all, ntrue_all)
    report["stage1_F05_by_threshold"] = s1_scores
    table("   stage-1 macro F0.5 by threshold:", s1_scores)
    log(f"   stage-1 best F0.5 = {max(s1_scores.values()):.4f}")

    banner("step 4/5: stage-2 S2<->S3 consensus re-scorer")
    S = D[D.p1 >= params.stage1_min_p].copy()
    log(f"   {len(S):,} of {len(D):,} pairs survive stage 1 (p >= {params.stage1_min_p})")
    rel = []
    for c in C:
        with stage(f"relational features for {c['country']}"):
            idx = S.index[S.country.values == c["country"]]
            sub = S.loc[idx, ["s1_i", "pool_i", "p1", "isS2"]].reset_index(drop=True)
            r = relational_features(sub, c["P"])
            r.index = idx
            rel.append(r)
    S = S.join(pd.concat(rel))
    S["p2"] = _oof(S[STAGE2].values, S.y.values, S.fold.values, params.lgb_stage2, params.n_folds, params.seed, "stage 2")
    s2_scores = scan_thresholds(S.gs1.values, S.p2.values, S.y.values, n_all, ntrue_all)
    th, best = choose_threshold(s2_scores, params.plateau_tol)
    report.update(stage2_F05_by_threshold=s2_scores, threshold=th,
                  stage2_F05_at_threshold=s2_scores[th], stage2_best=best)
    table("   stage-2 macro F0.5 by threshold:", s2_scores)
    log(f"   chosen threshold {th} (highest within {params.plateau_tol} of best {best:.4f})")
    m = S.p2.values >= th
    per = {}
    for c in C:
        mm = m & (S.country.values == c["country"])
        f = macro_f05(S.s1_i.values[mm], S.y.values[mm], c["n_s1"], c["ntrue"])
        per[c["country"]] = dict(F05=float(f.mean()), singleton_F05=float(f[c["ntrue"] == 0].mean()),
                                 nonsingleton_F05=float(f[c["ntrue"] > 0].mean()))
    report["per_country"] = per
    tp = S.y.values[m].sum()
    report["micro_precision"] = float(tp / m.sum())
    report["micro_recall_vs_all_true"] = float(tp / ntrue_all.sum())

    banner("step 5/5: accuracy gate + final model fit")
    report["gate_min_F05"] = params.min_val_f05
    report["passes_gate"] = bool(s2_scores[th] >= params.min_val_f05 and all(v["F05"] >= params.min_val_f05 for v in per.values()))
    log(f"   ceiling F0.5 (blocking limit)   : {report['ceiling_F05']:.4f}")
    log(f"   OVERALL validation macro F0.5   : {s2_scores[th]:.4f}")
    for k, v in per.items():
        log(f"   {k:>6}: F0.5 {v['F05']:.4f} | singletons {v['singleton_F05']:.4f} | non-singletons {v['nonsingleton_F05']:.4f}")
    log(f"   micro precision {report['micro_precision']:.4f} | micro recall {report['micro_recall_vs_all_true']:.4f}")
    log(f"   ACCURACY GATE (>= {params.min_val_f05}): {'PASS' if report['passes_gate'] else 'FAIL - infer will refuse to write a submission'}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with stage("fitting final stage-1 model on all pairs", 30):
        m1 = lgb.LGBMClassifier(**params.lgb_stage1, random_state=params.seed, n_jobs=-1).fit(X, D.y.values)
    with stage("fitting final stage-2 model", 30):
        m2 = lgb.LGBMClassifier(**params.lgb_stage2, random_state=params.seed, n_jobs=-1).fit(S[STAGE2].values, S.y.values)
    m1.booster_.save_model(str(MODEL_DIR / "stage1.txt"))
    m2.booster_.save_model(str(MODEL_DIR / "stage2.txt"))
    (MODEL_DIR / "meta.json").write_text(json.dumps(dict(threshold=th, report=report), indent=1))
    S[["country", "s1_i", "gs1", "pool_i", "y", "p1", "p2"]].to_parquet(WORK_DIR / "oof_train.parquet")
    log(f"   models saved to {MODEL_DIR}; total training time {_fmt(time.time() - t_all)}")
    return report, D, S


# ------------------------------------------------------------------ inference
def infer(params: Params = Params(), split: str = "test", countries=None, max_s1=None):
    """Blocking -> stage 1 -> stage 2 -> threshold, for every country label found (open set: France included)."""
    t_all = time.time()
    banner("INFERENCE  (step 1: gate check + load)")
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    if not meta["report"].get("passes_gate", False):
        raise RuntimeError(f"Validation F0.5 {meta['report']['stage2_F05_at_threshold']:.4f} is below the gate "
                           f"{meta['report']['gate_min_F05']}; refusing to produce a submission. Improve the model first.")
    log(f"   gate passed (validation F0.5 {meta['report']['stage2_F05_at_threshold']:.4f}); threshold {meta['threshold']}")
    s1, pool, _ = load_split(split)
    b1 = lgb.Booster(model_file=str(MODEL_DIR / "stage1.txt"))
    b2 = lgb.Booster(model_file=str(MODEL_DIR / "stage2.txt"))
    th = meta["threshold"]
    matches, cands, diag = {}, {}, {}
    todo = countries or sorted(s1.country.unique())
    log(f"   countries to process: {todo}")
    for k, country in enumerate(todo, 1):
        banner(f"INFERENCE: country {k}/{len(todo)} = {country}")
        s1c = s1[s1.country == country].reset_index(drop=True)
        if max_s1:
            s1c = s1c.head(max_s1)
        pc = pool[pool.country == country].reset_index(drop=True)
        log(f"   {len(s1c):,} S1 entities, {len(pc):,} S2/S3 records")
        with stage("normalising records"):
            S1, P = Records(s1c), Records(pc)
        cand_parts, surv = [], []

        def on_chunk(pairs, f):
            cand_parts.append(pairs[["s1_i", "pool_i"]].values.astype(np.int32))
            p1 = b1.predict(f.values)
            kk = p1 >= params.stage1_min_p
            surv.append(pd.concat([pairs.loc[kk, ["s1_i", "pool_i"]], f[kk]], axis=1).assign(p1=p1[kk]))

        candidate_pairs_for(S1, P, params, pc, s1c, on_chunk)
        C = np.concatenate(cand_parts)
        R = pd.concat(surv, ignore_index=True)
        log(f"   {len(C):,} candidate pairs; {len(R):,} survive stage 1")
        with stage("stage 2: consensus features + scoring"):
            R = R.join(relational_features(R[["s1_i", "pool_i", "p1", "isS2"]], P))
            R["p2"] = b2.predict(R[STAGE2].values)
        sel = R[R.p2 >= th]
        if params.one_owner:
            from .decision import apply_one_owner
            sel = apply_one_owner(sel)
        with stage("assembling per-S1 match and candidate lists"):
            for grp, store in ((sel, matches), (pd.DataFrame(C, columns=["s1_i", "pool_i"]), cands)):
                g = grp.groupby("s1_i").pool_i.agg(lambda x: ",".join(P.ids[x.values]))
                for i, v in g.items():
                    store[S1.ids[i]] = v
        n_pred = np.bincount(sel.s1_i.values, minlength=len(s1c))
        diag[country] = dict(n_s1=len(s1c), n_pool=len(pc), cand_pairs=int(len(C)), mean_cand_per_s1=len(C) / len(s1c),
                             s1_without_candidate=float(1 - len(np.unique(C[:, 0])) / len(s1c)),
                             pred_singleton_rate=float((n_pred == 0).mean()), mean_pred_matches=float(n_pred.mean()),
                             pool_records_matched_frac=float(sel.pool_i.nunique() / len(pc)))
        d = diag[country]
        log(f"   RESULT {country}: {d['mean_cand_per_s1']:.0f} candidates/S1 | predicted singletons {d['pred_singleton_rate']:.3f} "
            f"(train truth ~0.056) | {d['mean_pred_matches']:.2f} matches/S1 (train truth ~3.46) | "
            f"pool records matched {d['pool_records_matched_frac']:.3f}")
        del R, C, sel
    log(f"   inference finished in {_fmt(time.time() - t_all)}")
    return s1.entity_id.tolist(), matches, cands, diag


def write_outputs(s1_ids, matches, cands, out_dir=OUT_DIR):
    """One row per S1 entity (empty list for singletons). Matches are a subset of candidates by construction."""
    banner("WRITING SUBMISSION FILES")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, hdr, d in (("matching_results.tsv", "matched_entity_ids", matches),
                         ("candidate_pairs.tsv", "candidate_entity_ids", cands)):
        prog = Progress(len(s1_ids), f"writing {name}", unit=" rows")
        with open(out_dir / name, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"source1_entity_id\t{hdr}\n")
            for i, e in enumerate(s1_ids, 1):
                f.write(f"{e}\t{d.get(e, '')}\n")
                if i % 50_000 == 0:
                    prog.update(50_000)
        prog.update(len(s1_ids) - prog.n)
        log(f"   wrote {out_dir / name}")
