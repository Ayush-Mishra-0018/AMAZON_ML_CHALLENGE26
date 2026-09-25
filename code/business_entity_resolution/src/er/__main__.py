"""CLI (run from code/business_entity_resolution/src):  python -m er train | infer | all

Every step prints timestamped progress; the same lines are appended to work/run.log."""
import json
import sys
import time
import traceback
from .config import Params, WORK_DIR, OUT_DIR, find_data_dir
from .progress import log, banner, _fmt

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(line_buffering=True, errors="replace")
    except Exception:
        pass
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd not in ("train", "infer", "all"):
        sys.exit("usage: python -m er [train|infer|all]")
    from . import pipeline as pl          # imported after stdout is line-buffered
    t0 = time.time()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    P = Params()
    banner(f"BUSINESS ENTITY RESOLUTION  |  command: {cmd}")
    log(f"data dir : {find_data_dir()}")
    log(f"work dir : {WORK_DIR}   (live log: {WORK_DIR / 'run.log'})")
    log(f"out dir  : {OUT_DIR}")
    log(f"workers  : {P.n_workers} | chunk_records {P.chunk_records:,} | top-K {P.top_k_per_record} | accuracy gate F0.5 >= {P.min_val_f05}")
    log("first run converts the big TSVs to parquet (several minutes); later runs reuse the cache")
    try:
        if cmd in ("train", "all"):
            rep, _, _ = pl.train(P)
            (WORK_DIR / "validation_report.json").write_text(json.dumps(rep, indent=1))
            if not rep["passes_gate"]:
                log("STOPPING: validation accuracy is below the gate, no submission will be produced.")
                sys.exit(2)
        if cmd in ("infer", "all"):
            ids, m, c, diag = pl.infer(P)
            pl.write_outputs(ids, m, c)
            (WORK_DIR / "test_diagnostics.json").write_text(json.dumps(diag, indent=1))
            banner("DONE")
            log(f"submission files: {OUT_DIR / 'matching_results.tsv'}  and  {OUT_DIR / 'candidate_pairs.tsv'}")
            log("next: run utils/validate_submission.py (command in README.md)")
        log(f"total run time {_fmt(time.time() - t0)}")
    except SystemExit:
        raise
    except BaseException:
        log("FAILED after " + _fmt(time.time() - t0) + "\n" + traceback.format_exc())
        sys.exit(1)
