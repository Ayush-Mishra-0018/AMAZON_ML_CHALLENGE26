"""Paths and hyper-parameters. Every tunable lives here so experiments are reproducible.

Data location: set ER_DATA_DIR to the folder that contains ``train/`` and ``test/`` (the official
``dataset/`` folder). If unset we search the repository for it. Intermediate files go to ER_WORK_DIR
(default ``<repo>/work``, git-ignored); final files go to ER_OUT_DIR (default ``<repo>/output``).
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]


def find_data_dir() -> Path:
    env = os.environ.get("ER_DATA_DIR")
    if env:
        return Path(env)
    for p in REPO.glob("Data/**/dataset/train/train_source1.tsv"):
        return p.parents[1]
    raise FileNotFoundError("Set ER_DATA_DIR to the folder containing train/ and test/")


WORK_DIR = Path(os.environ.get("ER_WORK_DIR", REPO / "work"))
OUT_DIR = Path(os.environ.get("ER_OUT_DIR", REPO / "output"))


@dataclass
class Params:
    # ---- blocking (measured: conjunctive keys >> single tokens; see notebook section 3)
    cap_frac: float = 3e-5        # a key is usable if it is shared by <= max(cap_min, cap_frac * n_S1) S1 records
    cap_min: int = 8
    top_k_per_record: int = 8     # S1 candidates kept per S2/S3 record (ranked by key score)
    chunk_records: int = 400_000  # pool records processed per pass (bounds RAM)
    task_records: int = 8_000     # pool records per worker task
    n_workers: int = max(1, (os.cpu_count() or 4) - 2)
    # ---- model
    stage1_min_p: float = 0.05    # rows below this after stage 1 are dropped (measured: 98% of rows, ~0 true pairs)
    lgb_stage1: dict = field(default_factory=lambda: dict(
        n_estimators=250, learning_rate=0.06, num_leaves=63, min_child_samples=40,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1))
    lgb_stage2: dict = field(default_factory=lambda: dict(
        n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=30, verbose=-1))
    n_folds: int = 3
    # ---- decision
    plateau_tol: float = 0.001    # choose the HIGHEST threshold within tol of the best OOF F0.5 (precision insurance)
    min_val_f05: float = 0.90     # ACCURACY GATE: no submission files are written unless grouped-OOF macro F0.5 >= this
    one_owner: bool = False       # measured ~neutral (+0.0003 at best threshold) -> off by default
    seed: int = 0
