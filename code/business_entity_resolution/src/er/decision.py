"""Threshold selection.

F0.5 vs threshold is flat near its optimum (US: 0.940-0.942 for thresholds 0.5-0.8), so we take the HIGHEST
threshold within ``plateau_tol`` of the best: precision insurance against an unknown shift in orphan rate
(test orphan prevalence cannot be inferred from unlabeled data - see notebook section 9).
"""
import numpy as np
from .evaluate import macro_f05


def scan_thresholds(s1_idx, p, y, n_s1, ntrue, grid=None):
    grid = np.round(np.arange(0.30, 0.96, 0.05), 2) if grid is None else grid
    res = {}
    for th in grid:
        m = p >= th
        res[float(th)] = float(macro_f05(s1_idx[m], y[m], n_s1, ntrue).mean())
    return res


def choose_threshold(scores: dict, tol: float):
    best = max(scores.values())
    return max(t for t, v in scores.items() if v >= best - tol), best


def apply_one_owner(df, p_col="p2"):
    """Optional (off by default): each pool record keeps only its highest-scoring S1."""
    idx = df.groupby("pool_i")[p_col].idxmax()
    return df.loc[idx]
