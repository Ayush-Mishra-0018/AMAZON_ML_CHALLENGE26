"""Metric + diagnostics. Macro F0.5 over S1 entities; singletons score 1.0 iff the prediction is empty."""
import numpy as np


def f05_from_counts(tp, n, nt):
    """Vectorised per-S1 F0.5. tp: true positives, n: predicted, nt: true matches."""
    tp, n, nt = (np.asarray(x, dtype=np.float64) for x in (tp, n, nt))
    with np.errstate(divide="ignore", invalid="ignore"):
        P, R = tp / n, tp / nt
        f = 1.25 * P * R / (0.25 * P + R)
    f = np.where((tp == 0) | (n == 0) | (nt == 0), 0.0, f)
    return np.where(nt == 0, (n == 0).astype(np.float64), f)


def macro_f05(s1_idx, y, n_s1, ntrue):
    """s1_idx/y describe PREDICTED pairs (y=1 if the pair is true).
    ntrue[i] = number of true matches of S1 i (including ones blocking missed)."""
    tp = np.bincount(s1_idx, weights=y, minlength=n_s1)
    n = np.bincount(s1_idx, minlength=n_s1)
    return f05_from_counts(tp, n, ntrue)


def macro_f05_dict(pred: dict, truth: dict) -> float:
    """Reference implementation on {s1_id: set(ids)} dicts (used to unit-test the vectorised version)."""
    out = []
    for e, T in truth.items():
        P = set(pred.get(e, ()))
        if not T:
            out.append(1.0 if not P else 0.0)
            continue
        tp = len(P & T)
        if tp == 0:
            out.append(0.0)
            continue
        p, r = tp / len(P), tp / len(T)
        out.append(1.25 * p * r / (0.25 * p + r))
    return float(np.mean(out))
