from __future__ import annotations

import numpy as np
from scipy import stats


def bootstrap_ci(values, confidence: float = 0.95, n_resamples: int = 5000, seed: int = 2026):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(n_resamples)])
    a = (1.0 - confidence) / 2.0
    return float(np.quantile(means, a)), float(np.quantile(means, 1.0 - a))


def paired_wilcoxon(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 2:
        return float("nan")
    return float(stats.wilcoxon(a[keep], b[keep]).pvalue)
