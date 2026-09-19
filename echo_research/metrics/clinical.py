from __future__ import annotations

from typing import Tuple

import numpy as np


def _principal_width_profile(mask: np.ndarray, spacing: Tuple[float, float], n_discs: int = 20):
    pts = np.argwhere(mask > 0)
    if len(pts) < 10:
        return np.zeros(n_discs), 0.0
    pts_mm = pts.astype(np.float64) * np.asarray(spacing, dtype=np.float64)[None, :]
    centered = pts_mm - pts_mm.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    major = vecs[:, np.argmax(vals)]
    minor = np.array([-major[1], major[0]])
    t = centered @ major
    u = centered @ minor
    lo, hi = float(t.min()), float(t.max())
    length = hi - lo
    if length <= 1e-6:
        return np.zeros(n_discs), 0.0
    edges = np.linspace(lo, hi, n_discs + 1)
    widths = np.zeros(n_discs, dtype=np.float64)
    for i in range(n_discs):
        sel = (t >= edges[i]) & (t <= edges[i + 1] if i == n_discs - 1 else t < edges[i + 1])
        if sel.any():
            widths[i] = float(u[sel].max() - u[sel].min())
    nz = np.flatnonzero(widths > 0)
    if len(nz) >= 2:
        widths = np.interp(np.arange(n_discs), nz, widths[nz])
    return widths, length


def estimate_biplane_volume_ml(mask_2ch: np.ndarray, spacing_2ch: Tuple[float, float], mask_4ch: np.ndarray, spacing_4ch: Tuple[float, float], n_discs: int = 20) -> float:
    """Approximate biplane Simpson volume from two orthogonal LV masks.

    IMPORTANT: this estimator is provided for reproducible experimentation, not as a
    claim that it exactly matches the CAMUS challenge's clinical evaluation code.
    Validate it against official/reference clinical values before reporting mL in a paper.
    """
    d2, l2 = _principal_width_profile(mask_2ch, spacing_2ch, n_discs)
    d4, l4 = _principal_width_profile(mask_4ch, spacing_4ch, n_discs)
    if l2 <= 0 or l4 <= 0:
        return 0.0
    length = max(l2, l4)
    h = length / n_discs
    volume_mm3 = np.sum((np.pi / 4.0) * d2 * d4 * h)
    return float(volume_mm3 / 1000.0)


def ejection_fraction(edv_ml: float, esv_ml: float) -> float:
    if edv_ml <= 1e-8:
        return float("nan")
    return float((edv_ml - esv_ml) / edv_ml)
