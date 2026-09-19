from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def _surface(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return mask
    return mask ^ binary_erosion(mask)


def _surface_distances(a: np.ndarray, b: np.ndarray, spacing: Tuple[float, float]) -> np.ndarray:
    sa, sb = _surface(a), _surface(b)
    if not sa.any() or not sb.any():
        return np.array([], dtype=np.float64)
    dt_b = distance_transform_edt(~sb, sampling=spacing)
    return dt_b[sa]


def segmentation_metrics(pred: np.ndarray, target: np.ndarray, spacing: Tuple[float, float] = (1.0, 1.0)) -> Dict[str, float]:
    pred = pred.astype(bool)
    target = target.astype(bool)
    p, t = pred.sum(), target.sum()
    if p == 0 and t == 0:
        return {"dice": 1.0, "iou": 1.0, "precision": 1.0, "recall": 1.0, "hd95": 0.0, "asd": 0.0}
    inter = np.logical_and(pred, target).sum()
    dice = 2.0 * inter / (p + t + 1e-8)
    union = np.logical_or(pred, target).sum()
    iou = inter / (union + 1e-8)
    precision = inter / (p + 1e-8)
    recall = inter / (t + 1e-8)
    d1 = _surface_distances(pred, target, spacing)
    d2 = _surface_distances(target, pred, spacing)
    if len(d1) == 0 or len(d2) == 0:
        hd95, asd = math.inf, math.inf
    else:
        both = np.concatenate([d1, d2])
        hd95 = float(np.percentile(both, 95))
        asd = float(both.mean())
    return {
        "dice": float(dice), "iou": float(iou), "precision": float(precision),
        "recall": float(recall), "hd95": hd95, "asd": asd,
    }
