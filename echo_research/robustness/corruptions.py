from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def apply_corruption(image: np.ndarray, kind: str, severity: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Controlled test-time corruptions for robustness evaluation. Input expected in [0,1]."""
    rng = rng or np.random.default_rng(2026)
    x = image.astype(np.float32).copy()
    if kind == "gaussian_noise":
        x += rng.normal(0.0, 0.02 + 0.08 * severity, size=x.shape).astype(np.float32)
    elif kind == "speckle":
        x += x * rng.normal(0.0, 0.04 + 0.16 * severity, size=x.shape).astype(np.float32)
    elif kind == "blur":
        x = gaussian_filter(x, sigma=0.5 + 2.0 * severity)
    elif kind == "low_contrast":
        factor = max(0.1, 1.0 - 0.8 * severity)
        x = (x - 0.5) * factor + 0.5
    elif kind == "shadow":
        h, w = x.shape[-2:]
        width = max(1, int(w * (0.08 + 0.20 * severity)))
        start = int(rng.integers(0, max(1, w - width)))
        attenuation = 0.65 - 0.5 * severity
        x[..., :, start:start + width] *= max(0.05, attenuation)
    else:
        raise ValueError(f"Unknown corruption {kind!r}")
    return np.clip(x, 0.0, 1.0)
