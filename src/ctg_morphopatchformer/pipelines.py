"""Pure prediction transformations; no signal preprocessing or model training."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def validate_mask(mask) -> np.ndarray:
    x = np.asarray(mask)
    numeric = np.issubdtype(x.dtype, np.integer) or np.issubdtype(x.dtype, np.floating)
    if x.ndim != 1 or x.size == 0 or not numeric:
        raise ValueError("Expected a nonempty, one-dimensional numeric class mask.")
    if not np.isfinite(x).all() or not np.isin(x, (0, 1, 2)).all():
        raise ValueError("Labels must be exactly 0, 1 or 2; no rounding is performed.")
    return x.astype(np.int64, copy=True)


def validate_probabilities(probabilities) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] != 3:
        raise ValueError("Expected [time,3] probabilities in background/acceleration/deceleration order.")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Probabilities must be finite and between zero and one.")
    if not np.allclose(p.sum(axis=1), 1.0, rtol=0, atol=1e-5):
        raise ValueError("Probabilities must sum to one at each time point.")
    return p


def _runs(mask: np.ndarray, cls: int) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask == cls)
    if not indices.size:
        return []
    split = np.flatnonzero(np.diff(indices) > 1)
    return list(zip(np.r_[indices[0], indices[split + 1]].tolist(),
                    np.r_[indices[split] + 1, indices[-1] + 1].tolist()))


def _integer(value, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def common_mask(mask, *, min_acc: int = 0, min_dec: int = 20,
                merge_gap: int = 15) -> np.ndarray:
    """Merge background-only gaps, then filter short runs; units are samples."""
    out = validate_mask(mask)
    min_acc = _integer(min_acc, "min_acc")
    min_dec = _integer(min_dec, "min_dec")
    merge_gap = _integer(merge_gap, "merge_gap")
    for cls in (1, 2):
        runs = _runs(out, cls)
        for (_, end), (start, _) in zip(runs, runs[1:]):
            if 0 < start - end <= merge_gap and np.all(out[end:start] == 0):
                out[end:start] = cls
    for cls, length in ((1, min_acc), (2, min_dec)):
        for start, end in _runs(out, cls):
            if end - start < length:
                out[start:end] = 0
    return out


@dataclass(frozen=True)
class FullRule:
    acc_threshold: float = 0.38
    dec_threshold: float = 0.50
    event_margin: float = -0.15
    min_length: int = 20


def full_mask(probabilities, rule: FullRule = FullRule()) -> np.ndarray:
    """Frozen probability decoder, zero gap merging; thresholds use sample units."""
    p = validate_probabilities(probabilities)
    if not 0 <= rule.acc_threshold <= 1 or not 0 <= rule.dec_threshold <= 1:
        raise ValueError("Absolute class thresholds must lie in [0,1].")
    if not np.isfinite(rule.event_margin) or not -1 <= rule.event_margin <= 1:
        raise ValueError("event_margin must lie in [-1,1].")
    minimum = _integer(rule.min_length, "min_length")
    bg, acc, dec = p.T
    acc_ok = (acc >= rule.acc_threshold) & (acc - bg >= rule.event_margin)
    dec_ok = (dec >= rule.dec_threshold) & (dec - bg >= rule.event_margin)
    out = np.zeros(len(p), dtype=np.int64)
    out[acc_ok] = 1
    out[dec_ok] = 2
    both = acc_ok & dec_ok
    out[both] = np.where(acc[both] >= dec[both], 1, 2)
    return common_mask(out, min_acc=minimum, min_dec=minimum, merge_gap=0)


def output_masks(probabilities) -> dict[str, np.ndarray]:
    """Return parallel Raw/Common/Full paths from the same [time,3] probabilities."""
    p = validate_probabilities(probabilities)
    raw = p.argmax(axis=1).astype(np.int64)
    return {"Raw": raw, "Common": common_mask(raw), "Full pipeline": full_mask(p)}
