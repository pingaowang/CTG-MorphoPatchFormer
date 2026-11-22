"""Event-level precision / recall / F1 / mean IoU / boundary MAE.

All numbers are computed **per class** plus a **macro average** over the two
event classes (background never contributes). Mean IoU and boundary MAEs use
only matched events; precision/recall/F1 use the full TP/FP/FN counts.

Zero-division policy (documented in :class:`EventMetrics` and report metadata):

- ``precision = TP / (TP + FP)``; ``0.0`` when ``TP + FP == 0`` (no predictions).
- ``recall = TP / (TP + FN)``; ``0.0`` when ``TP + FN == 0`` (no ground truth).
- ``f1 = 2 P R / (P + R)``; ``0.0`` when ``P + R == 0``.
- ``mean_iou``, ``start_mae``, ``end_mae``, ``duration_mae`` use only matched
  pairs. When no pair exists, the value is **NaN** (not 0.0) so the user can
  distinguish "perfect localisation" from "nothing to localise".
- Macro averages are computed only over per-class values that are *finite*. If
  both per-class values are NaN, the macro average is NaN.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

from .event_extraction import Event
from .event_matching import (
    MatchedPair,
    MatchingResult,
    match_events_per_class,
)


@dataclass
class PerClassEventMetrics:
    """Event-level metrics for one class."""

    cls: str
    n_gt: int
    n_pred: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    mean_iou: float
    start_mae: float
    end_mae: float
    duration_mae: float
    matched_pairs: list[MatchedPair] = field(default_factory=list)


@dataclass
class EventMetrics:
    """Event-level metrics aggregated across the two event classes.

    Attributes:
        per_class: keyed by class name (``"acceleration"`` / ``"deceleration"``).
        macro_precision / macro_recall / macro_f1: arithmetic mean over the
            event classes (NaN-tolerant; see module docstring).
        macro_mean_iou / macro_start_mae / macro_end_mae / macro_duration_mae:
            arithmetic mean over per-class values for matched pairs.
        iou_threshold: threshold used for matching.
        strategy: matching strategy used.
        time_unit_scale: multiplier applied to convert frame indices into the
            unit used by ``start_mae`` / ``end_mae`` / ``duration_mae``.
            ``1.0`` keeps frames; ``1 / fs_hz`` converts to seconds.
    """

    per_class: Mapping[str, PerClassEventMetrics]
    macro_precision: float
    macro_recall: float
    macro_f1: float
    macro_mean_iou: float
    macro_start_mae: float
    macro_end_mae: float
    macro_duration_mae: float
    iou_threshold: float
    strategy: str
    time_unit_scale: float


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def _nan_mean(values: Iterable[float]) -> float:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    finite = arr[np.isfinite(arr)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _per_class_from_matching(
    cls: str, n_gt: int, n_pred: int, result: MatchingResult, *, time_unit_scale: float
) -> PerClassEventMetrics:
    tp = result.n_tp
    fp = result.n_fp
    fn = result.n_fn
    p = _safe_div(tp, tp + fp)
    r = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * p * r, p + r)

    if result.matched:
        ious = [m.iou for m in result.matched]
        start_errs = [abs(m.start_error) * time_unit_scale for m in result.matched]
        end_errs = [abs(m.end_error) * time_unit_scale for m in result.matched]
        dur_errs = [abs(m.duration_error) * time_unit_scale for m in result.matched]
        mean_iou = float(np.mean(ious))
        start_mae = float(np.mean(start_errs))
        end_mae = float(np.mean(end_errs))
        duration_mae = float(np.mean(dur_errs))
    else:
        mean_iou = float("nan")
        start_mae = float("nan")
        end_mae = float("nan")
        duration_mae = float("nan")

    return PerClassEventMetrics(
        cls=cls,
        n_gt=int(n_gt),
        n_pred=int(n_pred),
        tp=int(tp),
        fp=int(fp),
        fn=int(fn),
        precision=float(p),
        recall=float(r),
        f1=float(f1),
        mean_iou=mean_iou,
        start_mae=start_mae,
        end_mae=end_mae,
        duration_mae=duration_mae,
        matched_pairs=list(result.matched),
    )


def compute_event_metrics(
    gt_events: Sequence[Event],
    pred_events: Sequence[Event],
    *,
    classes: Sequence[str] = ("acceleration", "deceleration"),
    iou_threshold: float = 0.3,
    strategy: str = "greedy",
    fs_hz: float | None = None,
) -> EventMetrics:
    """Compute event-level metrics from two lists of events.

    Args:
        gt_events: ground-truth events (already extracted from the GT mask).
        pred_events: predicted events.
        classes: event class names to evaluate, in the order used for macro
            averaging. Background ("background") is intentionally excluded.
        iou_threshold: matching threshold passed to :func:`match_events`.
        strategy: matching strategy (``"greedy"`` or ``"hungarian"``).
        fs_hz: if given, ``start_mae`` / ``end_mae`` / ``duration_mae`` are
            returned in **seconds** (``time_unit_scale = 1 / fs_hz``). When
            ``None``, errors stay in **frames** (``time_unit_scale = 1.0``).

    Returns:
        :class:`EventMetrics` with per-class and macro values.
    """
    if fs_hz is not None:
        if fs_hz <= 0:
            raise ValueError(f"fs_hz must be positive, got {fs_hz!r}")
        time_unit_scale = 1.0 / float(fs_hz)
    else:
        time_unit_scale = 1.0

    per_class_match = match_events_per_class(
        gt_events,
        pred_events,
        classes=classes,
        iou_threshold=iou_threshold,
        strategy=strategy,  # type: ignore[arg-type]
    )

    per_class: dict[str, PerClassEventMetrics] = {}
    for cls in classes:
        gt_n = sum(1 for e in gt_events if e.cls == cls)
        pred_n = sum(1 for e in pred_events if e.cls == cls)
        per_class[cls] = _per_class_from_matching(
            cls, gt_n, pred_n, per_class_match[cls], time_unit_scale=time_unit_scale
        )

    # Classes with no GT *and* no prediction are "truly absent" and excluded
    # from macro averaging so they cannot drag the score down. Classes with
    # only one side populated still contribute (precision/recall = 0) so the
    # model is correctly penalised for missing or hallucinating events.
    active_classes = [cls for cls in classes if not (per_class[cls].n_gt == 0 and per_class[cls].n_pred == 0)]

    def _macro_scalar(attr: str) -> float:
        if not active_classes:
            return float("nan")
        vals = [getattr(per_class[cls], attr) for cls in active_classes]
        finite = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
        return float(np.mean(finite)) if finite else float("nan")

    return EventMetrics(
        per_class=per_class,
        macro_precision=_macro_scalar("precision"),
        macro_recall=_macro_scalar("recall"),
        macro_f1=_macro_scalar("f1"),
        macro_mean_iou=_nan_mean([per_class[c].mean_iou for c in active_classes]),
        macro_start_mae=_nan_mean([per_class[c].start_mae for c in active_classes]),
        macro_end_mae=_nan_mean([per_class[c].end_mae for c in active_classes]),
        macro_duration_mae=_nan_mean([per_class[c].duration_mae for c in active_classes]),
        iou_threshold=float(iou_threshold),
        strategy=str(strategy),
        time_unit_scale=float(time_unit_scale),
    )
