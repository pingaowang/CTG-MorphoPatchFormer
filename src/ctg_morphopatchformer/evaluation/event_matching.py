"""Temporal IoU and one-to-one matching between predicted and ground-truth events.

Matching rules (deterministic):

1. Only events of the **same record and class** can match.
2. A predicted event can match **at most one** ground-truth event, and vice
   versa.
3. Two strategies are supported, controlled by ``strategy``:

   - ``"greedy"`` (default): repeatedly take the pair with the highest IoU
     above ``iou_threshold`` that has not been used on either side. Ties are
     broken by smaller ``gt.start``, then smaller ``pred.start``, then by
     stored insertion order so output is reproducible.
   - ``"hungarian"``: solve the maximum-weight bipartite matching using
     ``scipy.optimize.linear_sum_assignment`` on the IoU matrix; only pairs
     with IoU >= ``iou_threshold`` are kept. Falls back to greedy if scipy is
     unavailable.

The matcher returns a :class:`MatchingResult` containing the matched pairs
plus the unmatched ground-truth (false negatives) and unmatched predictions
(false positives). All matched pairs carry per-pair diagnostics so downstream
metrics never have to recompute them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from .event_extraction import Event

try:  # pragma: no cover - import guard only
    from scipy.optimize import linear_sum_assignment as _scipy_lsa
except Exception:  # pragma: no cover
    _scipy_lsa = None


Strategy = Literal["greedy", "hungarian"]


def temporal_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    """1-D IoU on half-open intervals.

    Args:
        a: ``(start, end)`` with ``end >= start``.
        b: ``(start, end)`` with ``end >= start``.

    Returns:
        ``intersection / union`` in ``[0, 1]``. ``0.0`` when intervals are
        disjoint or either has zero length.
    """
    a0, a1 = float(a[0]), float(a[1])
    b0, b1 = float(b[0]), float(b[1])
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    if inter <= 0.0:
        return 0.0
    union = (max(a1, b1) - min(a0, b0))
    if union <= 0.0:
        return 0.0
    return float(inter / union)


@dataclass(frozen=True)
class MatchedPair:
    """One matched (GT, prediction) pair with per-pair diagnostics."""

    gt: Event
    pred: Event
    iou: float
    start_error: float
    end_error: float
    duration_error: float


@dataclass
class MatchingResult:
    """Outcome of :func:`match_events`.

    Attributes:
        matched: list of :class:`MatchedPair`.
        unmatched_gt: ground-truth events that no prediction matched (FN).
        unmatched_pred: predicted events that did not match any GT (FP).
        cls: class name this result is for. For results aggregated across all
            event classes (e.g. by :func:`match_events_per_class`) this is the
            string ``"<all>"``.
        iou_threshold: threshold actually used.
        strategy: strategy actually used (``"greedy"`` or ``"hungarian"``).
    """

    matched: list[MatchedPair]
    unmatched_gt: list[Event]
    unmatched_pred: list[Event]
    cls: str
    iou_threshold: float
    strategy: str

    @property
    def n_tp(self) -> int:
        return len(self.matched)

    @property
    def n_fp(self) -> int:
        return len(self.unmatched_pred)

    @property
    def n_fn(self) -> int:
        return len(self.unmatched_gt)


def _pair_diagnostics(gt: Event, pred: Event, iou: float) -> MatchedPair:
    return MatchedPair(
        gt=gt,
        pred=pred,
        iou=float(iou),
        start_error=float(pred.start) - float(gt.start),
        end_error=float(pred.end) - float(gt.end),
        duration_error=float(pred.duration) - float(gt.duration),
    )


def _greedy_match(
    gts: Sequence[Event],
    preds: Sequence[Event],
    iou_threshold: float,
) -> tuple[list[MatchedPair], set[int], set[int]]:
    candidates: list[tuple[float, float, float, int, int]] = []
    for i, g in enumerate(gts):
        for j, p in enumerate(preds):
            if g.case_id != p.case_id or g.cls != p.cls:
                continue
            iou = temporal_iou((g.start, g.end), (p.start, p.end))
            if iou >= iou_threshold and iou > 0.0:
                candidates.append((-iou, float(g.start), float(p.start), i, j))
    candidates.sort()
    used_g: set[int] = set()
    used_p: set[int] = set()
    matched: list[MatchedPair] = []
    for neg_iou, _gs, _ps, i, j in candidates:
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        matched.append(_pair_diagnostics(gts[i], preds[j], -neg_iou))
    return matched, used_g, used_p


def _hungarian_match(
    gts: Sequence[Event],
    preds: Sequence[Event],
    iou_threshold: float,
) -> tuple[list[MatchedPair], set[int], set[int]]:
    if _scipy_lsa is None or not gts or not preds:
        return _greedy_match(gts, preds, iou_threshold)
    n_g = len(gts)
    n_p = len(preds)
    cost = [[0.0] * n_p for _ in range(n_g)]
    for i, g in enumerate(gts):
        for j, p in enumerate(preds):
            if g.case_id == p.case_id and g.cls == p.cls:
                cost[i][j] = -temporal_iou((g.start, g.end), (p.start, p.end))
    row_ind, col_ind = _scipy_lsa(cost)
    used_g: set[int] = set()
    used_p: set[int] = set()
    matched: list[MatchedPair] = []
    for i, j in zip(row_ind.tolist(), col_ind.tolist()):
        iou = -cost[int(i)][int(j)]
        if iou >= iou_threshold and iou > 0.0:
            used_g.add(int(i))
            used_p.add(int(j))
            matched.append(_pair_diagnostics(gts[int(i)], preds[int(j)], iou))
    return matched, used_g, used_p


def match_events(
    gt_events: Sequence[Event],
    pred_events: Sequence[Event],
    *,
    iou_threshold: float = 0.3,
    strategy: Strategy = "greedy",
    cls: str = "<all>",
) -> MatchingResult:
    """One-to-one match predictions to ground-truth events for a single class.

    Only pairs from the same record and class are eligible. Use
    :func:`match_events_per_class` for separate results by class.

    Args:
        gt_events: ground-truth events.
        pred_events: predicted events.
        iou_threshold: minimum IoU for a pair to be kept (default ``0.3``).
        strategy: ``"greedy"`` (descending IoU) or ``"hungarian"`` (optimal
            assignment).
        cls: optional class label stored on the result (default ``"<all>"``).

    Returns:
        :class:`MatchingResult` with matched pairs and unmatched lists.

    Edge cases (explicit; never silently raise):
        - Both empty: returns an empty result (no TP/FP/FN).
        - Empty predictions: all GT become FN.
        - Empty ground truth: all predictions become FP.
        - ``iou_threshold <= 0``: any non-disjoint pair is eligible.
    """
    if iou_threshold < 0.0 or iou_threshold > 1.0:
        raise ValueError(f"iou_threshold must be in [0,1], got {iou_threshold!r}")

    gts = list(gt_events)
    preds = list(pred_events)

    if strategy == "greedy":
        matched, used_g, used_p = _greedy_match(gts, preds, iou_threshold)
    elif strategy == "hungarian":
        matched, used_g, used_p = _hungarian_match(gts, preds, iou_threshold)
    else:
        raise ValueError(f"unknown strategy: {strategy!r}")

    unmatched_gt = [g for i, g in enumerate(gts) if i not in used_g]
    unmatched_pred = [p for j, p in enumerate(preds) if j not in used_p]
    return MatchingResult(
        matched=matched,
        unmatched_gt=unmatched_gt,
        unmatched_pred=unmatched_pred,
        cls=cls,
        iou_threshold=float(iou_threshold),
        strategy=str(strategy),
    )


def match_events_per_class(
    gt_events: Sequence[Event],
    pred_events: Sequence[Event],
    classes: Sequence[str],
    *,
    iou_threshold: float = 0.3,
    strategy: Strategy = "greedy",
) -> dict[str, MatchingResult]:
    """Group events by class then call :func:`match_events` for each class.

    Returned dictionary preserves the order of ``classes``.
    """
    out: dict[str, MatchingResult] = {}
    for cls in classes:
        g = [e for e in gt_events if e.cls == cls]
        p = [e for e in pred_events if e.cls == cls]
        out[cls] = match_events(
            g, p, iou_threshold=iou_threshold, strategy=strategy, cls=cls
        )
    return out
