"""Dense per-frame mask -> structured event list.

An *event* is a maximally contiguous run of frames that carry the same non-zero
class label in a per-frame integer mask. Inputs and outputs use **half-open
intervals** ``[start, end)`` measured in frame index. This convention matches
how ``find_adjacent_ones_regions`` is already used in
``src/ml/metrics/event_f1.py`` (``end`` is exclusive), so events round-trip
losslessly back to a binary mask.

Example::

    mask = [0, 0, 1, 1, 1, 0, 0, 2, 2, 0]
    extract_events_from_mask(mask, case_id="case42")
    # -> [
    #     Event(case_id="case42", cls="acceleration", start=2, end=5),
    #     Event(case_id="case42", cls="deceleration", start=7, end=9),
    # ]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Event:
    """A single acceleration / deceleration event.

    Attributes:
        case_id: trace identifier; the same value as the file basename in the
            standard pipeline (e.g. ``"1001"``).
        cls: human-readable class name (``"acceleration"`` or
            ``"deceleration"``). Background is *never* represented as an event.
        start: inclusive start frame index (0-based).
        end: exclusive end frame index, so ``duration = end - start``.

    Notes:
        - ``start`` and ``end`` are typed as ``float`` to allow second-domain
          conversion (multiply by ``1.0 / fs_hz``). The default extractor
          returns integers as frames.
        - The dataclass is frozen so events can live in sets / be used as
          dictionary keys when needed by tests.
    """

    case_id: str
    cls: str
    start: float
    end: float
    extras: Mapping[str, float] = field(default_factory=dict, hash=False, compare=False)

    @property
    def duration(self) -> float:
        return float(self.end) - float(self.start)


_DEFAULT_CLASS_NAMES: tuple[str, str, str] = ("background", "acceleration", "deceleration")


def _runs_for_label(mask: np.ndarray, label: int) -> list[tuple[int, int]]:
    """Return maximally contiguous half-open intervals where ``mask == label``."""
    if mask.size == 0:
        return []
    bin_mask = (mask == label).astype(np.int8)
    diff = np.diff(np.concatenate(([0], bin_mask, [0])))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _filter_min_duration(
    runs: Sequence[tuple[int, int]], min_duration: int
) -> list[tuple[int, int]]:
    if min_duration <= 0:
        return list(runs)
    return [(s, e) for (s, e) in runs if (e - s) >= int(min_duration)]


def _merge_gaps(
    runs: Sequence[tuple[int, int]], merge_gap: int, mask: np.ndarray, background_label: int
) -> list[tuple[int, int]]:
    """Merge same-class runs separated by ``<= merge_gap`` background frames.

    The gap is defined as ``next_start - prev_end``; runs that touch
    (``next_start == prev_end``) are always merged. A negative or zero
    ``merge_gap`` only merges adjacent runs.
    """
    if not runs:
        return []
    runs_sorted = sorted(runs, key=lambda x: x[0])
    merged: list[tuple[int, int]] = [runs_sorted[0]]
    gap = max(0, int(merge_gap))
    for s, e in runs_sorted[1:]:
        ps, pe = merged[-1]
        if s - pe <= gap and np.all(mask[pe:s] == background_label):
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def extract_events_from_mask(
    mask: np.ndarray | Sequence[int],
    case_id: str,
    *,
    class_names: Sequence[str] = _DEFAULT_CLASS_NAMES,
    min_duration: int = 0,
    merge_gap: int = 0,
    background_label: int = 0,
) -> list[Event]:
    """Convert a 1-D per-frame label mask into a list of :class:`Event`.

    Args:
        mask: 1-D integer-valued array of length ``T`` with class labels.
        case_id: trace identifier copied into every produced :class:`Event`.
        class_names: index-aligned class names. ``class_names[0]`` must be the
            background class; only labels ``>= 1`` become events.
        min_duration: drop runs shorter than this many frames *after* merging.
            ``0`` keeps all runs. Use this on the prediction side to filter
            short false positives.
        merge_gap: merge two runs of the same class separated by at most this
            many background frames. Applied **before** ``min_duration`` so
            short genuine events that bracket a short background dip are not
            wrongly killed.
        background_label: label value to treat as background (default ``0``).

    Returns:
        A list of :class:`Event` ordered by ``start``, then by class.

    Raises:
        ValueError: if ``mask`` is not 1-D or has negative labels.

    The function is intentionally a pure transformation and never logs nor
    raises on empty input (an empty mask yields an empty list).
    """
    arr = np.asarray(mask)
    if arr.ndim != 1:
        raise ValueError(f"mask must be 1-D, got ndim={arr.ndim}")
    if arr.size == 0:
        return []
    arr = arr.astype(np.int64, copy=False)
    if (arr < 0).any():
        raise ValueError("mask contains negative labels")

    events: list[Event] = []
    for label, name in enumerate(class_names):
        if label == background_label:
            continue
        runs = _runs_for_label(arr, label)
        runs = _merge_gaps(runs, merge_gap, arr, background_label)
        runs = _filter_min_duration(runs, min_duration=min_duration)
        for s, e in runs:
            events.append(Event(case_id=case_id, cls=name, start=float(s), end=float(e)))
    events.sort(key=lambda ev: (ev.start, ev.cls))
    return events


def events_to_mask(
    events: Iterable[Event],
    length: int,
    *,
    class_names: Sequence[str] = _DEFAULT_CLASS_NAMES,
    background_label: int = 0,
) -> np.ndarray:
    """Rasterise an event list back to a per-frame mask.

    Mostly used by tests and visualisation; downstream models never need this.
    Overlapping events from different classes resolve in input order, so caller
    should pass events without inter-class overlap.
    """
    if length <= 0:
        return np.zeros((0,), dtype=np.int64)
    name_to_label = {name: i for i, name in enumerate(class_names)}
    out = np.full((int(length),), int(background_label), dtype=np.int64)
    for ev in events:
        if ev.cls not in name_to_label:
            raise ValueError(f"unknown class name: {ev.cls!r}")
        label = name_to_label[ev.cls]
        s = max(0, int(round(ev.start)))
        e = min(int(length), int(round(ev.end)))
        if e > s:
            out[s:e] = label
    return out
