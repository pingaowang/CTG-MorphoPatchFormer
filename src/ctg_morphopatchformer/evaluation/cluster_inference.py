"""Dependence-aware inference from per-case OOF predictions.

The functions in this module treat the recording/case as the resampling
cluster. Predictions from different random seeds for the same case always
travel together. This prevents repeated evaluations of one test case from
being counted as independent clinical observations.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from .event_extraction import extract_events_from_mask
from .event_matching import match_events_per_class


EVENT_CLASSES = ("acceleration", "deceleration")
CLASS_LABELS = {"background": 0, "acceleration": 1, "deceleration": 2}

FEATURE_COLUMNS = tuple(
    [
        f"event_{field}_{short}"
        for short in ("acc", "dec")
        for field in (
            "gt",
            "pred",
            "tp",
            "fp",
            "fn",
            "match_n",
            "iou_sum",
            "start_abs_sum",
            "end_abs_sum",
            "duration_abs_sum",
        )
    ]
    + [
        f"frame_{field}_{short}"
        for short in ("bg", "acc", "dec")
        for field in ("tp", "fp", "fn")
    ]
    + ["n_frames", "n_correct"]
)

DEFAULT_METRICS = (
    "event_precision_acc",
    "event_recall_acc",
    "event_f1_acc",
    "event_precision_dec",
    "event_recall_dec",
    "event_f1_dec",
    "event_precision_macro",
    "event_recall_macro",
    "event_f1_macro",
    "pred_gt_event_ratio",
    "mean_iou_acc",
    "mean_iou_dec",
    "mean_iou_macro",
    "start_mae_macro_sec",
    "end_mae_macro_sec",
    "duration_mae_macro_sec",
    "boundary_mae_macro_sec",
    "timestep_f1_acc",
    "timestep_f1_dec",
    "timestep_f1_event_macro",
    "timestep_iou_acc",
    "timestep_iou_dec",
    "timestep_iou_event_macro",
    "overall_accuracy",
)


def stable_seed(base_seed: int, key: str) -> int:
    """Derive a deterministic 32-bit RNG seed from a base seed and key."""

    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return (int(base_seed) + int(digest[:8], 16)) % (2**32 - 1)


def case_sufficient_statistics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    case_id: str,
    fs_hz: float = 1.0,
    iou_threshold: float = 0.3,
    matching_strategy: str = "greedy",
) -> dict[str, float]:
    """Return additive event and frame statistics for one case."""

    yt = np.asarray(y_true).reshape(-1).astype(np.int64)
    yp = np.asarray(y_pred).reshape(-1).astype(np.int64)
    if yt.shape != yp.shape:
        raise ValueError(f"{case_id}: y_true {yt.shape} != y_pred {yp.shape}")
    if fs_hz <= 0:
        raise ValueError(f"{case_id}: fs_hz must be positive, got {fs_hz}")
    valid = {0, 1, 2}
    if not set(np.unique(yt)).issubset(valid) or not set(np.unique(yp)).issubset(valid):
        raise ValueError(f"{case_id}: labels must be a subset of {sorted(valid)}")

    gt_events = extract_events_from_mask(yt, case_id=case_id)
    pred_events = extract_events_from_mask(yp, case_id=case_id)
    matched = match_events_per_class(
        gt_events,
        pred_events,
        classes=EVENT_CLASSES,
        iou_threshold=iou_threshold,
        strategy=matching_strategy,
    )

    out: dict[str, float] = {}
    time_scale = 1.0 / float(fs_hz)
    for cls, short in (("acceleration", "acc"), ("deceleration", "dec")):
        result = matched[cls]
        cls_gt = [event for event in gt_events if event.cls == cls]
        cls_pred = [event for event in pred_events if event.cls == cls]
        out[f"event_gt_{short}"] = float(len(cls_gt))
        out[f"event_pred_{short}"] = float(len(cls_pred))
        out[f"event_tp_{short}"] = float(result.n_tp)
        out[f"event_fp_{short}"] = float(result.n_fp)
        out[f"event_fn_{short}"] = float(result.n_fn)
        out[f"event_match_n_{short}"] = float(len(result.matched))
        out[f"event_iou_sum_{short}"] = float(sum(pair.iou for pair in result.matched))
        out[f"event_start_abs_sum_{short}"] = float(
            sum(abs(pair.start_error) * time_scale for pair in result.matched)
        )
        out[f"event_end_abs_sum_{short}"] = float(
            sum(abs(pair.end_error) * time_scale for pair in result.matched)
        )
        out[f"event_duration_abs_sum_{short}"] = float(
            sum(abs(pair.duration_error) * time_scale for pair in result.matched)
        )

    for cls, short in (("background", "bg"), ("acceleration", "acc"), ("deceleration", "dec")):
        label = CLASS_LABELS[cls]
        gt_pos = yt == label
        pred_pos = yp == label
        out[f"frame_tp_{short}"] = float(np.logical_and(gt_pos, pred_pos).sum())
        out[f"frame_fp_{short}"] = float(np.logical_and(~gt_pos, pred_pos).sum())
        out[f"frame_fn_{short}"] = float(np.logical_and(gt_pos, ~pred_pos).sum())

    out["n_frames"] = float(yt.size)
    out["n_correct"] = float((yt == yp).sum())
    return out


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    num_arr = np.asarray(num, dtype=np.float64)
    den_arr = np.asarray(den, dtype=np.float64)
    return np.divide(num_arr, den_arr, out=np.zeros_like(num_arr), where=den_arr > 0)


def _nanmean_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    stacked = np.stack([a, b], axis=-1)
    finite = np.isfinite(stacked)
    count = finite.sum(axis=-1)
    total = np.where(finite, stacked, 0.0).sum(axis=-1)
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)


def metrics_from_sums(sums: np.ndarray) -> dict[str, np.ndarray]:
    """Convert additive sufficient statistics into paper metrics.

    ``sums`` may have any leading shape as long as its final dimension follows
    :data:`FEATURE_COLUMNS`.
    """

    arr = np.asarray(sums, dtype=np.float64)
    if arr.shape[-1] != len(FEATURE_COLUMNS):
        raise ValueError(f"expected last dimension {len(FEATURE_COLUMNS)}, got {arr.shape[-1]}")
    idx = {name: pos for pos, name in enumerate(FEATURE_COLUMNS)}
    get = lambda name: arr[..., idx[name]]

    out: dict[str, np.ndarray] = {}
    event_active: dict[str, np.ndarray] = {}
    for short in ("acc", "dec"):
        tp = get(f"event_tp_{short}")
        fp = get(f"event_fp_{short}")
        fn = get(f"event_fn_{short}")
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        out[f"event_precision_{short}"] = precision
        out[f"event_recall_{short}"] = recall
        out[f"event_f1_{short}"] = f1
        event_active[short] = (get(f"event_gt_{short}") + get(f"event_pred_{short}")) > 0

        matched = get(f"event_match_n_{short}")
        out[f"mean_iou_{short}"] = np.divide(
            get(f"event_iou_sum_{short}"),
            matched,
            out=np.full_like(matched, np.nan),
            where=matched > 0,
        )
        for field, metric_name in (
            ("start_abs_sum", "start_mae"),
            ("end_abs_sum", "end_mae"),
            ("duration_abs_sum", "duration_mae"),
        ):
            out[f"{metric_name}_{short}_sec"] = np.divide(
                get(f"event_{field}_{short}"),
                matched,
                out=np.full_like(matched, np.nan),
                where=matched > 0,
            )

    for base in ("event_precision", "event_recall", "event_f1"):
        acc = out[f"{base}_acc"]
        dec = out[f"{base}_dec"]
        acc_masked = np.where(event_active["acc"], acc, np.nan)
        dec_masked = np.where(event_active["dec"], dec, np.nan)
        out[f"{base}_macro"] = _nanmean_pair(acc_masked, dec_masked)

    out["pred_gt_event_ratio"] = _safe_div(
        get("event_pred_acc") + get("event_pred_dec"),
        get("event_gt_acc") + get("event_gt_dec"),
    )
    out["mean_iou_macro"] = _nanmean_pair(out["mean_iou_acc"], out["mean_iou_dec"])
    for metric in ("start_mae", "end_mae", "duration_mae"):
        out[f"{metric}_macro_sec"] = _nanmean_pair(
            out[f"{metric}_acc_sec"], out[f"{metric}_dec_sec"]
        )
    out["boundary_mae_macro_sec"] = _nanmean_pair(
        out["start_mae_macro_sec"], out["end_mae_macro_sec"]
    )

    for short in ("bg", "acc", "dec"):
        tp = get(f"frame_tp_{short}")
        fp = get(f"frame_fp_{short}")
        fn = get(f"frame_fn_{short}")
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        out[f"timestep_f1_{short}"] = _safe_div(2.0 * precision * recall, precision + recall)
        out[f"timestep_iou_{short}"] = _safe_div(tp, tp + fp + fn)
    out["timestep_f1_event_macro"] = (out["timestep_f1_acc"] + out["timestep_f1_dec"]) / 2.0
    out["timestep_iou_event_macro"] = (out["timestep_iou_acc"] + out["timestep_iou_dec"]) / 2.0
    out["overall_accuracy"] = _safe_div(get("n_correct"), get("n_frames"))
    return out


def tensor_from_frame(frame: pd.DataFrame) -> tuple[list[int], list[str], np.ndarray]:
    """Return a complete ``seed x case x feature`` tensor from a long frame."""

    required = {"seed", "case_id", *FEATURE_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if frame.duplicated(["seed", "case_id"]).any():
        dup = frame.loc[frame.duplicated(["seed", "case_id"], keep=False), ["seed", "case_id"]]
        raise ValueError(f"duplicate seed/case rows: {dup.head().to_dict('records')}")

    seeds = sorted(int(value) for value in frame["seed"].unique())
    case_sets = {
        seed: set(frame.loc[frame["seed"].astype(int).eq(seed), "case_id"].astype(str)) for seed in seeds
    }
    cases = sorted(case_sets[seeds[0]])
    for seed in seeds[1:]:
        if case_sets[seed] != set(cases):
            missing_cases = sorted(set(cases).difference(case_sets[seed]))[:5]
            extra_cases = sorted(case_sets[seed].difference(cases))[:5]
            raise ValueError(f"seed {seed} case set differs; missing={missing_cases}, extra={extra_cases}")

    indexed = frame.copy()
    indexed["seed"] = indexed["seed"].astype(int)
    indexed["case_id"] = indexed["case_id"].astype(str)
    indexed = indexed.set_index(["seed", "case_id"])
    order = pd.MultiIndex.from_product([seeds, cases], names=["seed", "case_id"])
    tensor = indexed.loc[order, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    tensor = tensor.reshape(len(seeds), len(cases), len(FEATURE_COLUMNS))
    return seeds, cases, tensor


def seed_oof_metrics(
    seeds: Sequence[int], tensor: np.ndarray, metrics: Iterable[str] = DEFAULT_METRICS
) -> pd.DataFrame:
    """Compute one pooled four-fold OOF metric row per seed."""

    metric_names = tuple(metrics)
    values = metrics_from_sums(np.asarray(tensor, dtype=np.float64).sum(axis=1))
    unknown = sorted(set(metric_names).difference(values))
    if unknown:
        raise ValueError(f"unknown metrics: {unknown}")
    rows = []
    for seed_idx, seed in enumerate(seeds):
        row: dict[str, float | int] = {"seed": int(seed)}
        for metric in metric_names:
            row[metric] = float(values[metric][seed_idx])
        rows.append(row)
    return pd.DataFrame(rows)


def case_bootstrap_absolute(
    tensor: np.ndarray,
    *,
    metrics: Iterable[str] = DEFAULT_METRICS,
    n_resamples: int = 10_000,
    seed: int = 20260713,
    batch_size: int = 250,
) -> dict[str, tuple[float, float]]:
    """Case-cluster percentile intervals, averaging pooled OOF metrics over seeds."""

    arr = np.asarray(tensor, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[-1] != len(FEATURE_COLUMNS):
        raise ValueError("tensor must have shape [seed, case, feature]")
    metric_names = tuple(metrics)
    n_cases = arr.shape[1]
    if n_cases < 2:
        return {metric: (float("nan"), float("nan")) for metric in metric_names}

    rng = np.random.default_rng(seed)
    draws = {metric: np.empty(n_resamples, dtype=np.float64) for metric in metric_names}
    probs = np.full(n_cases, 1.0 / n_cases, dtype=np.float64)
    offset = 0
    while offset < n_resamples:
        size = min(batch_size, n_resamples - offset)
        counts = rng.multinomial(n_cases, probs, size=size).astype(np.float64)
        pooled = np.einsum("bc,scf->bsf", counts, arr, optimize=True)
        values = metrics_from_sums(pooled)
        for metric in metric_names:
            draws[metric][offset : offset + size] = np.nanmean(values[metric], axis=1)
        offset += size
    return {
        metric: tuple(float(value) for value in np.nanquantile(draws[metric], [0.025, 0.975]))
        for metric in metric_names
    }


def case_bootstrap_paired(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    metrics: Iterable[str] = DEFAULT_METRICS,
    n_resamples: int = 10_000,
    seed: int = 20260713,
    batch_size: int = 250,
) -> dict[str, tuple[float, float]]:
    """Paired case-cluster percentile intervals for candidate-reference deltas."""

    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if cand.shape != ref.shape or cand.ndim != 3:
        raise ValueError(f"paired tensors must share [seed, case, feature] shape, got {cand.shape} and {ref.shape}")
    metric_names = tuple(metrics)
    n_cases = cand.shape[1]
    rng = np.random.default_rng(seed)
    probs = np.full(n_cases, 1.0 / n_cases, dtype=np.float64)
    draws = {metric: np.empty(n_resamples, dtype=np.float64) for metric in metric_names}
    offset = 0
    while offset < n_resamples:
        size = min(batch_size, n_resamples - offset)
        counts = rng.multinomial(n_cases, probs, size=size).astype(np.float64)
        cand_sum = np.einsum("bc,scf->bsf", counts, cand, optimize=True)
        ref_sum = np.einsum("bc,scf->bsf", counts, ref, optimize=True)
        cand_metrics = metrics_from_sums(cand_sum)
        ref_metrics = metrics_from_sums(ref_sum)
        for metric in metric_names:
            delta = cand_metrics[metric] - ref_metrics[metric]
            draws[metric][offset : offset + size] = np.nanmean(delta, axis=1)
        offset += size
    return {
        metric: tuple(float(value) for value in np.nanquantile(draws[metric], [0.025, 0.975]))
        for metric in metric_names
    }


def case_paired_permutation_pvalue(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    metric: str = "event_f1_macro",
    n_resamples: int = 10_000,
    seed: int = 20260714,
    batch_size: int = 250,
) -> float:
    """Two-sided paired permutation test with one swap decision per case cluster."""

    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if cand.shape != ref.shape or cand.ndim != 3:
        raise ValueError(f"paired tensors must share [seed, case, feature] shape, got {cand.shape} and {ref.shape}")
    total_cand = cand.sum(axis=1)
    total_ref = ref.sum(axis=1)
    observed = float(
        np.nanmean(metrics_from_sums(total_cand)[metric] - metrics_from_sums(total_ref)[metric])
    )
    case_shift = ref - cand
    rng = np.random.default_rng(seed)
    extreme = 0
    offset = 0
    tolerance = 1e-15
    while offset < n_resamples:
        size = min(batch_size, n_resamples - offset)
        swaps = rng.integers(0, 2, size=(size, cand.shape[1]), dtype=np.int8).astype(np.float64)
        shift = np.einsum("bc,scf->bsf", swaps, case_shift, optimize=True)
        perm_cand = total_cand[None, :, :] + shift
        perm_ref = total_ref[None, :, :] - shift
        delta = metrics_from_sums(perm_cand)[metric] - metrics_from_sums(perm_ref)[metric]
        permuted = np.nanmean(delta, axis=1)
        extreme += int(np.sum(np.abs(permuted) >= abs(observed) - tolerance))
        offset += size
    return float((extreme + 1) / (n_resamples + 1))


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down family-wise error correction."""

    p = np.asarray(p_values, dtype=np.float64)
    out = np.full_like(p, np.nan)
    finite_idx = np.flatnonzero(np.isfinite(p))
    if finite_idx.size == 0:
        return out
    order = finite_idx[np.argsort(p[finite_idx], kind="stable")]
    m = len(order)
    running = 0.0
    for rank, original_idx in enumerate(order):
        adjusted = min(1.0, (m - rank) * float(p[original_idx]))
        running = max(running, adjusted)
        out[original_idx] = running
    return out
