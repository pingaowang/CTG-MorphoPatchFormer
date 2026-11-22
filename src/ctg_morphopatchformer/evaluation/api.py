"""Validated interfaces around the archived formal metric implementation."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from ..pipelines import validate_mask
from .cluster_inference import (
    DEFAULT_METRICS, FEATURE_COLUMNS, case_sufficient_statistics,
    metrics_from_sums, tensor_from_frame, seed_oof_metrics,
    case_bootstrap_absolute, case_bootstrap_paired, case_paired_permutation_pvalue,
    holm_adjust as _holm_adjust,
)


def _statistics(target, prediction, *, record_id, fs_hz):
    yt, yp = validate_mask(target), validate_mask(prediction)
    if yt.shape != yp.shape:
        raise ValueError("Target and prediction lengths differ.")
    if isinstance(fs_hz, (bool, np.bool_)) or not isinstance(fs_hz, (int, float, np.integer, np.floating)) or not np.isfinite(fs_hz) or fs_hz <= 0:
        raise ValueError("Supply the actual positive sampling rate; no rate is inferred.")
    return case_sufficient_statistics(yt, yp, case_id=str(record_id), fs_hz=float(fs_hz),
                                      iou_threshold=0.30, matching_strategy="greedy")


def holm_adjust(p_values):
    """Correct the entire predeclared comparison family, not selected small p values."""
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Provide a nonempty one-dimensional family of finite p values.")
    if (values < 0).any() or (values > 1).any():
        raise ValueError("P values must lie in [0,1].")
    return _holm_adjust(values)


def evaluate_record(target, prediction, *, fs_hz: float) -> dict[str, float]:
    """Score one record. Boundary errors use explicit fs_hz; no-GT FPs count."""
    stats = _statistics(target, prediction, record_id="input", fs_hz=fs_hz)
    values = metrics_from_sums(np.asarray([stats[k] for k in FEATURE_COLUMNS]))
    return {key: float(value) for key, value in values.items()}


def _resampling_arguments(resamples, random_seed):
    if isinstance(resamples, bool) or not isinstance(resamples, (int, np.integer)) or resamples < 1:
        raise ValueError("resamples must be a positive integer.")
    if isinstance(random_seed, bool) or not isinstance(random_seed, (int, np.integer)) or random_seed < 0:
        raise ValueError("random_seed must be a nonnegative integer.")


def summarize_runs(runs: Mapping, *, resamples: int = 10_000,
                   random_seed: int = 20260713) -> dict:
    """Input: seed -> record ID -> {target, prediction, fs_hz}.

    Supply one held-out complete prediction per record per seed. Folds are not
    independent replicates. Sampling units travel together across all seeds.
    This function checks coverage and reference consistency, not training origin.
    """
    _resampling_arguments(resamples, random_seed)
    if not isinstance(runs, Mapping) or not runs:
        raise ValueError("Provide at least one complete run.")
    rows, reference, rates = [], {}, {}
    for seed, records in runs.items():
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("Run keys must be nonnegative integer seeds.")
        if not isinstance(records, Mapping) or not records:
            raise ValueError("Each run must contain records.")
        seen = set()
        for record_id, record in records.items():
            name = str(record_id)
            if name in seen:
                raise ValueError("Record IDs collide after string conversion.")
            seen.add(name)
            yt = validate_mask(record["target"])
            rate = record["fs_hz"]
            if name in reference and (not np.array_equal(reference[name], yt) or rates[name] != rate):
                raise ValueError("The reference or sampling rate differs across seeds.")
            stats = _statistics(yt, record["prediction"], record_id=name, fs_hz=rate)
            reference[name], rates[name] = yt, rate
            rows.append({"seed": int(seed), "case_id": name, **stats})
    seeds, cases, tensor = tensor_from_frame(pd.DataFrame(rows))
    per_seed = seed_oof_metrics(seeds, tensor)
    intervals = case_bootstrap_absolute(tensor, n_resamples=resamples, seed=random_seed)
    estimates = {metric: float(per_seed[metric].mean()) for metric in DEFAULT_METRICS}
    sd = {metric: float(per_seed[metric].std(ddof=1)) if len(seeds) > 1 else None
          for metric in DEFAULT_METRICS}
    return {"estimates": estimates, "ci95": intervals, "seed_sd": sd,
            "per_seed": per_seed.to_dict("records"), "seeds": seeds,
            "record_ids": cases, "statistics_tensor": tensor,
            "resamples": resamples, "random_seed": random_seed,
            "fs_hz_by_record": rates,
            "references_by_record": reference}


def paired_comparison(candidate: dict, reference: dict, *,
                      resamples: int = 10_000, random_seed: int = 20260714,
                      deterministic_reference: bool = False) -> dict:
    """Paired delta CI and two-sided permutation p; Holm is a separate family step.

    Deterministic-reference broadcasting is explicit and only allowed when the
    reference contains one run. It creates no extra record sampling units.
    """
    _resampling_arguments(resamples, random_seed)
    if candidate["record_ids"] != reference["record_ids"]:
        raise ValueError("Paired comparisons require identical ordered record sets.")
    if candidate["fs_hz_by_record"] != reference["fs_hz_by_record"]:
        raise ValueError("Paired sampling rates differ.")
    for key in candidate["record_ids"]:
        if not np.array_equal(candidate["references_by_record"][key], reference["references_by_record"][key]):
            raise ValueError("Paired expert labels differ.")
    cand, ref = candidate["statistics_tensor"], reference["statistics_tensor"]
    if deterministic_reference:
        if len(reference["seeds"]) != 1:
            raise ValueError("A deterministic reference must contain exactly one run.")
        ref = np.repeat(ref, len(candidate["seeds"]), axis=0)
    elif candidate["seeds"] != reference["seeds"]:
        raise ValueError("Align identical seed sets explicitly before a paired comparison.")
    metric = "event_f1_macro"
    delta = float(np.nanmean(metrics_from_sums(cand.sum(1))[metric]
                            - metrics_from_sums(ref.sum(1))[metric]))
    return {"metric": metric, "delta": delta,
            "delta_ci95": case_bootstrap_paired(cand, ref, metrics=(metric,),
                n_resamples=resamples, seed=random_seed)[metric],
            "p_two_sided": case_paired_permutation_pvalue(cand, ref, metric=metric,
                n_resamples=resamples, seed=random_seed),
            "holm_adjusted": False, "resamples": resamples,
            "random_seed": random_seed, "deterministic_reference": deterministic_reference}
