"""Windowed inference on already prepared [channel,time] tensors only."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from .pipelines import validate_probabilities


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def window_starts(length: int, window: int = 1200, stride: int = 300) -> list[int]:
    """Append a tail-aligned full window. Short records are rejected, not padded."""
    length = _positive_integer(length, "length")
    window = _positive_integer(window, "window")
    stride = _positive_integer(stride, "stride")
    if length < window or stride > window:
        raise ValueError("Require record length >= window and stride <= window.")
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] + window < length:
        starts.append(length - window)
    return starts


def average_window_probabilities(length: int, starts: Sequence[int],
                                 probabilities: Sequence[np.ndarray]) -> np.ndarray:
    """Average softmax probabilities, never logits or hard-voted labels."""
    length = _positive_integer(length, "length")
    if len(starts) != len(probabilities) or not len(starts):
        raise ValueError("Provide one probability array per nonempty window-start list.")
    if len(set(starts)) != len(starts):
        raise ValueError("Duplicate window starts are not allowed.")
    total = np.zeros((length, 3), dtype=np.float64)
    count = np.zeros((length, 1), dtype=np.int64)
    for start, values in zip(starts, probabilities):
        if isinstance(start, bool) or not isinstance(start, (int, np.integer)) or start < 0:
            raise ValueError("Window starts must be nonnegative integers.")
        p = validate_probabilities(values)
        end = start + len(p)
        if end > length:
            raise ValueError("A window extends beyond the record.")
        total[start:end] += p
        count[start:end] += 1
    if not (count > 0).all():
        raise ValueError("Windows do not cover the complete record.")
    return total / count


@torch.inference_mode()
def predict_record_probabilities(model: torch.nn.Module, prepared_input: torch.Tensor,
                                 *, window: int = 1200, stride: int = 300) -> np.ndarray:
    """Use the caller's device/dtype and eval-mode model; return [time,3].

    No cleaning, normalization, resampling, checkpoint loading or training occurs.
    No output-length interpolation is silently applied to an incompatible model.
    """
    if model.training:
        raise ValueError("Call model.eval() before inference.")
    if any(module.training for module in model.modules()):
        raise ValueError("All model submodules must be in evaluation mode.")
    if not isinstance(prepared_input, torch.Tensor) or prepared_input.ndim != 2:
        raise ValueError("Expected an already prepared tensor [channel,time].")
    if not prepared_input.is_floating_point() or not bool(torch.isfinite(prepared_input).all()):
        raise ValueError("Input must be a finite floating-point tensor.")
    length = prepared_input.shape[-1]
    starts = window_starts(length, window, stride)
    predictions = []
    for start in starts:
        output = model(prepared_input[:, start:start + window].unsqueeze(0))
        if not isinstance(output, torch.Tensor) or tuple(output.shape) != (1, 3, window):
            raise ValueError("Model must return logits [1,3,window]; hard votes are not logits.")
        if not bool(torch.isfinite(output).all()):
            raise ValueError("Model returned nonfinite logits.")
        predictions.append(output.float().softmax(dim=1)[0].cpu().numpy().T)
    return average_window_probabilities(length, starts, predictions)


@torch.inference_mode()
def predict_wavefhr_record(model, prepared_input: torch.Tensor, *, window: int = 1200,
                          stride: int = 300) -> np.ndarray:
    """Reconstruct each basis independently, then hard-vote full-record labels."""
    from .comparators.voting import WaveFHRHardVote1D, hard_vote_labels
    if not isinstance(model, WaveFHRHardVote1D) or model.training:
        raise ValueError("Use an eval-mode WaveFHRHardVote1D model.")
    labels = [torch.from_numpy(predict_record_probabilities(
        branch, prepared_input, window=window, stride=stride).argmax(axis=1))
        for branch in model.branches]
    return hard_vote_labels(labels).numpy()
