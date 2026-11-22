"""Hard voting, kept separate from calibrated class probabilities."""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .wavefhr_vnet1d import WaveFHRNet1D, hard_vote_from_logits

BASES = ("db4", "db6", "sym4", "sym5", "bior3.5")


def hard_vote_labels(labels_by_basis: Sequence[torch.Tensor]) -> torch.Tensor:
    """Vote over same-shaped integer labels; ties select the lowest class ID."""
    if not labels_by_basis:
        raise ValueError("At least one basis prediction is required.")
    stacked = torch.stack(list(labels_by_basis))
    if stacked.dtype.is_floating_point or stacked.dtype == torch.bool:
        raise ValueError("Votes must be integer class labels.")
    if bool(((stacked < 0) | (stacked > 2)).any()):
        raise ValueError("Class IDs must be 0, 1 or 2.")
    return torch.mode(stacked, dim=0).values


class WaveFHRHardVote1D(nn.Module):
    """Five single-basis networks with a hard-vote window inspection interface.

    This forward returns labels [B,T], not probabilities or three-class logits.
    For full records, average probabilities separately for each basis before
    hard voting; use inference.predict_wavefhr_record, not probability averaging
    over this wrapper's voted windows. No weights or training logic are supplied.
    """
    def __init__(self, *, bases=BASES, **branch_parameters):
        super().__init__()
        self.bases = tuple(bases)
        if not self.bases or len(set(self.bases)) != len(self.bases):
            raise ValueError("Wavelet bases must be nonempty and unique.")
        self.branches = nn.ModuleList(
            WaveFHRNet1D(basis=basis, **branch_parameters) for basis in self.bases)

    def forward(self, x):
        return hard_vote_from_logits([branch(x) for branch in self.branches])
