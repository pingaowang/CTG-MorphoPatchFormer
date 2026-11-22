"""Paper-aligned 1D LCResU-Net / ELCResU-Net.

Liu et al. (ESWA 2023) use six encoder levels with channels
8/16/32/64/128/256, a mirrored decoder, LC-Residual blocks at every level
except the first and last, and four independently trained kernel-size arms.

For 1 Hz inputs the paper's 4 Hz kernels [21, 31, 61, 81] are represented by
[5, 7, 15, 21], preserving their approximate duration in seconds.  The
ELCResUNet1D wrapper remains for backward compatibility; paper-faithful runs
must train LCResUNet1D arms independently and apply ``elcres_or_vote_from_logits``
only at inference time.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


PAPER_KERNELS_4HZ = (21, 31, 61, 81)
PAPER_KERNELS_1HZ = (5, 7, 15, 21)


def _odd_kernel(kernel_size: int) -> int:
    k = int(kernel_size)
    if k <= 0 or k % 2 == 0:
        raise ValueError("LCResU-Net kernel_size must be a positive odd integer.")
    return k


class _Relu1(nn.Module):
    """Paper ReLU-1 activation: clip the positive response to [0, 1]."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=0.0, max=1.0)


class LCResBlock1d(nn.Module):
    """LC-Residual block following Fig. 3 and Eqs. (1)-(6).

    C-Net produces a channel-wise scale for the first convolutional layer.
    L-Net produces a block/layer scale for the two-convolution backbone output.
    The third shortcut is the residual projection from input to output.
    """

    def __init__(self, ci: int, co: int, kernel_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        k = _odd_kernel(kernel_size)
        pad = k // 2
        self.conv1 = nn.Conv1d(ci, co, kernel_size=k, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(co)
        self.conv2 = nn.Conv1d(co, co, kernel_size=k, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(co)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.c_net = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(1), nn.Linear(ci, co), _Relu1())
        self.l_net = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(1), nn.Linear(ci, 1), _Relu1())
        self.residual = nn.Identity() if ci == co else nn.Conv1d(ci, co, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_scale = self.c_net(x).unsqueeze(-1)
        layer_scale = self.l_net(x).unsqueeze(-1)
        y = F.relu(self.bn1(self.conv1(x)), inplace=False)
        y = y * channel_scale
        y = F.relu(self.bn2(self.conv2(y)), inplace=False)
        y = self.dropout(y) * layer_scale
        return F.relu(y + self.residual(x), inplace=False)


class _UpBlock1d(nn.Module):
    def __init__(self, ci: int, co: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.up = nn.ConvTranspose1d(ci, co, kernel_size=2, stride=2)
        self.block = LCResBlock1d(co * 2, co, kernel_size, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class LCResUNet1D(nn.Module):
    """One independently trainable LCResU-Net arm."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        num_classes: int = 3,
        base_channels: int = 8,
        kernel_size: int = 7,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        bc = int(base_channels)
        k = _odd_kernel(kernel_size)
        p = float(dropout)
        # Paper Fig. 2: first level is Conv1+BN+ReLU, followed by five
        # LC-Residual encoder levels (six resolutions in total).
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, bc, kernel_size=k, padding=k // 2, bias=False),
            nn.BatchNorm1d(bc),
            nn.ReLU(inplace=False),
        )
        self.enc2 = LCResBlock1d(bc, bc * 2, k, p)
        self.enc3 = LCResBlock1d(bc * 2, bc * 4, k, p)
        self.enc4 = LCResBlock1d(bc * 4, bc * 8, k, p)
        self.enc5 = LCResBlock1d(bc * 8, bc * 16, k, p)
        self.enc6 = LCResBlock1d(bc * 16, bc * 32, k, p)
        self.up5 = _UpBlock1d(bc * 32, bc * 16, k, p)
        self.up4 = _UpBlock1d(bc * 16, bc * 8, k, p)
        self.up3 = _UpBlock1d(bc * 8, bc * 4, k, p)
        self.up2 = _UpBlock1d(bc * 4, bc * 2, k, p)
        self.up1 = _UpBlock1d(bc * 2, bc, k, p)
        self.out_conv = nn.Conv1d(bc, int(num_classes), kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.stem(x)
        e2 = self.enc2(F.max_pool1d(e1, 2))
        e3 = self.enc3(F.max_pool1d(e2, 2))
        e4 = self.enc4(F.max_pool1d(e3, 2))
        e5 = self.enc5(F.max_pool1d(e4, 2))
        e6 = self.enc6(F.max_pool1d(e5, 2))
        d = self.up5(e6, e5)
        d = self.up4(d, e4)
        d = self.up3(d, e3)
        d = self.up2(d, e2)
        d = self.up1(d, e1)
        return self.out_conv(d)


class ELCResUNet1D(nn.Module):
    """Compatibility wrapper; strict paper runs train the arms separately."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        num_classes: int = 3,
        base_channels: int = 8,
        kernel_sizes: Iterable[int] = PAPER_KERNELS_1HZ,
        dropout: float = 0.0,
        fusion: str = "or_logits",
    ) -> None:
        super().__init__()
        self.fusion = str(fusion).strip().lower()
        self.branches = nn.ModuleList(
            LCResUNet1D(
                in_channels=in_channels,
                num_classes=num_classes,
                base_channels=base_channels,
                kernel_size=int(k),
                dropout=dropout,
            )
            for k in kernel_sizes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.stack([branch(x) for branch in self.branches], dim=0)
        if self.fusion in ("mean", "mean_logits", "avg", "avg_logits"):
            return logits.mean(dim=0)
        if self.fusion in ("max", "max_logits"):
            return logits.max(dim=0).values
        if self.fusion in ("or", "or_logits"):
            bg = logits[:, :, 0:1, :].min(dim=0).values
            events = logits[:, :, 1:, :].max(dim=0).values
            return torch.cat([bg, events], dim=1)
        raise ValueError(f"Unknown ELCResUNet1D fusion={self.fusion!r}.")


def elcres_or_vote_from_logits(logits_by_kernel: Iterable[torch.Tensor]) -> torch.Tensor:
    """Hard OR of independently trained arm predictions.

    Any arm's event prediction overrides background.  If acceleration and
    deceleration are both proposed, the event class with the strongest softmax
    probability across arms wins.
    """
    logits = torch.stack(list(logits_by_kernel), dim=0)
    branch_pred = torch.argmax(logits, dim=2)
    event_present = (branch_pred > 0).any(dim=0)
    event_prob = torch.softmax(logits, dim=2)[:, :, 1:, :].max(dim=0).values
    best_event = torch.argmax(event_prob, dim=1) + 1
    return torch.where(event_present, best_event, torch.zeros_like(best_event))
