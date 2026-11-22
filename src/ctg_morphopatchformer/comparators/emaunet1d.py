"""1D EMAU-Net components from Liu et al. (Information Sciences, 2023).

In the paper, EMAU-Net means *Ensemble MultiAttention U-Net*: four MAU-Net
branches with different temporal convolution kernels, fused by an OR rule.
This module implements a CTGNet-compatible 1D adaptation for FHR segmentation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECAAdaptive1d(nn.Module):
    """Efficient Channel Attention with paper's adaptive kernel rule."""

    def __init__(self, channels: int, gamma: int = 2, b: int = 1) -> None:
        super().__init__()
        c = max(1, int(channels))
        k = int(abs((math.log2(c) + int(b)) / int(gamma)))
        k = max(1, k)
        if k % 2 == 0:
            k += 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.mean(dim=-1, keepdim=True)
        y = self.conv(y.transpose(1, 2)).transpose(1, 2)
        return x * torch.sigmoid(y)


class NonLocalAttention1d(nn.Module):
    """Embedded-Gaussian non-local attention over the temporal axis."""

    def __init__(self, channels: int, reduction: int = 2) -> None:
        super().__init__()
        c = int(channels)
        inter = max(1, c // max(1, int(reduction)))
        self.theta = nn.Conv1d(c, inter, kernel_size=1, bias=False)
        self.phi = nn.Conv1d(c, inter, kernel_size=1, bias=False)
        self.g = nn.Conv1d(c, inter, kernel_size=1, bias=False)
        self.out = nn.Conv1d(inter, c, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(c)
        nn.init.zeros_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, t = x.shape
        theta = self.theta(x).transpose(1, 2)  # [B, T, C']
        phi = self.phi(x)  # [B, C', T]
        attn = torch.bmm(theta, phi) / math.sqrt(max(1, phi.shape[1]))
        attn = torch.softmax(attn, dim=-1)
        g = self.g(x).transpose(1, 2)  # [B, T, C']
        y = torch.bmm(attn, g).transpose(1, 2).contiguous()
        y = self.bn(self.out(y))
        if y.shape[-1] != t:
            y = F.interpolate(y, size=t, mode="linear", align_corners=False)
        return x + y


class _DoubleConv1d(nn.Module):
    def __init__(self, ci: int, co: int, kernel_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        k = int(kernel_size)
        if k % 2 == 0:
            raise ValueError("MAU-Net convolution kernel_size must be odd.")
        pad = k // 2
        drop = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        self.net = nn.Sequential(
            nn.Conv1d(ci, co, kernel_size=k, padding=pad, bias=False),
            nn.BatchNorm1d(co),
            nn.ReLU(inplace=True),
            drop,
            nn.Conv1d(co, co, kernel_size=k, padding=pad, bias=False),
            nn.BatchNorm1d(co),
            nn.ReLU(inplace=True),
            drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _UpProject1d(nn.Module):
    """Paper-style upsample + 1x2 convolution that halves channels."""

    def __init__(self, ci: int, co: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(ci, co, kernel_size=2, padding=0, bias=False)

    def forward(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        # Upsample one sample longer so the valid 1x2 convolution returns target_len.
        z = F.interpolate(x, size=int(target_len) + 1, mode="linear", align_corners=False)
        z = self.conv(z)
        if z.shape[-1] != int(target_len):
            z = F.interpolate(z, size=int(target_len), mode="linear", align_corners=False)
        return z


class MauNet1D(nn.Module):
    """Single MAU-Net branch with ECA and non-local attention."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        num_classes: int = 3,
        base_channels: int = 32,
        kernel_size: int = 31,
        dropout: float = 0.0,
        use_eca: bool = True,
        use_nla: bool = True,
        nla_reduction: int = 2,
    ) -> None:
        super().__init__()
        bc = int(base_channels)
        p = float(dropout)
        if p < 0.0 or p >= 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self.num_classes = int(num_classes)
        self.enc1 = _DoubleConv1d(in_channels, bc, kernel_size, p)
        self.enc2 = _DoubleConv1d(bc, bc * 2, kernel_size, p)
        self.enc3 = _DoubleConv1d(bc * 2, bc * 4, kernel_size, p)
        self.enc4 = _DoubleConv1d(bc * 4, bc * 8, kernel_size, p)
        self.bottleneck = _DoubleConv1d(bc * 8, bc * 16, kernel_size, p)

        if use_eca:
            self.eca1 = ECAAdaptive1d(bc)
            self.eca2 = ECAAdaptive1d(bc * 2)
            self.eca3 = ECAAdaptive1d(bc * 4)
            self.eca4 = ECAAdaptive1d(bc * 8)
        else:
            self.eca1 = self.eca2 = self.eca3 = self.eca4 = nn.Identity()
        self.nla = NonLocalAttention1d(bc * 16, reduction=nla_reduction) if use_nla else nn.Identity()

        self.up4 = _UpProject1d(bc * 16, bc * 8)
        self.dec4 = _DoubleConv1d(bc * 16, bc * 8, kernel_size, p)
        self.up3 = _UpProject1d(bc * 8, bc * 4)
        self.dec3 = _DoubleConv1d(bc * 8, bc * 4, kernel_size, p)
        self.up2 = _UpProject1d(bc * 4, bc * 2)
        self.dec2 = _DoubleConv1d(bc * 4, bc * 2, kernel_size, p)
        self.up1 = _UpProject1d(bc * 2, bc)
        self.dec1 = _DoubleConv1d(bc * 2, bc, kernel_size, p)
        self.out_conv = nn.Conv1d(bc, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.eca1(self.enc1(x))
        e2 = self.eca2(self.enc2(F.max_pool1d(e1, 2)))
        e3 = self.eca3(self.enc3(F.max_pool1d(e2, 2)))
        e4 = self.eca4(self.enc4(F.max_pool1d(e3, 2)))
        bot = self.nla(self.bottleneck(F.max_pool1d(e4, 2)))

        d = self.up4(bot, e4.shape[-1])
        d = self.dec4(torch.cat([d, e4], dim=1))
        d = self.up3(d, e3.shape[-1])
        d = self.dec3(torch.cat([d, e3], dim=1))
        d = self.up2(d, e2.shape[-1])
        d = self.dec2(torch.cat([d, e2], dim=1))
        d = self.up1(d, e1.shape[-1])
        d = self.dec1(torch.cat([d, e1], dim=1))

        logits = self.out_conv(d)
        if logits.shape[-1] != x.shape[-1]:
            logits = F.interpolate(logits, size=x.shape[-1], mode="linear", align_corners=False)
        return logits


class EmaUNet1D(nn.Module):
    """Four-kernel EMAU-Net ensemble with differentiable OR-style fusion."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        num_classes: int = 3,
        base_channels: int = 32,
        kernel_sizes: Iterable[int] = (21, 31, 61, 81),
        dropout: float = 0.0,
        use_eca: bool = True,
        use_nla: bool = True,
        nla_reduction: int = 2,
        fusion: str = "or_logits",
    ) -> None:
        super().__init__()
        ks = tuple(int(k) for k in kernel_sizes)
        if not ks:
            raise ValueError("EmaUNet1D requires at least one kernel size.")
        self.num_classes = int(num_classes)
        self.fusion = str(fusion).strip().lower()
        self.branches = nn.ModuleList(
            [
                MauNet1D(
                    in_channels=in_channels,
                    num_classes=num_classes,
                    base_channels=base_channels,
                    kernel_size=k,
                    dropout=dropout,
                    use_eca=use_eca,
                    use_nla=use_nla,
                    nla_reduction=nla_reduction,
                )
                for k in ks
            ]
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
        raise ValueError(f"Unknown EmaUNet1D fusion={self.fusion!r}.")
