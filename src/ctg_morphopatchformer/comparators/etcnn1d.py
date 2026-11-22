"""ETCNN-style 1D segmentation network for CTG acceleration/deceleration labels.

This is a CTGNet-compatible implementation inspired by the ETCNN paper:

- four TCNN branches with temporal kernels 21, 31, 61, and 81;
- C-Res residual convolution blocks with bounded channel calibration;
- a channel-wise cross-fusion transformer over multi-scale encoder features;
- OR-style ensemble fusion for event classes.

It is not author-code parity: the paper does not publish every engineering
detail needed for exact reproduction, so this module keeps the paper-level
mechanisms while preserving this repository's tensor contract
``[B, C_in, T] -> [B, num_classes, T]``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _odd_kernel(kernel_size: int) -> int:
    k = int(kernel_size)
    if k <= 0:
        raise ValueError("kernel_size must be positive.")
    if k % 2 == 0:
        k += 1
    return k


class BoundedReLU1(nn.Module):
    """ReLU clipped to [0, 1], matching the paper's ReLU1 notation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(F.relu(x), max=1.0)


class CNet1D(nn.Module):
    """Channel calibration network: GAP -> FC -> ReLU1."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        c = int(channels)
        hidden = max(1, c // max(1, int(reduction)))
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(c, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, c, kernel_size=1),
            BoundedReLU1(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CResBlock1D(nn.Module):
    """Residual temporal convolution block with C-Net feature calibration."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        k = _odd_kernel(kernel_size)
        pad = k // 2
        p = float(dropout)
        drop = nn.Dropout(p) if p > 0.0 else nn.Identity()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=k, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.cnet = CNet1D(out_channels)
        self.drop = drop
        self.shortcut = (
            nn.Identity()
            if int(in_channels) == int(out_channels)
            else nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = y * self.cnet(y)
        y = self.drop(y)
        y = self.bn2(self.conv2(y))
        y = F.relu(y + self.shortcut(x), inplace=True)
        return self.drop(y)


class ChannelCrossFusionTransformer1D(nn.Module):
    """Channel-wise cross-fusion over multi-scale encoder features.

    The paper describes CCT as channel-axis transformer fusion. Here each
    channel is treated as a token and a fixed temporal patch summary is used as
    the token embedding. The resulting channel gates modulate a fused multi-scale
    context at the decoder skip resolution.
    """

    def __init__(
        self,
        *,
        source_channels: Sequence[int],
        target_channels: int,
        patch_size: int = 48,
        n_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        heads = max(1, int(n_heads))
        if self.patch_size % heads != 0:
            heads = 1
        self.proj = nn.ModuleList(
            [nn.Conv1d(int(ch), int(target_channels), kernel_size=1, bias=False) for ch in source_channels]
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.patch_size,
            nhead=heads,
            dim_feedforward=max(self.patch_size * 2, 32),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, int(num_layers)))
        self.gate = nn.Sequential(nn.Linear(self.patch_size, 1), nn.Sigmoid())
        self.out = nn.Sequential(
            nn.Conv1d(int(target_channels), int(target_channels), kernel_size=1, bias=False),
            nn.BatchNorm1d(int(target_channels)),
        )

    def forward(self, target: torch.Tensor, sources: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(sources) != len(self.proj):
            raise ValueError(f"Expected {len(self.proj)} source features, got {len(sources)}.")
        target_len = int(target.shape[-1])
        projected = []
        for conv, src in zip(self.proj, sources):
            z = src
            if z.shape[-1] != target_len:
                z = F.interpolate(z, size=target_len, mode="linear", align_corners=False)
            projected.append(conv(z))
        context = torch.stack(projected, dim=0).mean(dim=0)
        tokens = F.adaptive_avg_pool1d(context, self.patch_size)
        tokens = self.encoder(tokens)
        gates = self.gate(tokens)
        return F.relu(target + self.out(context * gates), inplace=True)


class _SkipFusion1D(nn.Module):
    def __init__(
        self,
        source_channels: Sequence[int],
        target_channels: int,
        *,
        use_cct: bool,
        patch_size: int,
        n_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.use_cct = bool(use_cct)
        self.cct = (
            ChannelCrossFusionTransformer1D(
                source_channels=source_channels,
                target_channels=target_channels,
                patch_size=patch_size,
                n_heads=n_heads,
                num_layers=num_layers,
                dropout=dropout,
            )
            if self.use_cct
            else None
        )

    def forward(self, target: torch.Tensor, sources: Sequence[torch.Tensor]) -> torch.Tensor:
        if self.cct is None:
            return target
        return self.cct(target, sources)


class TCNN1D(nn.Module):
    """Single ETCNN TCNN branch."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        num_classes: int = 3,
        base_channels: int = 16,
        kernel_size: int = 31,
        dropout: float = 0.0,
        use_cct: bool = True,
        cct_patch_size: int = 48,
        cct_heads: int = 4,
        cct_layers: int = 1,
    ) -> None:
        super().__init__()
        bc = int(base_channels)
        if bc <= 0:
            raise ValueError("base_channels must be positive.")
        k = _odd_kernel(kernel_size)
        p = float(dropout)
        channels = [bc, bc * 2, bc * 4, bc * 8, bc * 16]

        self.enc1 = CResBlock1D(in_channels, channels[0], k, p)
        self.enc2 = CResBlock1D(channels[0], channels[1], k, p)
        self.enc3 = CResBlock1D(channels[1], channels[2], k, p)
        self.enc4 = CResBlock1D(channels[2], channels[3], k, p)
        self.enc5 = CResBlock1D(channels[3], channels[4], k, p)
        self.bottleneck = CResBlock1D(channels[4], channels[4] * 2, k, p)

        self.up5 = nn.Conv1d(channels[4] * 2, channels[4], kernel_size=1, bias=False)
        self.up4 = nn.Conv1d(channels[4], channels[3], kernel_size=1, bias=False)
        self.up3 = nn.Conv1d(channels[3], channels[2], kernel_size=1, bias=False)
        self.up2 = nn.Conv1d(channels[2], channels[1], kernel_size=1, bias=False)
        self.up1 = nn.Conv1d(channels[1], channels[0], kernel_size=1, bias=False)

        cct_kw = dict(
            source_channels=channels,
            use_cct=use_cct,
            patch_size=int(cct_patch_size),
            n_heads=int(cct_heads),
            num_layers=int(cct_layers),
            dropout=p,
        )
        self.fuse5 = _SkipFusion1D(target_channels=channels[4], **cct_kw)
        self.fuse4 = _SkipFusion1D(target_channels=channels[3], **cct_kw)
        self.fuse3 = _SkipFusion1D(target_channels=channels[2], **cct_kw)
        self.fuse2 = _SkipFusion1D(target_channels=channels[1], **cct_kw)
        self.fuse1 = _SkipFusion1D(target_channels=channels[0], **cct_kw)

        self.dec5 = CResBlock1D(channels[4] * 2, channels[4], k, p)
        self.dec4 = CResBlock1D(channels[3] * 2, channels[3], k, p)
        self.dec3 = CResBlock1D(channels[2] * 2, channels[2], k, p)
        self.dec2 = CResBlock1D(channels[1] * 2, channels[1], k, p)
        self.dec1 = CResBlock1D(channels[0] * 2, channels[0], k, p)
        self.out_conv = nn.Conv1d(channels[0], int(num_classes), kernel_size=1)

    @staticmethod
    def _pool(x: torch.Tensor) -> torch.Tensor:
        return F.max_pool1d(x, kernel_size=2, stride=2, ceil_mode=True)

    @staticmethod
    def _up(x: torch.Tensor, target_len: int) -> torch.Tensor:
        return F.interpolate(x, size=int(target_len), mode="linear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self._pool(e1))
        e3 = self.enc3(self._pool(e2))
        e4 = self.enc4(self._pool(e3))
        e5 = self.enc5(self._pool(e4))
        b = self.bottleneck(self._pool(e5))
        sources = (e1, e2, e3, e4, e5)

        d = self.up5(self._up(b, e5.shape[-1]))
        d = self.dec5(torch.cat([d, self.fuse5(e5, sources)], dim=1))
        d = self.up4(self._up(d, e4.shape[-1]))
        d = self.dec4(torch.cat([d, self.fuse4(e4, sources)], dim=1))
        d = self.up3(self._up(d, e3.shape[-1]))
        d = self.dec3(torch.cat([d, self.fuse3(e3, sources)], dim=1))
        d = self.up2(self._up(d, e2.shape[-1]))
        d = self.dec2(torch.cat([d, self.fuse2(e2, sources)], dim=1))
        d = self.up1(self._up(d, e1.shape[-1]))
        d = self.dec1(torch.cat([d, self.fuse1(e1, sources)], dim=1))

        logits = self.out_conv(d)
        if logits.shape[-1] != x.shape[-1]:
            logits = F.interpolate(logits, size=x.shape[-1], mode="linear", align_corners=False)
        return logits


class ETCNN1D(nn.Module):
    """Four-kernel TCNN ensemble with ETCNN-style event OR fusion."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        num_classes: int = 3,
        base_channels: int = 16,
        kernel_sizes: Iterable[int] = (21, 31, 61, 81),
        dropout: float = 0.0,
        use_cct: bool = True,
        cct_patch_size: int = 48,
        cct_heads: int = 4,
        cct_layers: int = 1,
        fusion: str = "or_logits",
    ) -> None:
        super().__init__()
        ks = tuple(_odd_kernel(int(k)) for k in kernel_sizes)
        if not ks:
            raise ValueError("ETCNN1D requires at least one kernel size.")
        self.num_classes = int(num_classes)
        self.fusion = str(fusion).strip().lower()
        self.branches = nn.ModuleList(
            [
                TCNN1D(
                    in_channels=in_channels,
                    num_classes=num_classes,
                    base_channels=base_channels,
                    kernel_size=k,
                    dropout=dropout,
                    use_cct=use_cct,
                    cct_patch_size=cct_patch_size,
                    cct_heads=cct_heads,
                    cct_layers=cct_layers,
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
        raise ValueError(f"Unknown ETCNN1D fusion={self.fusion!r}.")
