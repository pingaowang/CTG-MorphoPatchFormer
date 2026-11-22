"""1D U-Net for CTGNet (encoder: Conv–BN–ReLU + MaxPool 1×2; decoder: upsample + conv)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECA1d(nn.Module):
    """Efficient Channel Attention (ECA-Net) for 1D features — near-zero extra params (M2b)."""

    def __init__(self, channels: int, k_size: int = 5) -> None:
        super().__init__()
        k = int(k_size)
        if k % 2 == 0:
            k += 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        y = x.mean(dim=-1, keepdim=True)  # [B, C, 1] global pooled per channel
        y = self.conv(y.transpose(1, 2)).transpose(1, 2)  # channel interaction
        return x * torch.sigmoid(y)


class MultiScaleBlock(nn.Module):
    """Parallel dilated-conv branches (d=1,2,4) fused by 1x1 — multi-receptive-field (M2a)."""

    def __init__(self, ci: int, co: int, kernel_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        k = int(kernel_size)
        dilations = (1, 2, 4)
        branch_c = co
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(ci, branch_c, kernel_size=k, padding=(k // 2) * d, dilation=d),
                    nn.BatchNorm1d(branch_c),
                    nn.ReLU(inplace=True),
                )
                for d in dilations
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(branch_c * len(dilations), co, kernel_size=1),
            nn.BatchNorm1d(co),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)) if dropout > 0.0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]
        return self.fuse(torch.cat(feats, dim=1))


class CtgNet1D(nn.Module):
    """Single-branch 1D U-Net matching the paper description (one conv per encoder stage).

    Optional mechanism flags (default off → identical to paper baseline):
    - ``multiscale``: replace bottleneck with a parallel-dilated MultiScaleBlock (M2a).
    - ``eca``: insert ECA channel attention after each decoder block (M2b).
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        num_classes: int = 3,
        base_channels: int = 32,
        kernel_size: int = 5,
        dropout: float = 0.0,
        multiscale: bool = False,
        eca: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.use_eca = bool(eca)
        k = int(kernel_size)
        pad = k // 2
        bc = int(base_channels)
        p = float(dropout)
        if p < 0.0 or p >= 1.0:
            raise ValueError("dropout must be in [0, 1).")
        drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()

        def enc_block(ci: int, co: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(ci, co, kernel_size=k, padding=pad),
                nn.BatchNorm1d(co),
                nn.ReLU(inplace=True),
                drop,
            )

        self.enc1 = enc_block(in_channels, bc)
        self.enc2 = enc_block(bc, bc * 2)
        self.enc3 = enc_block(bc * 2, bc * 4)
        self.enc4 = enc_block(bc * 4, bc * 8)
        if multiscale:
            self.bottleneck = MultiScaleBlock(bc * 8, bc * 16, kernel_size=k, dropout=p)
        else:
            self.bottleneck = enc_block(bc * 8, bc * 16)

        def dec_block(ci: int, co: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(ci, co, kernel_size=k, padding=pad),
                nn.BatchNorm1d(co),
                nn.ReLU(inplace=True),
                drop,
            )

        self.dec4 = dec_block(bc * 16 + bc * 8, bc * 8)
        self.dec3 = dec_block(bc * 8 + bc * 4, bc * 4)
        self.dec2 = dec_block(bc * 4 + bc * 2, bc * 2)
        self.dec1 = dec_block(bc * 2 + bc + in_channels, bc)
        if self.use_eca:
            self.eca4 = ECA1d(bc * 8)
            self.eca3 = ECA1d(bc * 4)
            self.eca2 = ECA1d(bc * 2)
            self.eca1 = ECA1d(bc)
        self.out_conv = nn.Conv1d(bc, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool1d(e1, 2))
        e3 = self.enc3(F.max_pool1d(e2, 2))
        e4 = self.enc4(F.max_pool1d(e3, 2))
        bot = self.bottleneck(F.max_pool1d(e4, 2))

        d = F.interpolate(bot, size=e4.shape[-1], mode="linear", align_corners=False)
        d = self.dec4(torch.cat([d, e4], dim=1))
        if self.use_eca:
            d = self.eca4(d)
        d = F.interpolate(d, size=e3.shape[-1], mode="linear", align_corners=False)
        d = self.dec3(torch.cat([d, e3], dim=1))
        if self.use_eca:
            d = self.eca3(d)
        d = F.interpolate(d, size=e2.shape[-1], mode="linear", align_corners=False)
        d = self.dec2(torch.cat([d, e2], dim=1))
        if self.use_eca:
            d = self.eca2(d)
        d = F.interpolate(d, size=e1.shape[-1], mode="linear", align_corners=False)
        d = self.dec1(torch.cat([d, e1, x], dim=1))
        if self.use_eca:
            d = self.eca1(d)

        logits = self.out_conv(d)
        if logits.shape[-1] != x.shape[-1]:
            logits = F.interpolate(logits, size=x.shape[-1], mode="linear", align_corners=False)
        return logits
