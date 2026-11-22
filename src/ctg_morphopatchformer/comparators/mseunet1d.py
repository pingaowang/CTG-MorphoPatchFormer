"""1D MSEU-Net (Lu et al., IJCNN 2025; Li & Lu, ACM CF 2024).

Faithful method identity for this repository:
- Encoder-decoder U-Net with 1x2 max-pooling downsampling and factor-2 upsampling.
- Convolutional block: Conv1d -> BatchNorm -> Dropout -> LeakyReLU (paper Fig. 3b).
- Multi-Scale feature Extraction (MSE) module: parallel convolutions with several
  kernel sizes, each followed by BatchNorm, concatenated into a comprehensive
  feature map (paper Fig. 3c). Paper kernels 21/33/65 (best ablation 31/61/81).
- Skip connections between encoder and decoder.

Adaptation: direct 3-class segmentation (background / acceleration / deceleration)
on prepared windows, matching the project CV4 + paper_metrics evaluation. The paper's
long-/short-period median-filter baseline post-processing is not part of this
architecture-only adapter. Multi-modal maternal
covariates (age/parity/pregnancy status) are not used: the project pipeline is
FHR-only across all three datasets. The factory selects the paper-aligned
concat_only profile; see comparators/registry.py for explicit parameters.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _odd_kernel(kernel_size: int) -> int:
    k = int(kernel_size)
    if k <= 0 or k % 2 == 0:
        raise ValueError("MSEU-Net kernel_size must be a positive odd integer.")
    return k


class _ConvBlock1d(nn.Module):
    """Paper Fig. 3(b): Conv1d -> BN -> Dropout -> LeakyReLU."""

    def __init__(
        self,
        ci: int,
        co: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
        leaky_slope: float = 0.01,
    ) -> None:
        super().__init__()
        k = _odd_kernel(kernel_size)
        p = float(dropout)
        if p < 0.0 or p >= 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self.net = nn.Sequential(
            nn.Conv1d(ci, co, kernel_size=k, padding=k // 2, bias=False),
            nn.BatchNorm1d(co),
            nn.Dropout(p) if p > 0.0 else nn.Identity(),
            nn.LeakyReLU(negative_slope=float(leaky_slope), inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MSEModule1d(nn.Module):
    """Paper Fig. 3(c): multi-scale parallel convs + BN, concatenated.

    ``projected`` reproduces the study implementation: every branch maps
    ``ci -> co`` and concatenated features are projected back to ``co``.
    ``concat_only`` partitions ``co`` channels across branches so concatenation
    already has width ``co`` and adds no unreported post-concatenation layer.
    """

    def __init__(
        self,
        ci: int,
        co: int,
        kernel_sizes: Iterable[int] = (7, 15, 21),
        leaky_slope: float = 0.01,
        fusion: str = "projected",
    ) -> None:
        super().__init__()
        ks = tuple(_odd_kernel(k) for k in kernel_sizes)
        if not ks:
            raise ValueError("MSEModule1d requires at least one kernel size.")
        fusion_key = str(fusion).strip().lower().replace("-", "_")
        if fusion_key not in {"projected", "concat_only"}:
            raise ValueError("fusion must be 'projected' or 'concat_only'.")

        if fusion_key == "projected":
            branch_channels = [int(co)] * len(ks)
        else:
            if int(co) < len(ks):
                raise ValueError("concat_only fusion requires co >= number of branches.")
            q, r = divmod(int(co), len(ks))
            branch_channels = [q + (1 if i < r else 0) for i in range(len(ks))]

        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(ci, branch_co, kernel_size=k, padding=k // 2, bias=False),
                    nn.BatchNorm1d(branch_co),
                )
                for k, branch_co in zip(ks, branch_channels, strict=True)
            ]
        )
        self.fuse = (
            nn.Sequential(
                nn.Conv1d(co * len(ks), co, kernel_size=1, bias=False),
                nn.BatchNorm1d(co),
                nn.LeakyReLU(negative_slope=float(leaky_slope), inplace=False),
            )
            if fusion_key == "projected"
            else nn.Identity()
        )
        self.fusion = fusion_key

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [branch(x) for branch in self.branches]
        return self.fuse(torch.cat(feats, dim=1))


class _EncoderStage1d(nn.Module):
    """Conv block followed by the multi-scale extraction module (paper Fig. 3a)."""

    def __init__(
        self,
        ci: int,
        co: int,
        kernel_sizes: Iterable[int],
        base_kernel_size: int = 3,
        dropout: float = 0.0,
        leaky_slope: float = 0.01,
        mse_fusion: str = "projected",
    ) -> None:
        super().__init__()
        self.conv = _ConvBlock1d(
            ci,
            co,
            kernel_size=base_kernel_size,
            dropout=dropout,
            leaky_slope=leaky_slope,
        )
        self.mse = MSEModule1d(
            co,
            co,
            kernel_sizes=kernel_sizes,
            leaky_slope=leaky_slope,
            fusion=mse_fusion,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mse(self.conv(x))


class MSEUNet1D(nn.Module):
    """Multi-scale encoder-decoder for 1D FHR segmentation."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        num_classes: int = 3,
        base_channels: int = 32,
        kernel_sizes: Iterable[int] = (7, 15, 21),
        base_kernel_size: int = 3,
        mse_fusion: str = "projected",
        dropout: float = 0.0,
        leaky_slope: float = 0.01,
    ) -> None:
        super().__init__()
        bc = int(base_channels)
        ks = tuple(_odd_kernel(k) for k in kernel_sizes)
        base_k = _odd_kernel(base_kernel_size)

        stage_kw = dict(
            kernel_sizes=ks,
            base_kernel_size=base_k,
            dropout=dropout,
            leaky_slope=leaky_slope,
            mse_fusion=mse_fusion,
        )
        self.enc1 = _EncoderStage1d(in_channels, bc, **stage_kw)
        self.enc2 = _EncoderStage1d(bc, bc * 2, **stage_kw)
        self.enc3 = _EncoderStage1d(bc * 2, bc * 4, **stage_kw)
        self.enc4 = _EncoderStage1d(bc * 4, bc * 8, **stage_kw)
        self.bottleneck = _EncoderStage1d(bc * 8, bc * 16, **stage_kw)

        self.up4 = nn.ConvTranspose1d(bc * 16, bc * 8, kernel_size=2, stride=2, bias=False)
        self.dec4 = _ConvBlock1d(bc * 16, bc * 8, kernel_size=base_k, dropout=dropout, leaky_slope=leaky_slope)
        self.up3 = nn.ConvTranspose1d(bc * 8, bc * 4, kernel_size=2, stride=2, bias=False)
        self.dec3 = _ConvBlock1d(bc * 8, bc * 4, kernel_size=base_k, dropout=dropout, leaky_slope=leaky_slope)
        self.up2 = nn.ConvTranspose1d(bc * 4, bc * 2, kernel_size=2, stride=2, bias=False)
        self.dec2 = _ConvBlock1d(bc * 4, bc * 2, kernel_size=base_k, dropout=dropout, leaky_slope=leaky_slope)
        self.up1 = nn.ConvTranspose1d(bc * 2, bc, kernel_size=2, stride=2, bias=False)
        self.dec1 = _ConvBlock1d(bc * 2, bc, kernel_size=base_k, dropout=dropout, leaky_slope=leaky_slope)
        self.out_conv = nn.Conv1d(bc, int(num_classes), kernel_size=1)

    @staticmethod
    def _up(layer: nn.Module, x: torch.Tensor, target_len: int) -> torch.Tensor:
        y = layer(x)
        if y.shape[-1] != int(target_len):
            y = F.interpolate(y, size=int(target_len), mode="linear", align_corners=False)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool1d(e1, 2))
        e3 = self.enc3(F.max_pool1d(e2, 2))
        e4 = self.enc4(F.max_pool1d(e3, 2))
        bot = self.bottleneck(F.max_pool1d(e4, 2))

        d = self.dec4(torch.cat([self._up(self.up4, bot, e4.shape[-1]), e4], dim=1))
        d = self.dec3(torch.cat([self._up(self.up3, d, e3.shape[-1]), e3], dim=1))
        d = self.dec2(torch.cat([self._up(self.up2, d, e2.shape[-1]), e2], dim=1))
        d = self.dec1(torch.cat([self._up(self.up1, d, e1.shape[-1]), e1], dim=1))
        logits = self.out_conv(d)
        if logits.shape[-1] != x.shape[-1]:
            logits = F.interpolate(logits, size=x.shape[-1], mode="linear", align_corners=False)
        return logits
