"""1D WaveFHR-VNet reproduction adapter.

Paper target: Lu et al., "Multi-Scale, Multi-Basis Wavelet Voting Network for
Automatic Analysis of Fetal Heart Rate Signals", IEEE JBHI, 2025.

This implementation keeps the method identity from the paper text:
- U-Net-style encoder/decoder.
- DWT-like low/high-frequency split in every encoder block.
- Transformer long-term context modeling (LTCM) at the bottleneck.
- Interactive Coefficient Selection (ICS) + CTAB on detail skip streams.
- Inverse reconstruction in the decoder.
- Multi-basis voting fusion across wavelet branches.

The branch filters are deterministic orthogonal/biorthogonal-style FIR pairs
encoded as fixed grouped convolutions. It is a paper-faithful 1D adaptation for
the repository's three-class segmentation interface.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt


_FILTERS: dict[str, tuple[list[float], list[float]]] = {
    "haar": ([0.70710678, 0.70710678], [-0.70710678, 0.70710678]),
    "db4": (
        [-0.0105974, 0.0328830, 0.0308414, -0.1870348, -0.0279838, 0.6308808, 0.7148466, 0.2303778],
        [-0.2303778, 0.7148466, -0.6308808, -0.0279838, 0.1870348, 0.0308414, -0.0328830, -0.0105974],
    ),
    "db6": (
        [
            0.0010773,
            -0.0047773,
            0.0005538,
            0.0315820,
            0.0275229,
            -0.0975016,
            -0.1297669,
            0.2262647,
            0.3152504,
            -0.7511339,
            0.4946239,
            0.1115407,
        ],
        [
            -0.1115407,
            0.4946239,
            0.7511339,
            0.3152504,
            -0.2262647,
            -0.1297669,
            0.0975016,
            0.0275229,
            -0.0315820,
            0.0005538,
            0.0047773,
            0.0010773,
        ],
    ),
    "sym4": (
        [-0.0757657, -0.0296355, 0.4976187, 0.8037388, 0.2978578, -0.0992195, -0.0126039, 0.0322231],
        [-0.0322231, -0.0126039, 0.0992195, 0.2978578, -0.8037388, 0.4976187, 0.0296355, -0.0757657],
    ),
    "sym5": (
        [0.0273331, 0.0295195, -0.0391342, 0.1993975, 0.7234077, 0.6339789, 0.0166021, -0.1753281, -0.0211018, 0.0195389],
        [-0.0195389, -0.0211018, 0.1753281, 0.0166021, -0.6339789, 0.7234077, -0.1993975, -0.0391342, -0.0295195, 0.0273331],
    ),
    "bior3.5": (
        [0.0, 0.0, -0.0138107, 0.0414321, 0.0524806, -0.2679272, -0.0718155, 0.9667476, 0.9667476, -0.0718155, -0.2679272, 0.0524806, 0.0414321, -0.0138107],
        [0.0, 0.0, 0.0, 0.0, -0.1767767, 0.5303301, -0.5303301, 0.1767767, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ),
}


class _ConvBNReLU1d(nn.Module):
    def __init__(self, ci: int, co: int, kernel_size: int = 31, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ci, co, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(co),
            nn.ReLU(inplace=False),
            nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FixedDWT1D(nn.Module):
    def __init__(self, channels: int, basis: str = "db4") -> None:
        super().__init__()
        self.basis = str(basis)
        try:
            dec_lo, dec_hi, _, _ = pywt.Wavelet(self.basis).filter_bank
        except Exception as exc:
            if self.basis not in _FILTERS:
                raise ValueError(f"Unsupported wavelet basis {basis!r}") from exc
            dec_lo, dec_hi = _FILTERS[self.basis]
        self.register_buffer("dec_lo", torch.tensor(dec_lo, dtype=torch.float32).view(1, 1, -1))
        self.register_buffer("dec_hi", torch.tensor(dec_hi, dtype=torch.float32).view(1, 1, -1))
        self.channels = int(channels)
        self.pad = len(dec_lo) // 2 - 1

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = x.size(1)
        lo = self.dec_lo.to(dtype=x.dtype).repeat(c, 1, 1)
        hi = self.dec_hi.to(dtype=x.dtype).repeat(c, 1, 1)
        approx = F.conv1d(x, lo, stride=2, padding=self.pad, groups=c)
        detail = F.conv1d(x, hi, stride=2, padding=self.pad, groups=c)
        return approx, detail


class FixedIDWT1D(nn.Module):
    def __init__(self, channels: int, basis: str = "db4") -> None:
        super().__init__()
        self.basis = str(basis)
        try:
            _, _, rec_lo, rec_hi = pywt.Wavelet(self.basis).filter_bank
        except Exception as exc:
            if self.basis not in _FILTERS:
                raise ValueError(f"Unsupported wavelet basis {basis!r}") from exc
            rec_lo, rec_hi = _FILTERS[self.basis]
        self.register_buffer("rec_lo", torch.tensor(rec_lo, dtype=torch.float32).view(1, 1, -1))
        self.register_buffer("rec_hi", torch.tensor(rec_hi, dtype=torch.float32).view(1, 1, -1))
        self.channels = int(channels)
        self.pad = (len(rec_lo) - 1) // 2

    def forward(self, c_a: torch.Tensor, c_d: torch.Tensor, target_len: int) -> torch.Tensor:
        c = c_a.size(1)
        lo = self.rec_lo.to(dtype=c_a.dtype).repeat(c, 1, 1)
        hi = self.rec_hi.to(dtype=c_a.dtype).repeat(c, 1, 1)
        y = F.conv_transpose1d(c_a, lo, stride=2, padding=self.pad, groups=c)
        y = y + F.conv_transpose1d(c_d, hi, stride=2, padding=self.pad, groups=c)
        if y.shape[-1] != int(target_len):
            y = F.interpolate(y, size=int(target_len), mode="linear", align_corners=False)
        return y


class _CTAB1d(nn.Module):
    """Channel-temporal attention block for refined detail coefficients."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, int(channels) // int(reduction))
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv1d(hidden, channels, kernel_size=1, bias=False),
        )
        self.temporal = nn.Conv1d(2, 1, kernel_size=31, padding=15, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_c = F.adaptive_avg_pool1d(x, 1)
        max_c = F.adaptive_max_pool1d(x, 1)
        ch_mask = torch.sigmoid(self.mlp(avg_c) + self.mlp(max_c))
        z = x * ch_mask
        avg_t = z.mean(dim=1, keepdim=True)
        max_t = z.max(dim=1, keepdim=True).values
        t_mask = torch.sigmoid(self.temporal(torch.cat([avg_t, max_t], dim=1)))
        return z * t_mask


class _ICS1d(nn.Module):
    """Interactive Coefficient Selection: cA-guided detail filtering."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv1d(channels * 2, channels * 2, kernel_size=31, padding=15),
            nn.BatchNorm1d(channels * 2),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
            nn.Conv1d(channels * 2, channels * 2, kernel_size=31, padding=15),
            nn.BatchNorm1d(channels * 2),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
            nn.Conv1d(channels * 2, channels, kernel_size=31, padding=15),
            nn.Sigmoid(),
        )
        self.ctab = _CTAB1d(channels)

    def forward(self, c_a: torch.Tensor, c_d: torch.Tensor) -> torch.Tensor:
        mask = self.gate(torch.cat([c_a, c_d], dim=1))
        gated = c_d * mask + c_d
        return self.ctab(gated)


class _LTCM1d(nn.Module):
    """Transformer bottleneck used as long-term context modeling."""

    def __init__(self, channels: int, heads: int = 4, layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        if int(channels) % int(heads) != 0:
            heads = 1
        enc_layer = nn.TransformerEncoderLayer(
            d_model=int(channels),
            nhead=int(heads),
            dim_feedforward=int(channels) * 4,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.encoder(x.transpose(1, 2))
        return y.transpose(1, 2)


class WaveFHRNet1D(nn.Module):
    """Single-basis WaveFHR-Net branch.

    This mirrors the official repository's ``UNet`` class: one wavelet basis is
    trained per model, and the full WaveFHR-VNet result is obtained by voting
    across five independently trained branches.
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        num_classes: int = 3,
        base_channels: int = 16,
        basis: str = "db4",
        dropout: float = 0.0,
        num_layers: int = 4,
        ltcm_heads: int = 4,
        ltcm_layers: int = 1,
    ) -> None:
        super().__init__()
        bc = int(base_channels)
        nl = int(num_layers)
        if nl != 4:
            raise ValueError("WaveFHRNet1D follows the official four-level WaveFHR-VNet.")
        self.stem = nn.Sequential(
            _ConvBNReLU1d(in_channels, bc, 31, dropout),
            _ConvBNReLU1d(bc, bc, 31, dropout),
        )
        enc_channels = [bc, bc * 2, bc * 4, bc * 8]
        dec_in_channels = [bc * 16, bc * 8, bc * 4, bc * 2]
        dec_out_channels = [bc * 8, bc * 4, bc * 2, bc]
        self.dwt = nn.ModuleList([FixedDWT1D(c, basis) for c in enc_channels])
        self.idwt = nn.ModuleList([FixedIDWT1D(c, basis) for c in dec_out_channels])
        self.approx_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    _ConvBNReLU1d(ci, co, 31, dropout),
                    _ConvBNReLU1d(co, co, 31, dropout),
                )
                for ci, co in zip(enc_channels, [bc * 2, bc * 4, bc * 8, bc * 16])
            ]
        )
        self.ics = nn.ModuleList([_ICS1d(c) for c in enc_channels])
        self.detail_proj = nn.ModuleList(
            [
                nn.Identity(),
                nn.Identity(),
                nn.Identity(),
                nn.Identity(),
            ]
        )
        self.ltcm = _LTCM1d(bc * 16, heads=ltcm_heads, layers=ltcm_layers, dropout=dropout)
        self.dec_reduce = nn.ModuleList(
            [nn.Conv1d(ci, co, kernel_size=1) for ci, co in zip(dec_in_channels, dec_out_channels)]
        )
        self.dec_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    _ConvBNReLU1d(co, co, 31, dropout),
                    _ConvBNReLU1d(co, co, 31, dropout),
                )
                for co in dec_out_channels
            ]
        )
        self.out_conv = nn.Conv1d(bc, int(num_classes), kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        target_lengths = [feat.shape[-1]]
        detail_skips: list[torch.Tensor] = []
        for dwt, approx_block, ics in zip(self.dwt, self.approx_blocks, self.ics):
            c_a, c_d = dwt(feat)
            detail_skips.append(ics(c_a, c_d))
            feat = approx_block(c_a)
            target_lengths.append(feat.shape[-1])

        y = self.ltcm(feat)
        for dec_idx, enc_idx in enumerate(reversed(range(len(self.idwt)))):
            target_len = target_lengths[enc_idx]
            y = self.dec_reduce[dec_idx](y)
            y = self.idwt[dec_idx](y, detail_skips[enc_idx], target_len)
            y = self.dec_blocks[dec_idx](y)
        logits = self.out_conv(y)
        if logits.shape[-1] != x.shape[-1]:
            logits = F.interpolate(logits, size=x.shape[-1], mode="linear", align_corners=False)
        return logits


WaveFHRBranch1D = WaveFHRNet1D


class WaveFHRVNet1D(nn.Module):
    """Multi-basis WaveFHR-VNet wrapper.

    The official code trains one ``WaveFHRNet1D`` per basis and applies hard
    voting at evaluation. This wrapper is useful for smoke tests or joint local
    experiments; for strict reproduction, train/export the single-basis models
    and vote over their predictions.
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        num_classes: int = 3,
        base_channels: int = 16,
        bases: Iterable[str] = ("db4", "db6", "sym4", "sym5", "bior3.5"),
        dropout: float = 0.0,
        fusion: str = "mean_logits",
        num_layers: int = 4,
        ltcm_heads: int = 4,
        ltcm_layers: int = 1,
    ) -> None:
        super().__init__()
        basis_list = tuple(str(b).strip() for b in bases if str(b).strip())
        if not basis_list:
            raise ValueError("WaveFHRVNet1D requires at least one wavelet basis.")
        self.fusion = str(fusion).strip().lower()
        self.branches = nn.ModuleList(
            [
                WaveFHRNet1D(
                    in_channels=in_channels,
                    num_classes=num_classes,
                    base_channels=base_channels,
                    basis=b,
                    dropout=dropout,
                    num_layers=num_layers,
                    ltcm_heads=ltcm_heads,
                    ltcm_layers=ltcm_layers,
                )
                for b in basis_list
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.stack([branch(x) for branch in self.branches], dim=0)
        if self.fusion in ("mean", "mean_logits"):
            return logits.mean(dim=0)
        if self.fusion in ("vote", "voting", "soft_vote", "soft-vote"):
            probs = torch.softmax(logits, dim=2)
            avg = probs.mean(dim=0).clamp_min(1.0e-8)
            return torch.log(avg)
        if self.fusion in ("max", "max_logits"):
            return logits.max(dim=0).values
        if self.fusion in ("or", "or_logits"):
            bg = logits[:, :, 0:1, :].min(dim=0).values
            events = logits[:, :, 1:, :].max(dim=0).values
            return torch.cat([bg, events], dim=1)
        raise ValueError(f"Unknown WaveFHRVNet1D fusion={self.fusion!r}.")


def hard_vote_from_logits(logits_by_basis: Iterable[torch.Tensor]) -> torch.Tensor:
    """Return class labels from official hard voting over basis logits."""
    stacked = torch.stack([torch.argmax(logits, dim=1) for logits in logits_by_basis], dim=0)
    return torch.mode(stacked, dim=0).values
