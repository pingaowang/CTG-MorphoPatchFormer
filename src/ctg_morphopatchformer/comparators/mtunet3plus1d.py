"""1D MTU-Net3+ reproduction adapter.

Paper target: Wang et al., "Automated analysis of fetal heart rate
baseline/acceleration/deceleration using MTU-Net3+ model", Biomedical
Engineering Letters, 2024.

Method identity retained for this repository:
- five-level U-Net3+ encoder/decoder with full-scale skip aggregation;
- deep-supervision segmentation heads at decoder scales;
- baseline regression branch using self-attention and BiLSTM.

The baseline output is exposed through ``last_baseline``. Intermediate decoder
outputs are structural auxiliary heads, not a supplied training procedure.
This five-level paper-oriented adapter is not certified as the executable that
generated the archived Ubuntu faithful_v3 benchmark predictions. See METHODS.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBNReLU1d(nn.Module):
    def __init__(self, ci: int, co: int, kernel_size: int = 3, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ci, co, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(co),
            nn.ReLU(inplace=False),
            nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _DoubleConv1d(nn.Module):
    def __init__(self, ci: int, co: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _ConvBNReLU1d(ci, co, 3, dropout),
            _ConvBNReLU1d(co, co, 3, dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SelfAttention1d(nn.Module):
    def __init__(self, channels: int, heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        if int(channels) % int(heads) != 0:
            heads = 1
        self.attn = nn.MultiheadAttention(channels, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2)
        z, _ = self.attn(y, y, y, need_weights=False)
        return self.norm(y + z).transpose(1, 2)


class _FullScaleDecoder1d(nn.Module):
    def __init__(self, in_channels_by_scale: list[int], out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.proj = nn.ModuleList([_ConvBNReLU1d(ci, out_channels, 3, dropout) for ci in in_channels_by_scale])
        self.fuse = _DoubleConv1d(out_channels * len(in_channels_by_scale), out_channels, dropout)

    def forward(self, features: list[torch.Tensor], target_len: int) -> torch.Tensor:
        aligned = []
        for feat, proj in zip(features, self.proj):
            if feat.shape[-1] != target_len:
                feat = F.interpolate(feat, size=target_len, mode="linear", align_corners=False)
            aligned.append(proj(feat))
        return self.fuse(torch.cat(aligned, dim=1))


class MTUNet3Plus1D(nn.Module):
    """U-Net3+ style 1D multitask FHR model."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        num_classes: int = 3,
        base_channels: int = 32,
        dropout: float = 0.0,
        bilstm_layers: int = 1,
        attn_heads: int = 4,
        deep_supervision: bool = True,
    ) -> None:
        super().__init__()
        bc = int(base_channels)
        self.num_classes = int(num_classes)
        self.deep_supervision = bool(deep_supervision)
        chs = [bc, bc * 2, bc * 4, bc * 8, bc * 16]

        self.enc1 = _DoubleConv1d(in_channels, chs[0], dropout)
        self.enc2 = _DoubleConv1d(chs[0], chs[1], dropout)
        self.enc3 = _DoubleConv1d(chs[1], chs[2], dropout)
        self.enc4 = _DoubleConv1d(chs[2], chs[3], dropout)
        self.enc5 = _DoubleConv1d(chs[3], chs[4], dropout)
        self.pool = nn.MaxPool1d(2)

        # U-Net3+ full-scale decoder: every decoder sees all encoder scales and
        # coarser decoder features that already exist.
        self.dec4 = _FullScaleDecoder1d([*chs], chs[3], dropout)
        self.dec3 = _FullScaleDecoder1d([*chs, chs[3]], chs[2], dropout)
        self.dec2 = _FullScaleDecoder1d([*chs, chs[3], chs[2]], chs[1], dropout)
        self.dec1 = _FullScaleDecoder1d([*chs, chs[3], chs[2], chs[1]], chs[0], dropout)

        self.seg_head = nn.Conv1d(chs[0], num_classes, kernel_size=1)

        self.base_attn = _SelfAttention1d(chs[0], heads=attn_heads, dropout=dropout)
        self.base_lstm = nn.LSTM(
            chs[0],
            chs[0],
            num_layers=int(bilstm_layers),
            batch_first=True,
            bidirectional=True,
            dropout=float(dropout) if int(bilstm_layers) > 1 else 0.0,
        )
        self.base_head = nn.Conv1d(chs[0] * 2, 1, kernel_size=1)

        if self.deep_supervision:
            self.ds4 = nn.Conv1d(chs[3], num_classes, kernel_size=1)
            self.ds3 = nn.Conv1d(chs[2], num_classes, kernel_size=1)
            self.ds2 = nn.Conv1d(chs[1], num_classes, kernel_size=1)

        self._last_baseline: torch.Tensor | None = None
        self._ds_logits: tuple[torch.Tensor, ...] = ()

    @property
    def last_baseline(self) -> torch.Tensor | None:
        return self._last_baseline

    @property
    def deep_supervision_logits(self) -> tuple[torch.Tensor, ...]:
        return self._ds_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        encs = [e1, e2, e3, e4, e5]

        d4 = self.dec4(encs, e4.shape[-1])
        d3 = self.dec3([*encs, d4], e3.shape[-1])
        d2 = self.dec2([*encs, d4, d3], e2.shape[-1])
        d1 = self.dec1([*encs, d4, d3, d2], e1.shape[-1])

        seg_logits = self.seg_head(d1)
        if seg_logits.shape[-1] != x.shape[-1]:
            seg_logits = F.interpolate(seg_logits, size=x.shape[-1], mode="linear", align_corners=False)

        b = self.base_attn(d1)
        seq, _ = self.base_lstm(b.transpose(1, 2))
        baseline = self.base_head(seq.transpose(1, 2))
        if baseline.shape[-1] != x.shape[-1]:
            baseline = F.interpolate(baseline, size=x.shape[-1], mode="linear", align_corners=False)
        self._last_baseline = baseline

        if self.training and self.deep_supervision:
            self._ds_logits = (
                F.interpolate(self.ds4(d4), size=x.shape[-1], mode="linear", align_corners=False),
                F.interpolate(self.ds3(d3), size=x.shape[-1], mode="linear", align_corners=False),
                F.interpolate(self.ds2(d2), size=x.shape[-1], mode="linear", align_corners=False),
            )
        else:
            self._ds_logits = ()
        return seg_logits
