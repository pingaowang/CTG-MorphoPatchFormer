"""Paper-model layers extracted from the research implementation."""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PatchMeta1D:
    """Bookkeeping needed to crop decoded patches back to the original length."""

    original_len: int
    padded_len: int
    num_patches: int
    pad_len: int


def _validate_patch_params(patch_len: int, stride: int) -> tuple[int, int]:
    patch_len = int(patch_len)
    stride = int(stride)
    if patch_len <= 0:
        raise ValueError("patch_len must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")
    if stride > patch_len:
        raise ValueError("stride must be <= patch_len so patches cover every timestep.")
    return patch_len, stride


class SinusoidalPositionalEncoding1D(nn.Module):
    """Dynamic sinusoidal position encoding for patch-token sequences."""

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D], where N is the number of patch tokens.
        _, n_tokens, d_model = x.shape
        pos = torch.arange(n_tokens, device=x.device, dtype=x.dtype).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, device=x.device, dtype=x.dtype)
            * (-math.log(10000.0) / max(1, d_model))
        )
        pe = torch.zeros(n_tokens, d_model, device=x.device, dtype=x.dtype)
        pe[:, 0::2] = torch.sin(pos * div)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        return self.dropout(x + pe.unsqueeze(0))


class PatchEmbedding1D(nn.Module):
    """Convert ``[B, C, T]`` features into ``[B, N, D]`` patch tokens."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        patch_len: int = 32,
        stride: int | None = None,
        d_model: int = 128,
        dropout: float = 0.0,
        pad_mode: str = "replicate",
    ) -> None:
        super().__init__()
        stride = patch_len if stride is None else stride
        self.patch_len, self.stride = _validate_patch_params(patch_len, stride)
        self.in_channels = int(in_channels)
        self.d_model = int(d_model)
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive.")
        self.pad_mode = str(pad_mode)
        self.proj = nn.Linear(self.in_channels * self.patch_len, self.d_model)
        self.norm = nn.LayerNorm(self.d_model)
        self.pos = SinusoidalPositionalEncoding1D(dropout=dropout)

    def _num_patches_and_pad(self, t: int) -> tuple[int, int, int]:
        if t <= 0:
            raise ValueError("input sequence length T must be positive.")
        if t <= self.patch_len:
            n_patches = 1
        else:
            n_patches = int(math.ceil((t - self.patch_len) / self.stride)) + 1
        padded_len = (n_patches - 1) * self.stride + self.patch_len
        pad_len = padded_len - t
        return n_patches, padded_len, pad_len

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, PatchMeta1D]:
        if x.ndim != 3:
            raise ValueError(f"PatchEmbedding1D expects [B, C, T], got shape {tuple(x.shape)}.")
        bsz, channels, t = x.shape
        if channels != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {channels}.")

        n_patches, padded_len, pad_len = self._num_patches_and_pad(int(t))
        if pad_len > 0:
            # [B, C, T] -> [B, C, padded_len]; tail padding lets the last patch cover T.
            x = F.pad(x, (0, pad_len), mode=self.pad_mode)

        # [B, C, padded_len] -> [B, C, N, patch_len]
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # [B, C, N, patch_len] -> [B, N, C * patch_len]
        patches = patches.permute(0, 2, 1, 3).contiguous().view(bsz, n_patches, -1)
        tokens = self.proj(patches)  # [B, N, D]
        tokens = self.norm(tokens)
        tokens = self.pos(tokens)
        meta = PatchMeta1D(
            original_len=int(t),
            padded_len=int(padded_len),
            num_patches=int(n_patches),
            pad_len=int(pad_len),
        )
        return tokens, meta


class TransformerEncoder1D(nn.Module):
    """Transformer encoder over patch tokens ``[B, N, D]``."""

    def __init__(
        self,
        *,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
        dim_feedforward: int | None = None,
    ) -> None:
        super().__init__()
        d_model = int(d_model)
        n_heads = int(n_heads)
        num_layers = int(num_layers)
        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if n_heads <= 0:
            raise ValueError("n_heads must be positive.")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if num_layers < 0:
            raise ValueError("num_layers must be >= 0.")
        if num_layers == 0:
            self.encoder = nn.Identity()
            return
        ff = int(dim_feedforward or d_model * 4)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, N, D] -> [B, N, D]
        return self.encoder(tokens)


class _ConvRefine1D(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        p = float(dropout)
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Dropout(p=p) if p > 0.0 else nn.Identity(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class PatchToSequenceDecoder1D(nn.Module):
    """Decode patch tokens into time-aligned features.

    The token decoder first predicts one feature patch per token, then folds
    those patches back onto the time axis with overlap-add averaging.
    The legacy ``num_classes`` argument is the number of output channels;
    MorphoPatchFormer sets it to 48 feature channels, not three event classes.
    """

    def __init__(
        self,
        *,
        d_model: int = 128,
        num_classes: int = 3,
        patch_len: int = 32,
        stride: int | None = None,
        decoder_channels: int = 64,
        dropout: float = 0.0,
        skip_channels: int = 0,
    ) -> None:
        super().__init__()
        stride = patch_len if stride is None else stride
        self.patch_len, self.stride = _validate_patch_params(patch_len, stride)
        self.d_model = int(d_model)
        self.num_classes = int(num_classes)
        self.decoder_channels = int(decoder_channels)
        self.skip_channels = int(skip_channels)
        if self.decoder_channels <= 0:
            raise ValueError("decoder_channels must be positive.")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive.")

        self.patch_project = nn.Linear(self.d_model, self.decoder_channels * self.patch_len)
        self.skip_proj = (
            nn.Conv1d(self.skip_channels, self.decoder_channels, kernel_size=1)
            if self.skip_channels > 0
            else None
        )
        self.refine = nn.Sequential(
            _ConvRefine1D(self.decoder_channels, dropout),
            _ConvRefine1D(self.decoder_channels, dropout),
        )
        self.out_conv = nn.Conv1d(self.decoder_channels, self.num_classes, kernel_size=1)

    def _overlap_add(self, patch_values: torch.Tensor, padded_len: int) -> torch.Tensor:
        # patch_values: [B, N, decoder_channels, patch_len]
        bsz, n_patches, channels, patch_len = patch_values.shape
        if patch_len != self.patch_len:
            raise ValueError(f"expected patch_len {self.patch_len}, got {patch_len}.")
        expected_len = (n_patches - 1) * self.stride + self.patch_len
        if int(padded_len) != expected_len:
            raise ValueError(f"padded_len={padded_len} does not match {n_patches} patches.")

        # [B, N, C, L] -> [B, C * L, N], the column format expected by fold().
        fold_in = patch_values.permute(0, 2, 3, 1).contiguous().view(
            bsz, channels * self.patch_len, n_patches
        )
        seq = F.fold(
            fold_in,
            output_size=(1, int(padded_len)),
            kernel_size=(1, self.patch_len),
            stride=(1, self.stride),
        ).squeeze(2)
        weight = F.fold(
            torch.ones_like(fold_in),
            output_size=(1, int(padded_len)),
            kernel_size=(1, self.patch_len),
            stride=(1, self.stride),
        ).squeeze(2)
        return seq / weight.clamp_min(1.0)

    def forward(
        self,
        tokens: torch.Tensor,
        meta: PatchMeta1D,
        *,
        skip: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"PatchToSequenceDecoder1D expects [B, N, D], got {tuple(tokens.shape)}.")
        bsz, n_patches, _ = tokens.shape
        if n_patches != meta.num_patches:
            raise ValueError(f"token count {n_patches} != meta.num_patches {meta.num_patches}.")

        # [B, N, D] -> [B, N, decoder_channels, patch_len]
        patch_values = self.patch_project(tokens).view(
            bsz, n_patches, self.decoder_channels, self.patch_len
        )
        # [B, N, Cdec, patch_len] -> [B, Cdec, padded_len] -> [B, Cdec, T]
        seq = self._overlap_add(patch_values, meta.padded_len)
        seq = seq[..., : meta.original_len]

        if skip is not None:
            if self.skip_proj is None:
                raise ValueError("skip was passed, but skip_channels=0 in decoder.")
            if skip.shape[-1] != meta.original_len:
                skip = F.interpolate(skip, size=meta.original_len, mode="linear", align_corners=False)
            seq = seq + self.skip_proj(skip)

        features = self.refine(seq)
        return self.out_conv(features)  # [B, output feature channels, T]


class DepthwiseConvBlock1D(nn.Module):
    """Depthwise-separable Conv1D block preserving ``[B, C, T]`` shape."""

    def __init__(self, channels: int, kernel_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        channels = int(channels)
        kernel_size = int(kernel_size)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        p = float(dropout)
        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=channels,
                bias=False,
            ),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Dropout(p=p) if p > 0.0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class CNNStem1D(nn.Module):
    """Local CTG morphology encoder.

    Shape: ``[B, in_channels, T] -> [B, stem_channels, T]``.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        stem_channels: int = 48,
        depth: int = 3,
        kernel_sizes: tuple[int, ...] = (7, 15, 31),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if int(depth) <= 0:
            raise ValueError("stem depth must be positive.")
        if not kernel_sizes:
            raise ValueError("kernel_sizes must not be empty.")
        self.input = nn.Sequential(
            nn.Conv1d(int(in_channels), int(stem_channels), kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, int(stem_channels)),
            nn.GELU(),
        )
        blocks = []
        for i in range(int(depth)):
            blocks.append(DepthwiseConvBlock1D(int(stem_channels), int(kernel_sizes[i % len(kernel_sizes)]), dropout))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.input(x))


class GatedFusion1D(nn.Module):
    """Fuse branch features with either concat projection or learned gates."""

    def __init__(self, *, channels: int, n_branches: int, fusion: str = "gated") -> None:
        super().__init__()
        self.channels = int(channels)
        self.n_branches = int(n_branches)
        self.fusion = str(fusion).strip().lower()
        if self.n_branches <= 0:
            raise ValueError("n_branches must be positive.")
        if self.fusion not in ("concat", "gated", "se_gated"):
            raise ValueError("fusion must be one of: concat, gated, se_gated.")
        self.proj = nn.Conv1d(self.channels * self.n_branches, self.channels, kernel_size=1)
        if self.fusion in ("gated", "se_gated"):
            self.gate = nn.Conv1d(self.channels * self.n_branches, self.n_branches, kernel_size=1)
        else:
            self.gate = None
        self.se = (
            nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(self.channels * self.n_branches, max(1, self.channels // 4), kernel_size=1),
                nn.GELU(),
                nn.Conv1d(max(1, self.channels // 4), self.n_branches, kernel_size=1),
            )
            if self.fusion == "se_gated"
            else None
        )

    def forward(self, branches: list[torch.Tensor]) -> torch.Tensor:
        if len(branches) != self.n_branches:
            raise ValueError(f"expected {self.n_branches} branches, got {len(branches)}.")
        if self.n_branches == 1:
            return branches[0]
        x = torch.cat(branches, dim=1)
        if self.gate is None:
            return self.proj(x)
        logits = self.gate(x)
        if self.se is not None:
            logits = logits + self.se(x)
        weights = torch.softmax(logits, dim=1)
        stacked = torch.stack(branches, dim=1)  # [B, K, C, T]
        return (stacked * weights.unsqueeze(2)).sum(dim=1)
