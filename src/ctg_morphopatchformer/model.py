"""The local-morphology / overlapping-patch model used by the paper."""
from __future__ import annotations

import torch
from torch import nn

from .layers import (CNNStem1D, DepthwiseConvBlock1D, GatedFusion1D,
                     PatchEmbedding1D, PatchToSequenceDecoder1D, TransformerEncoder1D)


class MorphoPatchFormer(nn.Module):
    """Input [batch, channels, time]; output three-class unnormalized logits.

    The experiments use one FHR channel and 1200-sample windows. Physical
    duration depends on the actual input sampling rate.
    Extra input channels are an architectural option, not a validated result.
    Experimental Fourier, wavelet and attention variants are not included.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 3,
                 stem_channels: int = 48, stem_depth: int = 3,
                 stem_kernel_sizes: tuple[int, ...] = (7, 15, 31),
                 dropout: float = 0.1, patch_len: int = 64, stride: int = 32,
                 d_model: int = 128, n_heads: int = 4, num_layers: int = 2,
                 decoder_channels: int = 48, refine_depth: int = 2):
        super().__init__()
        if in_channels < 1 or num_classes != 3 or not 0 <= dropout < 1:
            raise ValueError("Use positive input channels, three classes and dropout in [0, 1).")
        if stem_channels < 1 or refine_depth < 0:
            raise ValueError("Invalid stem width or refinement depth.")
        self.in_channels = in_channels
        self.stem = CNNStem1D(in_channels=in_channels, stem_channels=stem_channels,
                              depth=stem_depth, kernel_sizes=stem_kernel_sizes, dropout=dropout)
        self.patch_embedding = PatchEmbedding1D(in_channels=stem_channels, patch_len=patch_len,
                                                stride=stride, d_model=d_model, dropout=dropout)
        self.patch_encoder = TransformerEncoder1D(d_model=d_model, n_heads=n_heads,
                                                  num_layers=num_layers, dropout=dropout)
        self.patch_decoder = PatchToSequenceDecoder1D(d_model=d_model, num_classes=stem_channels,
            patch_len=patch_len, stride=stride, decoder_channels=decoder_channels,
            dropout=dropout, skip_channels=stem_channels)
        # Retain the original state-dict layout, including the inactive concat projection.
        self.fusion = GatedFusion1D(channels=stem_channels, n_branches=2, fusion="gated")
        self.refine = nn.Sequential(*[DepthwiseConvBlock1D(stem_channels,
            stem_kernel_sizes[i % len(stem_kernel_sizes)], dropout) for i in range(refine_depth)])
        self.out_conv = nn.Conv1d(stem_channels, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.in_channels or x.shape[2] < 1:
            raise ValueError("Expected nonempty [batch, configured_channels, time] input.")
        local = self.stem(x)
        tokens, meta = self.patch_embedding(local)
        context = self.patch_decoder(self.patch_encoder(tokens), meta, skip=local)
        return self.out_conv(self.refine(self.fusion([local, context])))
