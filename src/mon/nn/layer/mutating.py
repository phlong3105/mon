#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements feature extraction layers."""

from __future__ import annotations

__all__ = [
    "ChannelShuffle", "Chuncat", "Concat", "ExtractFeature", "ExtractFeatures",
    "ExtractItem", "ExtractItems", "Flatten", "FlattenSingle", "Fold",
    "Foldcut", "InterpolateConcat", "Join", "Max", "Permute", "PixelShuffle",
    "PixelUnshuffle", "Shortcut", "SoftmaxFusion", "Sum", "Unflatten", "Unfold",
]

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional
from torchvision.ops.misc import Permute

from mon.core import builtins
from mon.globals import LAYERS
from mon.nn.layer import base


# region Extraction

@LAYERS.register()
class ExtractFeature(base.LayerParsingMixin, nn.Module):
    """Extract a feature at :param:`index` in a tensor.
    
    Args:
        index: The index of the feature to extract.
    """

    def __init__(self, index: int):
        super().__init__()
        self.index = index
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not input.ndim == 4:
            raise ValueError(
                f"input's number of dimensions must be == 4, but got {input.ndim}."
            )
        x = input
        y = x[:, self.index, :, :]
        return y
    
    @classmethod
    def parse_layer_args(cls, f: int, args: list, ch: list) -> tuple[list, list]:
        c2 = args[0]
        ch.append(c2)
        return args, ch


@LAYERS.register()
class ExtractFeatures(base.LayerParsingMixin, nn.Module):
    """Extract features between :param:`start` index and :param:`end` index in a
    tensor.
    
    Args:
        start: The start index of the features to extract.
        end: The end index of the features to extract.
    """
    
    def __init__(self, start: int, end: int):
        super().__init__()
        self.start = start
        self.end   = end
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not input.ndim == 4:
            raise ValueError(
                f"input's number of dimensions must be == 4, but got {input.ndim}."
            )
        x = input
        y = x[:, self.start:self.end, :, :]
        return y
    
    @classmethod
    def parse_layer_args(cls, f: int, args: list, ch: list) -> tuple[list, list]:
        c2 = args[1] - args[0]
        ch.append(c2)
        return args, ch


@LAYERS.register()
class ExtractItem(base.PassThroughLayerParsingMixin, nn.Module):
    """Extract an item (feature) at :param:`index` in a sequence of tensors.
    
    Args:
        index: The index of the item to extract.
    """

    def __init__(self, index: int):
        super().__init__()
        self.index = index
    
    def forward(self, input: Sequence[torch.Tensor]) -> torch.Tensor:
        x = input
        if isinstance(x, torch.Tensor):
            return x
        elif isinstance(x, list | tuple):
            return x[self.index]
        else:
            raise TypeError(
                f"input must be a list or tuple of tensors, but got: {type(input)}."
            )


@LAYERS.register()
class ExtractItems(base.PassThroughLayerParsingMixin, nn.Module):
    """Extract a list of items (features) at `indexes` in a sequence of
    tensors.
    
    Args:
        indexes: The indexes of the items to extract.
    """
    
    def __init__(self, indexes: Sequence[int]):
        super().__init__()
        self.indexes = indexes
    
    def forward(self, input: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        x = input
        if isinstance(x, torch.Tensor):
            y = [x]
            return y
        elif isinstance(x, list | tuple):
            y = [x[i] for i in self.indexes]
            return y
        raise TypeError(
            f"input must be a list or tuple of tensors, but got: {type(input)}."
        )


@LAYERS.register()
class Max(base.PassThroughLayerParsingMixin, nn.Module):
    
    def __init__(self, dim: int, keepdim: bool = False):
        super().__init__()
        self.dim     = dim
        self.keepdim = keepdim
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = torch.max(input=x, dim=self.dim, keepdim=self.keepdim)
        return y


# endregion


# region Flattening

@LAYERS.register()
class FlattenSingle(base.PassThroughLayerParsingMixin, nn.Module):
    """Flatten a tensor along a single dimension.

    Args:
        dim: Dimension to flatten. Default: 1.
    """
    
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = torch.flatten(x, self.dim)
        return y


@LAYERS.register()
class Flatten(base.PassThroughLayerParsingMixin, nn.Flatten):
    pass


@LAYERS.register()
class Unflatten(base.PassThroughLayerParsingMixin, nn.Unflatten):
    pass


# endregion


# region Folding

@LAYERS.register()
class Fold(base.PassThroughLayerParsingMixin, nn.Fold):
    pass


@LAYERS.register()
class Unfold(base.PassThroughLayerParsingMixin, nn.Unfold):
    pass


# endregion


# region Merging

@LAYERS.register()
class Concat(base.ConcatLayerParsingMixin, nn.Module):
    """Concatenate a list of tensors along dimension.
    
    Args:
        dim: Dimension to concat to. Default: 1.
    """
    
    def __init__(self, dim: str | ellipsis | None = 1, ):
        super().__init__()
        self.dim = dim
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = torch.cat(list(x), dim=self.dim)
        return y


@LAYERS.register()
class Chuncat(base.ConcatLayerParsingMixin, nn.Module):
    """
    
    Args:
        dim: Dimension to concat to. Default: 1.
    """
    
    def __init__(self, dim: str | ellipsis | None = 1):
        super().__init__()
        self.dim = dim
    
    def forward(self, input: Sequence[torch.Tensor]) -> torch.Tensor:
        x  = input
        y1 = []
        y2 = []
        for x_i in x:
            x_i_1, x_i_2 = x_i.chunk(2, self.dim)
            y1.append(x_i_1)
            y2.append(x_i_2)
        y = torch.cat(y1 + y2, dim=self.dim)
        return y


@LAYERS.register()
class Foldcut(base.PassThroughLayerParsingMixin, nn.Module):
    """
    
    Args:
        dim: Dimension to concat to. Default: 0.
    """
    
    def __init__(self, dim: str | ellipsis | None = 0):
        super().__init__()
        self.dim = dim
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x      = input
        x1, x2 = x.chunk(2, dim=self.dim)
        y      = x1 + x2
        return y
    
    @classmethod
    def parse_layer_args(cls, f: int, args: list, ch: list) -> tuple[list, list]:
        c2 = ch[f] // 2
        ch.append(c2)
        return args, ch


@LAYERS.register()
class InterpolateConcat(base.ConcatLayerParsingMixin, nn.Module):
    """Concatenate a list of tensors along dimension.
    
    Args:
        dim: Dimension to concat to. Default: 1.
    """

    def __init__(self, dim: str | ellipsis | None = 1):
        super().__init__()
        self.dim = dim
    
    def forward(self, input: Sequence[torch.Tensor]) -> torch.Tensor:
        x     = input
        sizes = [list(x_i.size()) for x_i in x]
        hs    = [s[2] for s in sizes]
        ws    = [s[3] for s in sizes]
        h, w  = max(hs), max(ws)
        y = []
        for x_i in x:
            s = x_i.size()
            if s[2] != h or s[3] != w:
                y.append(functional.interpolate(input=x_i, size=(h, w)))
            else:
                y.append(x_i)
        y = torch.cat(builtins.to_list(y), dim=self.dim)
        return y


@LAYERS.register()
class Join(base.MergingLayerParsingMixin, nn.Module):
    """Join multiple features and return a list tensors."""
    
    def forward(self, input: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        x = input
        y = builtins.to_list(x)
        return y


@LAYERS.register()
class Shortcut(base.MergingLayerParsingMixin, nn.Module):
    """
    
    Args:
        dim: Dimension to concat to. Default: 0.
    """
    
    def __init__(self, dim: str | ellipsis | None = 0):
        super().__init__()
        self.dim = dim
    
    def forward(self, input: Sequence[torch.Tensor]) -> torch.Tensor:
        x = input
        y = x[0] + x[1]
        return y


@LAYERS.register()
class SoftmaxFusion(base.MergingLayerParsingMixin, nn.Module):
    """Weighted sum of multiple layers https://arxiv.org/abs/1911.09070. Apply
    softmax to each weight, such that all weights are normalized to be a
    probability with value range from 0 to 1, representing the importance of
    each input.
    
    Args:
        n: Number of inputs.
    """
    
    def __init__(self, n: int, weight: bool = False):
        super().__init__()
        self.weight = weight  # Apply weights boolean
        self.iter   = range(n - 1)  # iter object
        if weight:
            # Layer weights
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = x[0]
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


@LAYERS.register()
class Sum(base.MergingLayerParsingMixin, nn.Module):
    
    def forward(self, input: Sequence[torch.Tensor]) -> torch.Tensor:
        x = input
        y = x[0]
        for i in range(1, len(x)):
            y += x[i]
        return y


# endregion


# region Shuffling

@LAYERS.register()
class ChannelShuffle(base.PassThroughLayerParsingMixin, nn.ChannelShuffle):
    pass


@LAYERS.register()
class PixelShuffle(base.PassThroughLayerParsingMixin, nn.PixelShuffle):
    pass


@LAYERS.register()
class PixelUnshuffle(base.PassThroughLayerParsingMixin, nn.PixelUnshuffle):
    pass

# endregion
