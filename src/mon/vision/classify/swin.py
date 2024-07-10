#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements Swin Transformer models."""

from __future__ import annotations

__all__ = [
    "SwinTransformer",
    "Swin_B",
    "Swin_S",
    "Swin_T",
    "Swin_V2_B",
    "Swin_V2_S",
    "Swin_V2_T",
]

import functools
from abc import ABC
from typing import Any

import torch
from torchvision import ops

from mon import core, nn
from mon.core import _callable
from mon.globals import MODELS, Scheme
from mon.vision.classify import base

console = core.console


# region Module

class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block.
    
    Args:
        channels: Number of input channels.
        num_heads: Number of attention heads.
        window_size: Window size.
        shift_size: Shift size for shifted window attention.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim. Default: ``4.0``.
        dropout: Dropout rate. Default: ``0.0``.
        attention_dropout: Attention dropout rate. Default: ``0.0``.
        stochastic_depth_prob: Stochastic depth rate. Default: ``0.0``.
        norm_layer: Normalization layer. Default: :class:`nn.LayerNorm`.
        attn_layer: Attention layer. Default: :class`ShiftedWindowAttention`
    """

    def __init__(
        self,
        channels             : int,
        num_heads            : int,
        window_size          : list[int],
        shift_size           : list[int],
        mlp_ratio            : float     = 4.0,
        dropout              : float     = 0.0,
        attention_dropout    : float     = 0.0,
        stochastic_depth_prob: float     = 0.0,
        norm_layer           : _callable = nn.LayerNorm,
        attn_layer           : _callable = nn.ShiftedWindowAttention,
        *args, **kwargs
    ):
        super().__init__()
        self.norm1 = norm_layer(channels)
        self.attn  = attn_layer(
            channels          = channels,
            window_size       = window_size,
            shift_size        = shift_size,
            num_heads         = num_heads,
            attention_dropout = attention_dropout,
            dropout           = dropout,
        )
        self.stochastic_depth = ops.StochasticDepth(stochastic_depth_prob, "row")
        self.norm2 = norm_layer(channels)
        self.mlp   = nn.MLP(
            in_channels      = channels,
            hidden_channels  = [int(channels * mlp_ratio), channels],
            activation_layer = nn.GELU,
            inplace          = None,
            dropout          = dropout
        )

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.normal_(m.bias, std=1e-6)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        return x
    

class SwinTransformerBlockV2(SwinTransformerBlock):
    """Swin Transformer V2 Block.
    
    Args:
        channels: Number of input channels.
        num_heads: Number of attention heads.
        window_size: Window size.
        shift_size: Shift size for shifted window attention.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim. Default: ``4.0``.
        dropout: Dropout rate. Default: ``0.0``.
        attention_dropout: Attention dropout rate. Default: ``0.0``.
        stochastic_depth_prob: Stochastic depth rate. Default: ``0.0``.
        norm_layer: Normalization layer.  Default: :class:`nn.LayerNorm`.
        attn_layer: Attention layer. Default: :class:`ShiftedWindowAttentionV2`.
    """
    
    zoo = {}
    
    def __init__(
        self,
        channels             : int,
        num_heads            : int,
        window_size          : list[int],
        shift_size           : list[int],
        mlp_ratio            : float     = 4.0,
        dropout              : float     = 0.0,
        attention_dropout    : float     = 0.0,
        stochastic_depth_prob: float     = 0.0,
        norm_layer           : _callable = nn.LayerNorm,
        attn_layer           : _callable = nn.ShiftedWindowAttentionV2,
        *args, **kwargs
    ):
        super().__init__(
            channels              = channels,
            num_heads             = num_heads,
            window_size           = window_size,
            shift_size            = shift_size,
            mlp_ratio             = mlp_ratio,
            dropout               = dropout,
            attention_dropout     = attention_dropout,
            stochastic_depth_prob = stochastic_depth_prob,
            norm_layer            = norm_layer,
            attn_layer            = attn_layer,
            *args, **kwargs
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        # Here is the difference, we apply norm after the attention in V2.
        # In V1 we applied norm before the attention.
        x = x + self.stochastic_depth(self.norm1(self.attn(x)))
        x = x + self.stochastic_depth(self.norm2(self.mlp(x)))
        return x

# endregion


# region Model

class SwinTransformer(base.ImageClassificationModel, ABC):
    """Implements Swin Transformer from the `"Swin Transformer: Hierarchical
    Vision Transformer using Shifted Windows"
    <https://arxiv.org/pdf/2103.14030>`__ paper.
    
    Args:
        patch_size: Patch size.
        embed_dim: Patch embedding dimension.
        depths: Depth of each Swin Transformer layer.
        num_heads: Number of attention heads in different layers.
        window_size: Window size.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim. Default: ``4.0``.
        dropout: Dropout rate. Default: ``0.0``.
        attention_dropout: Attention dropout rate. Default: ``0.``0.
        stochastic_depth_prob: Stochastic depth rate. Default: ``0.1``.
        num_classes: Number of classes for classification head. Default: ``1000``.
        block: SwinTransformer Block. Default: ``None``.
        norm_layer: Normalization layer. Default: ``None``.
        downsample_layer: Downsample layer (patch merging). Default: :class:`PatchMerging`.
    
    See Also: :class:`base.ImageClassificationModel`
    """
    
    _scheme: list[Scheme] = [Scheme.SUPERVISED]
    _zoo   : dict = {}
    
    def __init__(
        self,
        patch_size           : list[int],
        embed_dim            : int,
        depths               : list[int],
        num_heads            : list[int],
        window_size          : list[int],
        mlp_ratio            : float     = 4.0,
        dropout              : float     = 0.0,
        attention_dropout    : float     = 0.0,
        stochastic_depth_prob: float     = 0.1,
        in_channels          : int       = 3,
        num_classes          : int       = 1000,
        norm_layer           : _callable = None,
        block                : _callable = None,
        downsample_layer     : _callable = nn.PatchMerging,
        weights              : Any       = None,
        *args, **kwargs,
    ):
        super().__init__(
            in_channels = in_channels,
            num_classes = num_classes,
            weights     = weights,
            *args, **kwargs
        )
        if block is None:
            block = SwinTransformerBlock
        if norm_layer is None:
            norm_layer = functools.partial(nn.LayerNorm, eps=1e-5)
        
        layers: list[nn.Module] = []
        # Split image into non-overlapping patches
        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels  = self.in_channels,
                    out_channels = embed_dim,
                    kernel_size  = (patch_size[0], patch_size[1]),
                    stride       = (patch_size[0], patch_size[1])
                ),
                nn.Permute([0, 2, 3, 1]),
                norm_layer(embed_dim),
            )
        )
        
        total_stage_blocks = sum(depths)
        stage_block_id     = 0
        # Build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage: list[nn.Module] = []
            dim = embed_dim * 2 ** i_stage
            for i_layer in range(depths[i_stage]):
                # Adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                stage.append(
                    block(
                        channels              = dim,
                        num_heads             = num_heads[i_stage],
                        window_size           = window_size,
                        shift_size            = [0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio             = mlp_ratio,
                        dropout               = dropout,
                        attention_dropout     = attention_dropout,
                        stochastic_depth_prob = sd_prob,
                        norm_layer            = norm_layer,
                    )
                )
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            # add patch merging layer
            if i_stage < (len(depths) - 1):
                layers.append(downsample_layer(dim, norm_layer))
        self.features = nn.Sequential(*layers)
        
        num_features = embed_dim * 2 ** (len(depths) - 1)
        self.norm    = norm_layer(num_features)
        self.permute = nn.Permute([0, 3, 1, 2])  # B H W C -> B C H W
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.head    = nn.Linear(num_features, self.num_classes)
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self._init_weights)
            
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    def forward(
        self,
        input    : torch.Tensor,
        augment  : _callable = None,
        profile  : bool      = False,
        out_index: int       = -1,
        *args, **kwargs
    ) -> torch.Tensor:
        x = input
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        y = self.head(x)
        return y
    

@MODELS.register(name="swin_t")
class Swin_T(SwinTransformer):
    """swin_tiny architecture from
    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/pdf/2103.14030>`__.
    
    See Also: :class:`SwinTransformer`
    """
    
    _zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/swin_t-704ceda3.pth",
            "path"       : "swin/swin_t/imagenet1k_v1/swin_t_imagenet1k_v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(self, args, **kwargs):
        super().__init__(
            name                  = "swin_t",
            patch_size            = [4, 4],
            embed_dim             = 96,
            depths                = [2, 2 , 6 , 2],
            num_heads             = [3, 6 , 12, 24],
            window_size           = [7, 7],
            stochastic_depth_prob = 0.2,
            *args, **kwargs
        )


@MODELS.register(name="swin_s")
class Swin_S(SwinTransformer):
    """swin_small architecture from
    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/pdf/2103.14030>`__.
    
    See Also: :class:`SwinTransformer`
    """
    
    _zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/swin_s-5e29d889.pth",
            "path"       : "swin/swin_s/imagenet1k_v1/swin_s_imagenet1k_v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(self, args, **kwargs):
        super().__init__(
            name                  = "swin_s",
            patch_size            = [4, 4],
            embed_dim             = 96,
            depths                = [2, 2 , 18, 2],
            num_heads             = [3, 6 , 12, 24],
            window_size           = [7, 7],
            stochastic_depth_prob = 0.3,
            *args, **kwargs
        )


@MODELS.register(name="swin_b")
class Swin_B(SwinTransformer):
    """swin_base architecture from
    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/pdf/2103.14030>`__.
    
    See Also: :class:`SwinTransformer`
    """
    
    _zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/swin_b-68c6b09e.pth",
            "path"       : "swin/swin_b/imagenet1k_v1/swin_b_imagenet1k_v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(self, args, **kwargs):
        super().__init__(
            name                  = "swin_b",
            patch_size            = [4, 4],
            embed_dim             = 128,
            depths                = [2, 2 , 18, 2],
            num_heads             = [4, 8 , 16, 32],
            window_size           = [7, 7],
            stochastic_depth_prob = 0.5,
            *args, **kwargs
        )
        

@MODELS.register(name="swin_v2_t")
class Swin_V2_T(SwinTransformer):
    """swin_v2_tiny architecture from
    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/pdf/2103.14030>`__.
    
    See Also: :class:`SwinTransformer`
    """
    
    _zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/swin_v2_t-b137f0e2.pth",
            "path"       : "swin/swin_v2_t/imagenet1k_v1/swin_v2_t_imagenet1k_v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(self, args, **kwargs):
        super().__init__(
            name                  = "swin_v2_t",
            patch_size            = [4, 4],
            embed_dim             = 96,
            depths                = [2, 2, 6 , 2],
            num_heads             = [3, 6, 12, 24],
            window_size           = [8, 8],
            stochastic_depth_prob = 0.2,
            block                 = SwinTransformerBlockV2,
            downsample_layer      = nn.PatchMergingV2,
            *args, **kwargs
        )
        

@MODELS.register(name="swin_v2_s")
class Swin_V2_S(SwinTransformer):
    """swin_v2_small architecture from
    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/pdf/2103.14030>`__.
    
    See Also: :class:`SwinTransformer`
    """
    
    _zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/swin_v2_s-637d8ceb.pth",
            "path"       : "swin/swin_v2_s/imagenet1k_v1/swin_v2_s_imagenet1k_v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(self, args, **kwargs):
        super().__init__(
            name                  = "swin_v2_s",
            patch_size            = [4, 4],
            embed_dim             = 96,
            depths                = [2, 2 , 18, 2],
            num_heads             = [3, 6 , 12, 24],
            window_size           = [8, 8],
            stochastic_depth_prob = 0.3,
            block                 = SwinTransformerBlockV2,
            downsample_layer      = nn.PatchMergingV2,
            *args, **kwargs
        )
        

@MODELS.register(name="swin_v2_b")
class Swin_V2_B(SwinTransformer):
    """swin_v2_base architecture from
    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/pdf/2103.14030>`__.
    
    See Also: :class:`SwinTransformer`
    """
    
    _zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/swin_v2_b-781e5279.pth",
            "path"       : "swin/swin_v2_b/imagenet1k_v1/swin_v2_b_imagenet1k_v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(self, args, **kwargs):
        super().__init__(
            name                  = "swin_v2_b",
            patch_size            = [4, 4],
            embed_dim             = 128,
            depths                = [2, 2, 18, 2],
            num_heads             = [4, 8, 16, 32],
            window_size           = [8, 8],
            stochastic_depth_prob = 0.5,
            block                 = SwinTransformerBlockV2,
            downsample_layer      = nn.PatchMergingV2,
            *args, **kwargs
        )
        
# endregion
