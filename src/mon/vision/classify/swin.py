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
from typing import Callable, Any

import torch
from torchvision import ops

from mon.globals import MODELS
from mon.vision import core, nn
from mon.vision.classify import base
from mon.vision.nn import functional as F

console      = core.console
math         = core.math
_current_dir = core.Path(__file__).absolute().parent


# region Module

def _get_relative_position_bias(
    relative_position_bias_table: torch.Tensor,
    relative_position_index     : torch.Tensor,
    window_size                 : list[int],
) -> torch.Tensor:
    n = window_size[0] * window_size[1]
    relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
    relative_position_bias = relative_position_bias.view(n, n, -1)
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
    return relative_position_bias


def shifted_window_attention(
    input                 : torch.Tensor,
    qkv_weight            : torch.Tensor,
    proj_weight           : torch.Tensor,
    relative_position_bias: torch.Tensor,
    window_size           : list[int],
    num_heads             : int,
    shift_size            : list[int],
    attention_dropout     : float               = 0.0,
    dropout               : float               = 0.0,
    qkv_bias              : torch.Tensor | None = None,
    proj_bias             : torch.Tensor | None = None,
    logit_scale           : torch.Tensor | None = None,
    training              : bool                = True,
) -> torch.Tensor:
    """Window-based multi-head self-attention (W-MSA) module with relative
    position bias. It supports both shifted and non-shifted windows.
    
    Args:
        input: An input of shape :math:`[N, C, H, W]`.
        qkv_weight: The weight tensor of query, key, value of shape
            :math:`[in_dim, out_dim]`.
        proj_weight: The weight tensor of projection of shape
            :math:`[in_dim, out_dim]`.
        relative_position_bias: The learned relative position bias added to
            attention.
        window_size: Window size.
        num_heads: Number of attention heads.
        shift_size: Shift size for shifted window attention.
        attention_dropout: Dropout ratio of attention weight. Default: ``0.0``.
        dropout: Dropout ratio of output. Default: ``0.0``.
        qkv_bias: The bias tensor of query, key, value. Default: ``None``.
        proj_bias: The bias tensor of projection. Default: ``None``.
        logit_scale: Logit scale of cosine attention for Swin Transformer V2.
            Default: ``None``.
        training: Training flag used by the dropout parameters. Default: ``True``.
    
    Returns:
        The output tensor after shifted window attention of shape
            :math:`[N, C, H, W]`.
    """
    b, h, w, c = input.shape
    # Pad feature maps to multiples of window size
    pad_r = (window_size[1] - w % window_size[1]) % window_size[1]
    pad_b = (window_size[0] - h % window_size[0]) % window_size[0]
    x     = F.pad(input, (0, 0, 0, pad_r, 0, pad_b))
    _, pad_h, pad_w, _ = x.shape

    shift_size = shift_size.copy()
    # If window size is larger than feature size, there is no need to shift window
    if window_size[0] >= pad_h:
        shift_size[0] = 0
    if window_size[1] >= pad_w:
        shift_size[1] = 0

    # Cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))

    # partition windows
    num_windows = (pad_h // window_size[0]) * (pad_w // window_size[1])
    x = x.view(b, pad_h // window_size[0], window_size[0], pad_w // window_size[1], window_size[1], c)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(b * num_windows, window_size[0] * window_size[1], c)  # B*nW, Ws*Ws, C
    
    # Multi-head attention
    if logit_scale is not None and qkv_bias is not None:
        qkv_bias = qkv_bias.clone()
        length   = qkv_bias.numel() // 3
        qkv_bias[length : 2 * length].zero_()
    qkv     = F.linear(x, qkv_weight, qkv_bias)
    qkv     = qkv.reshape(x.size(0), x.size(1), 3, num_heads, c // num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    if logit_scale is not None:
        # Cosine attention
        attn        = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(logit_scale, max=math.log(100.0)).exp()
        attn        = attn * logit_scale
    else:
        q    = q * (c // num_heads) ** -0.5
        attn = q.matmul(k.transpose(-2, -1))
    # Add relative position bias
    attn = attn + relative_position_bias

    if sum(shift_size) > 0:
        # Generate attention mask
        attn_mask = x.new_zeros((pad_h, pad_w))
        h_slices  = ((0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], None))
        w_slices  = ((0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], None))
        count     = 0
        for h in h_slices:
            for w in w_slices:
                attn_mask[h[0] : h[1], w[0] : w[1]] = count
                count += 1
        attn_mask = attn_mask.view(pad_h // window_size[0], window_size[0], pad_w // window_size[1], window_size[1])
        attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, window_size[0] * window_size[1])
        attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        attn      = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
        attn      = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn      = attn.view(-1, num_heads, x.size(1), x.size(1))

    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attention_dropout, training=training)

    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), c)
    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=dropout, training=training)

    # Reverse windows
    x = x.view(b, pad_h // window_size[0], pad_w // window_size[1], window_size[0], window_size[1], c)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(b, pad_h, pad_w, c)

    # Reverse cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

    # Unpad features
    x = x[:, :h, :w, :].contiguous()
    return x


class ShiftedWindowAttention(nn.Module):
    """See Also :func:`shifted_window_attention`."""

    def __init__(
        self,
        dim              : int,
        window_size      : list[int],
        shift_size       : list[int],
        num_heads        : int,
        qkv_bias         : bool  = True,
        proj_bias        : bool  = True,
        attention_dropout: float = 0.0,
        dropout          : float = 0.0,
        *args, **kwargs
    ):
        super().__init__()
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError(f":param:`window_size` and :param:`shift_size` must be of length ``2``.")
        self.window_size       = window_size
        self.shift_size        = shift_size
        self.num_heads         = num_heads
        self.attention_dropout = attention_dropout
        self.dropout           = dropout

        self.qkv  = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim,     bias=proj_bias)

        self.define_relative_position_bias_table()
        self.define_relative_position_index()

    def define_relative_position_bias_table(self):
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        torch.nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def define_relative_position_index(self):
        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords   = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten  = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).flatten()  # Wh*Ww*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def get_relative_position_bias(self) -> torch.Tensor:
        return _get_relative_position_bias(
            self.relative_position_bias_table,
            self.relative_position_index,
            self.window_size  # type: ignore[arg-type]
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        
        Args:
            input: Tensor of shape :math:`[B, H, W, C]`.
       
        Returns:
            Tensor of shape :math:`[B, H, W, C]`.
        """
        x = input
        relative_position_bias = self.get_relative_position_bias()
        return shifted_window_attention(
            input                  = x,
            qkv_weight             = self.qkv.weight,
            proj_weight            = self.proj.weight,
            relative_position_bias = relative_position_bias,
            window_size            = self.window_size,
            num_heads              = self.num_heads,
            shift_size             = self.shift_size,
            attention_dropout      = self.attention_dropout,
            dropout                = self.dropout,
            qkv_bias               = self.qkv.bias,
            proj_bias              = self.proj.bias,
            training               = self.training,
        )


class ShiftedWindowAttentionV2(ShiftedWindowAttention):
    """See Also: :class:`ShiftedWindowAttention`."""

    def __init__(
        self,
        dim              : int,
        window_size      : list[int],
        shift_size       : list[int],
        num_heads        : int,
        qkv_bias         : bool  = True,
        proj_bias        : bool  = True,
        attention_dropout: float = 0.0,
        dropout          : float = 0.0,
        *args, **kwargs
    ):
        super().__init__(
            dim               = dim,
            window_size       = window_size,
            shift_size        = shift_size,
            num_heads         = num_heads,
            qkv_bias          = qkv_bias,
            proj_bias         = proj_bias,
            attention_dropout = attention_dropout,
            dropout           = dropout,
        )

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        # MLP to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False)
        )
        if qkv_bias:
            length = self.qkv.bias.numel() // 3
            self.qkv.bias[length : 2 * length].data.zero_()

    def define_relative_position_bias_table(self):
        # Get relative_coords_table
        relative_coords_h     = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w     = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w], indexing="ij"))
        relative_coords_table = relative_coords_table.permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2

        relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
        relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1

        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = (torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / 3.0)
        self.register_buffer("relative_coords_table", relative_coords_table)

    def get_relative_position_bias(self) -> torch.Tensor:
        relative_position_bias = _get_relative_position_bias(
            self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads),
            self.relative_position_index,  # type: ignore[arg-type]
            self.window_size,
        )
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        return relative_position_bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        
        Args:
            input: Tensor of shape :math:`[B, H, W, C]`.
       
        Returns:
            Tensor of shape :math:`[B, H, W, C]`.
        """
        x = input
        relative_position_bias = self.get_relative_position_bias()
        return shifted_window_attention(
            input                  = x,
            qkv_weight             = self.qkv.weight,
            proj_weight            = self.proj.weight,
            relative_position_bias = relative_position_bias,
            window_size            = self.window_size,
            num_heads              = self.num_heads,
            shift_size             = self.shift_size,
            attention_dropout      = self.attention_dropout,
            dropout                = self.dropout,
            qkv_bias               = self.qkv.bias,
            proj_bias              = self.proj.bias,
            training               = self.training,
        )


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block.
    
    Args:
        dim: Number of input channels.
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
        dim                  : int,
        num_heads            : int,
        window_size          : list[int],
        shift_size           : list[int],
        mlp_ratio            : float        = 4.0,
        dropout              : float        = 0.0,
        attention_dropout    : float        = 0.0,
        stochastic_depth_prob: float        = 0.0,
        norm_layer           : Callable[..., nn.Module] = nn.LayerNorm,
        attn_layer           : Callable[..., nn.Module] = ShiftedWindowAttention,
        *args, **kwargs
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn  = attn_layer(
            dim               = dim,
            window_size       = window_size,
            shift_size        = shift_size,
            num_heads         = num_heads,
            attention_dropout = attention_dropout,
            dropout           = dropout,
        )
        self.stochastic_depth = ops.StochasticDepth(stochastic_depth_prob, "row")
        self.norm2 = norm_layer(dim)
        self.mlp   = nn.MLP(
            in_channels      = dim,
            hidden_channels  = [int(dim * mlp_ratio), dim],
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
        dim: Number of input channels.
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
        dim                  : int,
        num_heads            : int,
        window_size          : list[int],
        shift_size           : list[int],
        mlp_ratio            : float        = 4.0,
        dropout              : float        = 0.0,
        attention_dropout    : float        = 0.0,
        stochastic_depth_prob: float        = 0.0,
        norm_layer           : Callable[..., nn.Module] = nn.LayerNorm,
        attn_layer           : Callable[..., nn.Module] = ShiftedWindowAttentionV2,
        *args, **kwargs
    ):
        super().__init__(
            dim                   = dim,
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
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {}
    
    def __init__(
        self,
        patch_size           : list[int],
        embed_dim            : int,
        depths               : list[int],
        num_heads            : list[int],
        window_size          : list[int],
        mlp_ratio            : float = 4.0,
        dropout              : float = 0.0,
        attention_dropout    : float = 0.0,
        stochastic_depth_prob: float = 0.1,
        num_classes          : int   = 1000,
        norm_layer           : Callable[..., nn.Module] | None = None,
        block                : Callable[..., nn.Module] | None = None,
        downsample_layer     : Callable[..., nn.Module]        = nn.PatchMerging,
        weights              : Any   = None,
        name                 : str   = "swin",
        *args, **kwargs,
    ):
        super().__init__(
            num_classes = num_classes,
            weights     = weights,
            name        = name,
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
                    in_channels  = 3,
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
                        dim                   = dim,
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
            self.apply(self.init_weights)
            
    def init_weights(self, m: nn.Module):
        """Initialize model's weights."""
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    def forward_once(
        self,
        input    : torch.Tensor,
        profile  : bool = False,
        out_index: int  = -1,
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
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/swin_t-704ceda3.pth",
            "path"       : "swin_t-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "swin",
        variant: str = "swin_t",
        *args, **kwargs
    ):
        super().__init__(
            patch_size            = [4, 4],
            embed_dim             = 96,
            depths                = [2, 2 , 6 , 2],
            num_heads             = [3, 6 , 12, 24],
            window_size           = [7, 7],
            stochastic_depth_prob = 0.2,
            name                  = name,
            variant               = variant,
            *args, **kwargs
        )


@MODELS.register(name="swin_s")
class Swin_S(SwinTransformer):
    """swin_small architecture from
    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/pdf/2103.14030>`__.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/swin_s-5e29d889.pth",
            "path"       : "swin_s-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "swin",
        variant: str = "swin_s",
        *args, **kwargs
    ):
        super().__init__(
            patch_size            = [4, 4],
            embed_dim             = 96,
            depths                = [2, 2 , 18, 2],
            num_heads             = [3, 6 , 12, 24],
            window_size           = [7, 7],
            stochastic_depth_prob = 0.3,
            name                  = name,
            variant               = variant,
            *args, **kwargs
        )


@MODELS.register(name="swin_b")
class Swin_B(SwinTransformer):
    """swin_base architecture from
    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/pdf/2103.14030>`__.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/swin_b-68c6b09e.pth",
            "path"       : "swin_b-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "swin",
        variant: str = "swin_b",
        *args, **kwargs
    ):
        super().__init__(
            patch_size            = [4, 4],
            embed_dim             = 128,
            depths                = [2, 2 , 18, 2],
            num_heads             = [4, 8 , 16, 32],
            window_size           = [7, 7],
            stochastic_depth_prob = 0.5,
            name                  = name,
            variant               = variant,
            *args, **kwargs
        )
        

@MODELS.register(name="swin_v2_t")
class Swin_V2_T(SwinTransformer):
    """swin_v2_tiny architecture from
    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/pdf/2103.14030>`__.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/swin_v2_t-b137f0e2.pth",
            "path"       : "swin_v2_t-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "swin",
        variant: str = "swin_v2_t",
        *args, **kwargs
    ):
        super().__init__(
            patch_size            = [4, 4],
            embed_dim             = 96,
            depths                = [2, 2, 6 , 2],
            num_heads             = [3, 6, 12, 24],
            window_size           = [8, 8],
            stochastic_depth_prob = 0.2,
            block                 = SwinTransformerBlockV2,
            downsample_layer      = nn.PatchMergingV2,
            name                  = name,
            variant               = variant,
            *args, **kwargs
        )
        

@MODELS.register(name="swin_v2_s")
class Swin_V2_S(SwinTransformer):
    """swin_v2_small architecture from
    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/pdf/2103.14030>`__.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/swin_v2_s-637d8ceb.pth",
            "path"       : "swin_v2_s-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "swin",
        variant: str = "swin_v2_s",
        *args, **kwargs
    ):
        super().__init__(
            patch_size            = [4, 4],
            embed_dim             = 96,
            depths                = [2, 2 , 18, 2],
            num_heads             = [3, 6 , 12, 24],
            window_size           = [8, 8],
            stochastic_depth_prob = 0.3,
            block                 = SwinTransformerBlockV2,
            downsample_layer      = nn.PatchMergingV2,
            name                  = name,
            variant               = variant,
            *args, **kwargs
        )
        

@MODELS.register(name="swin_v2_b")
class Swin_V2_B(SwinTransformer):
    """swin_v2_base architecture from
    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/pdf/2103.14030>`__.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/swin_v2_b-781e5279.pth",
            "path"       : "swin_v2_b-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "swin",
        variant: str = "swin_v2_b",
        *args, **kwargs
    ):
        super().__init__(
            patch_size            = [4, 4],
            embed_dim             = 128,
            depths                = [2, 2, 18, 2],
            num_heads             = [4, 8, 16, 32],
            window_size           = [8, 8],
            stochastic_depth_prob = 0.5,
            block                 = SwinTransformerBlockV2,
            downsample_layer      = nn.PatchMergingV2,
            name                  = name,
            variant               = variant,
            *args, **kwargs
        )
        
# endregion
