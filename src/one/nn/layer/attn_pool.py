#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Attention + Pool Layers.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from one.core import ATTN_POOL_LAYERS
from one.core import Int2T
from one.core import to_2tuple
from one.nn.layer.embed import apply_rot_embed
from one.nn.layer.embed import rot
from one.nn.layer.embed import RotaryEmbedding
from one.nn.layer.weight_init import trunc_normal_

__all__ = [
    "apply_rot_embed_list",
    "AttentionPool2d",
    "RotAttentionPool2d",
    "AttentionPool",
    "RotAttentionPool",
]


# MARK: - Functional

def apply_rot_embed_list(input: list[Tensor], sin_emb, cos_emb):
    if isinstance(input, Tensor):
        input = [input]
    return [t * cos_emb + rot(t) * sin_emb for t in input]


# MARK: - Modules

@ATTN_POOL_LAYERS.register(name="attention_pool2d")
class AttentionPool2d(nn.Module):
    """Attention based 2D feature pooling w/ learned (absolute) pos embedding.
    This is a multi-head attention based replacement for (spatial) average
    pooling in NN architectures.

    It was based on impl in CLIP by OpenAI
    https://github.com/openai/CLIP/blob/3b473b0e682c091a9e53623eebc1ca1657385717/clip/model.py

    NOTE: This requires feature size upon construction and well prevent
    adaptive sizing of the network.
    """

    # MARK: Magic Functions

    def __init__(
        self,
        in_features : int,
        feat_size   : Int2T,
        out_features: Optional[int] = None,
        embed_dim   : Optional[int] = None,
        num_heads   : int           = 4,
        qkv_bias    : bool          = True,
    ):
        super().__init__()

        embed_dim    = embed_dim or in_features
        out_features = out_features or in_features
        
        if embed_dim % num_heads != 0:
            raise ValueError(f"`embed_dim` must be divisible by `num_heads`."
                             f"But got: {embed_dim} % {num_heads} != 0.")
        self.feat_size = to_2tuple(feat_size)
        self.qkv       = nn.Linear(in_features, embed_dim * 3, bias=qkv_bias)
        self.proj      = nn.Linear(embed_dim, out_features)
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        spatial_dim    = self.feat_size[0] * self.feat_size[1]
        self.pos_embed = nn.Parameter(torch.zeros(spatial_dim + 1, in_features))
        trunc_normal_(self.pos_embed, std=in_features ** -0.5)
        trunc_normal_(self.qkv.weight, std=in_features ** -0.5)
        nn.init.zeros_(self.qkv.bias)

    # MARK: Forward Pass

    def forward(self, x: Tensor) -> Tensor:
        b, _, h, w = x.shape
        n          = h * w
        if self.feat_size[0] != h:
            raise ValueError
        if self.feat_size[1] != w:
            raise ValueError
            
        x = x.reshape(b, -1, n).permute(0, 2, 1)
        x = torch.cat([x.mean(1, keepdim=True), x], dim=1)
        x = x + self.pos_embed.unsqueeze(0).to(x.dtype)

        x = self.qkv(x).reshape(b, n + 1, 3, self.num_heads,
                                self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = x[0], x[1], x[2]
        attn    = (q @ k.transpose(-2, -1)) * self.scale
        attn    = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(b, n + 1, -1)
        x = self.proj(x)
        return x[:, 0]
    
    
@ATTN_POOL_LAYERS.register(name="rot_attention_pool2d")
class RotAttentionPool2d(nn.Module):
    """Attention based 2D feature pooling w/ rotary (relative) pos embedding.
    This is a multi-head attention based replacement for (spatial) average
    pooling in NN architectures.

    Adapted from the AttentionPool2d in CLIP w/ rotary embedding instead of
    learned embed.
    https://github.com/openai/CLIP/blob/3b473b0e682c091a9e53623eebc1ca1657385717/clip/model.py

    NOTE: While this impl does not require a fixed feature size, performance at
    differeing resolutions from train varies widely and falls off
    dramatically. I'm not sure if there is a way around this... -RW
    """

    # MARK: Magic Functions

    def __init__(
        self,
        in_features : int,
        out_features: Optional[int] = None,
        embed_dim   : Optional[int] = None,
        num_heads   : int           = 4,
        qkv_bias    : bool          = True,
    ):
        super().__init__()
        embed_dim      = embed_dim    or in_features
        out_features   = out_features or in_features
        self.qkv       = nn.Linear(in_features, embed_dim * 3, bias=qkv_bias)
        self.proj      = nn.Linear(embed_dim, out_features)
        self.num_heads = num_heads
        
        if embed_dim % num_heads != 0:
            raise ValueError(f"`embed_dim` must be divisible by `num_heads`."
                             f"But got: {embed_dim} % {num_heads} != 0.")
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.pos_embed = RotaryEmbedding(self.head_dim)

        trunc_normal_(self.qkv.weight, std=in_features ** -0.5)
        nn.init.zeros_(self.qkv.bias)

    # MARK: Forward Pass

    def forward(self, x: Tensor) -> Tensor:
        b, _, h, w       = x.shape
        n                = h * w
        sin_emb, cos_emb = self.pos_embed.get_embed(x.shape[2:])

        x = x.reshape(b, -1, n).permute(0, 2, 1)
        x = torch.cat([x.mean(1, keepdim=True), x], dim=1)

        x = self.qkv(x).reshape(b, n + 1, 3, self.num_heads,
                                self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = x[0], x[1], x[2]

        qc, q = q[:, :, :1], q[:, :, 1:]
        q     = apply_rot_embed(q, sin_emb, cos_emb)
        q     = torch.cat([qc, q], dim=2)

        kc, k = k[:, :, :1], k[:, :, 1:]
        k     = apply_rot_embed(k, sin_emb, cos_emb)
        k     = torch.cat([kc, k], dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(b, n + 1, -1)
        x = self.proj(x)
        return x[:, 0]


# MARK: - Alias

AttentionPool    = AttentionPool2d
RotAttentionPool = RotAttentionPool2d


# MARK: - Register

ATTN_POOL_LAYERS.register(name="attention_pool",     module=AttentionPool)
ATTN_POOL_LAYERS.register(name="rot_attention_pool", module=RotAttentionPool)
