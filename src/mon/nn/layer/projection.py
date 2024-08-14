#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements projection layers.

A projection layer in neural networks typically refers to the process of
transforming the input data into a different space using a projection matrix.
This transformation can help in reducing the dimensionality of the input data
or extracting important features for further processing.
"""

from __future__ import annotations

__all__ = [
	"ConvProjection",
	"LinearProjection",
]

import math

import torch
from einops import rearrange
from torch import nn

from mon.core import _size_2_t
from mon.nn.layer import conv, linear


# region Embedding Q, K, V

class ConvProjection(nn.Module):
    
    def __init__(
        self,
        channels     : int,
        heads        : int       = 8,
        head_channels: int       = 64,
        kernel_size  : _size_2_t = 3,
        q_stride     : int       = 1,
        k_stride     : int       = 1,
        v_stride     : int       = 1,
        dropout      : float     = 0.0,
        last_stage   : bool      = False,
        bias         : bool      = True,
		*args, **kwargs
    ):
        super().__init__()
        inner_channels = head_channels * heads
        self.heads 	   = heads
        pad      	   = (kernel_size - q_stride) // 2
        self.to_q  	   = conv.DSConvAct2d(channels, inner_channels, kernel_size, q_stride, pad, bias)
        self.to_k      = conv.DSConvAct2d(channels, inner_channels, kernel_size, k_stride, pad, bias)
        self.to_v      = conv.DSConvAct2d(channels, inner_channels, kernel_size, v_stride, pad, bias)

    def forward(
        self,
        input  : torch.Tensor,
        attn_kv: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = input
        b, n, c, h = *x.shape, self.heads
        l = int(math.sqrt(n))
        w = int(math.sqrt(n))

        attn_kv = x if attn_kv is None else attn_kv
        x       = rearrange(x, 'b (l w) c -> b c l w', l=l, w=w)
        attn_kv = rearrange(attn_kv, 'b (l w) c -> b c l w', l=l, w=w)
        # print(attn_kv)
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)

        k = self.to_k(attn_kv)
        v = self.to_v(attn_kv)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)
        return q, k, v

    def flops(self, q_L, kv_L=None):
        kv_L = kv_L or q_L
        flops = 0
        flops += self.to_q.flops(q_L)
        flops += self.to_k.flops(kv_L)
        flops += self.to_v.flops(kv_L)
        return flops


class LinearProjection(nn.Module):
    
    def __init__(
        self,
        channels     : int,
        heads        : int   = 8,
        head_channels: int   = 64,
        dropout      : float = 0.0,
        bias         : bool  = True,
		*args, **kwargs
    ):
        super().__init__()
        inner_channels      = head_channels * heads
        self.heads          = heads
        self.to_q           = linear.Linear(channels, inner_channels    , bias=bias)
        self.to_kv          = linear.Linear(channels, inner_channels * 2, bias=bias)
        self.dim            = channels
        self.inner_channels = inner_channels

    def forward(
        self,
        input  : torch.Tensor,
        attn_kv: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = input
        b_, n, c = x.shape
        if attn_kv:
            attn_kv = attn_kv.unsqueeze(0).repeat(b_, 1, 1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q    = self.to_q(x).reshape(b_, n, 1, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        kv   = self.to_kv(attn_kv).reshape(b_, N_kv, 2, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        q    = q[0]
        k, v = kv[0], kv[1]
        return q, k, v

    def flops(self, q_L, kv_L = None):
        kv_L  = kv_L or q_L
        flops = q_L * self.dim * self.inner_channels + kv_L * self.dim * self.inner_channels * 2
        return flops

# endregion
