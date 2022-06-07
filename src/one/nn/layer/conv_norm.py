#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convolution Layers.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = [
	"fuse_conv_and_bn",
]


# MARK: - Functional

def fuse_conv_and_bn(conv: nn.Conv2d, bn):
    """https://tehnokv.com/posts/fusing-batchnorm-and-conv/."""
    
    with torch.no_grad():
        # Init
        fusedconv = nn.Conv2d(
	        in_channels  = conv.in_channels,
	        out_channels = conv.out_channels,
	        kernel_size  = conv.kernel_size,
	        stride       = conv.stride,
	        padding      = conv.padding,
	        bias         = True
        ).to(conv.weight.device)

        # prepare filtering
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn   = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

        # prepare spatial bias
        b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
        b_bn   = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        return fusedconv
