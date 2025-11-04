#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Positional Encoding (PE) MLP.

References:
    - Code: https://github.com/liuzhen0212/FINER/blob/main/models.py
"""

__all__ = [
    "PosEncodingNeRF",
    "PosEncodingMLP",
]

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn


# ----- Layer -----
class PosEncodingNeRF(nn.Module):
    """Module to add positional encoding as in NeRF [Mildenhall et al. 2020]."""
    
    def __init__(
        self,
        in_features    : int,
        sidelength     : int  = 256,
        num_frequencies: int  = 10,
        fn_samples     : Any  = None,
        use_nyquist    : bool = True
    ):
        super().__init__()
        self.in_features = in_features
        
        if self.in_features == 3:
            self.num_frequencies = num_frequencies
        elif self.in_features == 2:
            assert sidelength is not None
            if isinstance(sidelength, int):
                sidelength = (sidelength, sidelength)
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
        elif self.in_features == 1:
            assert fn_samples is not None
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)

        self.out_features = in_features + 2 * in_features * self.num_frequencies

    def get_num_frequencies_nyquist(self, samples: int) -> int:
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords   = coords.view(coords.shape[0], -1, self.in_features)
        encoding = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c        = coords[..., j]
                sin      = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos      = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)
                encoding = torch.cat((encoding, sin, cos), axis=-1)
        return encoding.reshape(coords.shape[0], -1, self.out_features)


# ----- MLP -----
class PosEncodingMLP(nn.Module):
    """Implements the Positional Encoding (PE) MLP.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        hidden_dim: Hidden channel dimensions.
        hidden_layers: Number of hidden layers.
        N_freqs: Number of frequency bands for positional encoding. Default: ``10``.
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``.
    
    References:
        - Code: https://github.com/liuzhen0212/FINER/blob/main/models.py
    """
    
    def __init__(
        self,
        in_features    : int,
        out_features   : int,
        hidden_dim     : int,
        hidden_layers  : int,
        num_frequencies: int  = 10,
        bias           : bool = True,
    ):
        super().__init__()
        self.encoding = PosEncodingNeRF(in_features=in_features, num_frequencies=num_frequencies)
        
        # First layer
        self.net = []
        self.net.append(nn.Linear(self.encoding.out_features, hidden_dim, bias=bias))
        self.net.append(nn.ReLU(True))
        # Hidden layers
        for i in range(hidden_layers):
            self.net.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            self.net.append(nn.ReLU(True))
        # Final layer
        final_linear = nn.Linear(hidden_dim, out_features, bias=bias)
        self.net.append(final_linear)
        
        self.net = nn.Sequential(*self.net)
        
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(self.encoding(coords))
