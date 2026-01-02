#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for Positional Encoding (PE) techniques.

This module implements various Positional Encoding (PE) techniques for
Implicit Neural Representation (INR) tasks.

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


# --- Layer ---
class PosEncodingNeRF(nn.Module):
    """An implementation of the NeRF Positional Encoding (PE)."""
    
    def __init__(
        self,
        in_features    : int,
        sidelength     : int  = 256,
        num_frequencies: int  = 10,
        fn_samples     : Any  = None,
        use_nyquist    : bool = True
    ):
        """Initializes the NeRF Positional Encoding (PE).
        
        Args:
            in_features (int): Size of each input sample.
            sidelength (int): Sidelength of the 2D input grid. Defaults to 256.
            num_frequencies (int): Number of frequency bands for positional
                encoding. Defaults to 10.
            fn_samples (Any): Number of samples for 1D input. Defaults to None.
            use_nyquist (bool): Whether to use Nyquist frequency to determine
                the number of frequencies. Defaults to True.
        """
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
        """Calculates the number of frequencies based on the Nyquist rate.
        
        Args:
            samples (int): Number of samples for 1D input.
        
        Returns:
            int: Number of frequencies based on the Nyquist rate.
        """
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward pass of the NeRF Positional Encoding (PE).
        
        Args:
            coords (torch.Tensor): Input tensor of shape (B, ..., in_features).
        
        Returns:
            torch.Tensor: Output tensor of shape (B, ..., out_features).
        """
        coords   = coords.view(coords.shape[0], -1, self.in_features)
        encoding = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c        = coords[..., j]
                sin      = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos      = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)
                encoding = torch.cat((encoding, sin, cos), axis=-1)
        return encoding.reshape(coords.shape[0], -1, self.out_features)


# --- MLP ---
class PosEncodingMLP(nn.Module):
    """An implementation of a Positional Encoding (PE) MLP.
    
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
        """Initializes the Positional Encoding (PE) MLP.
        
        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            hidden_dim (int): Number of hidden units in each hidden layer.
            hidden_layers (int): Number of hidden layers.
            num_frequencies (int): Number of frequency bands for positional
                encoding. Defaults to 10.
            bias (bool): If True, adds a learnable bias to the linear layers.
                Defaults to True.
        """
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
        """Forward pass of the Positional Encoding (PE) MLP.
        
        Args:
            coords (torch.Tensor): Input tensor of shape (B, ..., in_features).
            
        Returns:
            torch.Tensor: Output tensor of shape (B, ..., out_features).
        """
        return self.net(self.encoding(coords))
