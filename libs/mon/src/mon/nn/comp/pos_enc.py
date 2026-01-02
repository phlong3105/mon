#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Positional encoding (PE) layers.

This module implements various positional encoding (PE) layers used for
representing high-dimensional inputs.
"""

__all__ = [
    "PosEncodingFourier",
    "PosEncodingNeRF",
]

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class PosEncodingFourier(nn.Module):
    """Positional Encoding (PE) using Fourier features."""
    
    def __init__(self, in_features: int, B: float = 20.0):
        """Initialize a new instance.
        
        Args:
            in_features: Size of each input sample.
            B: Standard deviation of the Gaussian distribution used to sample
                the projection matrix. If set to None, no projection is applied.
                Defaults to 20.0.
        """
        super().__init__()
        self.in_features  = in_features
        self.out_features = in_features * 2
        if B is None:
            self.B = None
        else:
            self.register_buffer("B", torch.randn((in_features, 2)) * B)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            input: Input tensor with dimensions (..., in_features) and values
                ranging from 0.0 to 1.0.
            
        Returns:
            Output tensor with dimensions (..., out_features) and values ranging
            from 0.0 to 1.0.
        """
        if self.B is None:
            return input
        else:
            proj     = (2. * np.pi * input) @ self.B.T
            encoding = torch.cat([torch.sin(proj), torch.cos(proj)], axis=-1)
            return encoding


class PosEncodingNeRF(nn.Module):
    """NeRF Positional Encoding (PE).
    
    References:
        - Code: https://github.com/liuzhen0212/FINER/blob/main/models.py
    """
    
    def __init__(
        self,
        in_features    : int,
        sidelength     : int  = 256,
        num_frequencies: int  = 10,
        fn_samples     : Any  = None,
        use_nyquist    : bool = True
    ):
        """Initialize a new instance.
        
        Args:
            in_features: Size of each input sample.
            sidelength: Sidelength of the 2D input grid. Defaults to 256.
            num_frequencies: Number of frequency bands for positional encoding.
                Defaults to 10.
            fn_samples: Number of samples for 1D input. Defaults to None.
            use_nyquist: Whether to use Nyquist frequency to determine the
                number of frequencies. Defaults to True.
        """
        super().__init__()
        
        if in_features == 3:
            self.num_frequencies = num_frequencies
        elif in_features == 2:
            assert sidelength is not None
            if isinstance(sidelength, int):
                sidelength = (sidelength, sidelength)
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
        elif in_features == 1:
            assert fn_samples is not None
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)
        
        self.in_features  = in_features
        self.out_features = in_features + 2 * in_features * self.num_frequencies

    def get_num_frequencies_nyquist(self, samples: int) -> int:
        """Calculate the number of frequencies based on the Nyquist rate.
        
        Args:
            samples: Number of samples for 1D input.
        
        Returns:
            Number of frequencies based on the Nyquist rate.
        """
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            input: Input tensor with dimensions (..., in_features) and values
                ranging from 0.0 to 1.0.
        
        Returns:
            Output tensor with dimensions (..., out_features) and values ranging
            from 0.0 to 1.0.
        """
        input    = input.view(input.shape[0], -1, self.in_features)
        encoding = input
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c        = input[..., j]
                sin      = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos      = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)
                encoding = torch.cat((encoding, sin, cos), axis=-1)
        return encoding.reshape(input.shape[0], -1, self.out_features)
