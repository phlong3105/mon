#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Positional encoding (PE) layers.

This module provides various positional encoding (PE) layers used for
representing high-dimensional inputs.
"""

from __future__ import annotations

__all__ = [
    "PosEncodingFourier",
    "PosEncodingNeRF",
]

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn


# ==============================================================================
# region LAYERS
# ==============================================================================

class PosEncodingFourier(nn.Module):
    """Positional Encoding (PE) using Fourier features.

    Apply Fourier feature mapping to the input coordinates.

    Attributes:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        B: Projection matrix for Fourier features.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, mapping_size: int, B: float = 20.0):
        """Initialize a new instance.

        Args:
            mapping_size: Size of Fourier feature mapping.
            B: Standard deviation of the Gaussian distribution used to sample
                the projection matrix. If set to None, no projection is applied.
                Defaults to 20.0.
        """
        super().__init__()
        self.in_features  = mapping_size // 2
        self.out_features = mapping_size

        if B is None:
            self.B = None
        else:
            self.register_buffer("B", torch.randn((self.in_features, 2)) * B)

    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the input through the layer.

        Args:
            x: Input tensor of shape (..., in_features) and values ranging
                from -1.0 to 1.0.

        Returns:
            Output tensor of shape (..., out_features) and values ranging
            from -1.0 to 1.0.
        """
        if self.B is None:
            return x
        else:
            proj     = (2.0 * math.pi * x) @ self.B.T
            encoding = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
            return encoding


class PosEncodingNeRF(nn.Module):
    """NeRF Positional Encoding (PE).

    Apply positional encoding as described in the NeRF paper.

    References:
        - Code: https://github.com/liuzhen0212/FINER/blob/main/models.py

    Attributes:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        num_frequencies: Number of frequency bands for positional encoding.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_features    : int,
        sidelength     : int | tuple[int, int] = 256,
        num_frequencies: int                   = 10,
        fn_samples     : Any                   = None,
        use_nyquist    : bool                  = True
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
                self.num_frequencies = self.get_num_frequencies_nyquist(
                    min(sidelength[0], sidelength[1])
                )
        elif in_features == 1:
            assert fn_samples is not None
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)
        else:
            self.num_frequencies = num_frequencies

        self.in_features  = in_features
        self.out_features = in_features + 2 * in_features * self.num_frequencies

        # Pre-compute frequency bands to avoid recomputing in forward pass
        freq_bands = 2.0 ** torch.linspace(0.0, self.num_frequencies - 1, self.num_frequencies)
        self.register_buffer("freq_bands", freq_bands * np.pi)

    def get_num_frequencies_nyquist(self, samples: int) -> int:
        """Calculate the number of frequencies based on the Nyquist rate.

        Args:
            samples: Number of samples for 1D input.

        Returns:
            Number of frequencies based on the Nyquist rate.
        """
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the input through the layer.

        Args:
            x: Input tensor of shape (..., in_features) and values ranging
                from 0.0 to 1.0.

        Returns:
            Output tensor of shape (..., out_features) and values ranging
            from 0.0 to 1.0.
        """
        # x shape: [B, ..., C]
        # freq_bands shape: [num_frequencies]

        # Reshape for broadcasting: [B, ..., C, 1] * [1, ..., 1, num_frequencies]
        # Result: [B, ..., C, num_frequencies]
        spectrum = x.unsqueeze(-1) * self.freq_bands

        sin_enc  = torch.sin(spectrum)
        cos_enc  = torch.cos(spectrum)

        # Flatten the last two dimensions: [B, ..., C * num_frequencies]
        sin_enc  = sin_enc.view(*x.shape[:-1], -1)
        cos_enc  = cos_enc.view(*x.shape[:-1], -1)

        # Concatenate original input with encodings
        return torch.cat([x, sin_enc, cos_enc], dim=-1)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
