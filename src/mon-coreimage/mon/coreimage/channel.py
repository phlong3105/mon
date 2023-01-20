#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements channel manipulation functions. """

from __future__ import annotations

__all__ = [
    "get_atmosphere_channel", "get_dark_channel",
]

import torch
from torch.nn import functional


# region Dark Channel Prior

def get_dark_channel(image: torch.Tensor, size: int = 15) -> torch.Tensor:
    """Gets the dark channel prior in a :param:`image` data.

    References:
        https://github.com/liboyun/ZID/blob/master/utils/dcp.py

    Args:
        image: Image of shape [B, 3, H, W] to be transformed.
        size: Window size. Defaults to 15.

    Returns:
        Dark channel prior of shape [B, 1, H, W].
    """
    assert isinstance(image, torch.Tensor)
    assert image.ndim == 4
    
    b, c, h, w   = image.shape
    p            = size // 2
    padded       = functional.pad(input=image, pad=(p, p, p, p), mode="replicate")
    dark_channel = torch.zeros([b, 1, h, w])
    
    for k in range(b):
        for i in range(h):
            for j in range(w):
                dark_channel[k, 0, i, j] = torch.min(
                    padded[k, :, i:(i + size), j:(j + size)]
                )  # CVPR09, eq.5
    return dark_channel


def get_atmosphere_channel(
    image: torch.Tensor,
    p    : float = 0.0001,
    size : int   = 15
) -> torch.Tensor:
    """Gets the atmosphere light in an RGB image.
    
    References:
        https://github.com/liboyun/ZID/blob/master/utils/dcp.py
        
    Args:
        image: Image of shape [..., 3, H, W] to be transformed,
            where ... means it can have arbitrary several leading
            dimensions.
        p: percentage of pixels for estimating atmosphere light.
            Defaults to 0.0001.
        size: window size. Defaults to 15.
        
    Returns:
       Atmosphere light ([0, L-1]) for each channel.
    """
    assert isinstance(image, torch.Tensor)
    assert image.ndim == 4
    
    b, c, h, w = image.shape
    dark       = get_dark_channel(image=image, size=size)
    flat_i     = torch.reshape(input=image, shape=(b, 3, h * w))
    flat_dark  = torch.ravel(input=dark)
    # Find top h * w * p indexes
    search_idx = torch.argsort(input=(-flat_dark))[:int(h * w * p)]
    # Find the highest intensity for each channel
    atm        = torch.index_select(input=flat_i, dim=-1, index=search_idx)
    atm        = atm.max(dim=-1, keepdim=True)
    return atm[0]

# endregion
