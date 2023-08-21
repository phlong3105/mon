#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements channel filters."""

from __future__ import annotations

__all__ = [
    "get_atmosphere_channel",
    "get_dark_channel",
]

import torch

from mon.nn import functional as F
from mon.vision import core


# region Dark Channel Prior

def get_dark_channel(image: torch.Tensor, size: int = 15) -> torch.Tensor:
    """Get the dark channel prior in a given image.

    References:
        `<https://github.com/liboyun/ZID/blob/master/utils/dcp.py>`__

    Args:
        image: An image in channel-first format.
        size: A window size. Default: 15.

    Returns:
        A dark channel prior.
    """
    if not image.ndim == 4:
        raise ValueError(
            f"img's number of dimensions must be == 4, but got {image.ndim}."
        )
    if core.is_channel_first_image(input=image):
        raise ValueError(f"img must be in channel-first format.")
    
    b, c, h, w  = image.shape
    p            = size // 2
    padded       = F.pad(input=image, pad=(p, p, p, p), mode="replicate")
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
    size : int   = 15,
    p    : float = 0.0001,
) -> torch.Tensor:
    """Get the atmosphere light in an RGB image.
    
    References:
        https://github.com/liboyun/ZID/blob/master/utils/dcp.py
        
    Args:
        image: An image in channel-first format.
        size: A window size. Default: 15.
        p: A percentage of pixels for estimating atmosphere light. Default:
            0.0001.
    
    Returns:
       An atmosphere light ([0, L-1]) for each channel.
    """
    if not image.ndim == 4:
        raise ValueError(
            f"img's number of dimensions must be == 4, but got {image.ndim}."
        )
    if core.is_channel_first_image(input=image):
        raise ValueError(f"img must be in channel-first format.")
    
    b, c, h, w = image.shape
    dark       = get_dark_channel(image=image, size=size)
    flat_i     = torch.reshape(input=image, shape=(b, 3, h * w))
    flat_dark  = torch.ravel(input=dark)
    # Find top h * w * p indexes
    search_idx = torch.argsort(input=(-flat_dark))[:int(h * w * p)]
    # Find the highest intensity for each channel
    atm = torch.index_select(input=flat_i, dim=-1, index=search_idx)
    atm = atm.max(dim=-1, keepdim=True)
    return atm[0]

# endregion
