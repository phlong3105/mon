#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements kernels for filters.
"""

from __future__ import annotations

__all__ = [

]

from typing import Any

import mon.core
from mon import nn
from mon.nn import _size_2_t

console = mon.foundation.console


# region Binary Kernel

def get_binary_kernel2d(
    window_size: _size_2_t,
    *,
    device     : Any = None,
    dtype      : Any = None,
) -> torch.Tensor:
    """Create a binary kernel to extract the patches.

    If the window size is HxW will create a (H*W)x1xHxW kernel.
    """
    ky, kx            = nn.to_2d_kernel_size(window_size)
    window_range      = kx * ky
    kernel            = zeros((window_range, window_range), device=device, dtype=dtype)
    idx               = torch.arange(window_range, device=device)
    kernel[idx, idx] += 1.0
    return kernel.view(window_range, 1, ky, kx)

# endregion


# region Box Kernel

def get_box_kernel1d(
    kernel_size: int,
    *,
    device     : Any = None,
    dtype      : Any = None,
) -> torch.Tensor:
    r"""Utility function that returns a 1-D box filter.

    Args:
        kernel_size: The size of the kernel.
        device: The desired device of returned tensor.
        dtype: The desired data type of returned tensor.
    
    Returns:
        A tensor with shape :math:`(1, \text{kernel\_size})`, filled with the
        value :math:`\frac{1}{\text{kernel\_size}}`.
    """
    scale = tensor(1.0 / kernel_size, device=device, dtype=dtype)
    return scale.expand(1, kernel_size)

def get_box_kernel2d(
    kernel_size: _size_2_t,
    *,
    device     : Any = None,
    dtype      : Any = None,
) -> torch.Tensor:
    r"""Utility function that returns a 2-D box filter.
    
    Args:
        kernel_size: the size of the kernel.
        device: the desired device of returned tensor.
        dtype: the desired data type of returned tensor.
    Returns:
        A tensor with shape :math:`(1, \text{kernel\_size}[0], \text{kernel\_size}[1])`,
        filled with the value :math:`\frac{1}{\text{kernel\_size}[0] \times \text{kernel\_size}[1]}`.
    """
    ky, kx = nn.to_2d_kernel_size(kernel_size=kernel_size)
    scale  = tensor(1.0 / (kx * ky), device=device, dtype=dtype)
    return scale.expand(1, ky, kx)

# endregion


# region Pascal Kernel

def get_pascal_kernel1d(
    kernel_size: int,
    norm       : bool = False,
    *,
    device     : Any  = None,
    dtype      : Any  = None,
) -> torch.Tensor:
    """Generate Yang Hui triangle (Pascal's triangle) by a given number.

    Args:
        kernel_size: Height and width of the kernel.
        norm: If to normalize the kernel or not. Default: False.
        device: Tensor device desired to create the kernel.
        dtype: Tensor dtype desired to create the kernel.

    Returns:
        Kernel shaped as :math:`(kernel_size,)`

    Examples:
    >>> get_pascal_kernel1d(1)
    tensor([1.])
    >>> get_pascal_kernel1d(2)
    tensor([1., 1.])
    >>> get_pascal_kernel1d(3)
    tensor([1., 2., 1.])
    >>> get_pascal_kernel1d(4)
    tensor([1., 3., 3., 1.])
    >>> get_pascal_kernel1d(5)
    tensor([1., 4., 6., 4., 1.])
    >>> get_pascal_kernel1d(6)
    tensor([ 1.,  5., 10., 10.,  5.,  1.])
    """
    pre: list[float] = []
    cur: list[float] = []
    for i in range(kernel_size):
        cur = [1.0] * (i + 1)

        for j in range(1, i // 2 + 1):
            value = pre[j - 1] + pre[j]
            cur[j] = value
            if i != 2 * j:
                cur[-j - 1] = value
        pre = cur

    out = tensor(cur, device=device, dtype=dtype)

    if norm:
        out = out / out.sum()

    return out


def get_pascal_kernel2d(
    kernel_size: _size_2_t,
    norm       : bool = True,
    *,
    device     : Any  = None,
    dtype      : Any  = None,
) -> torch.Tensor:
    """Generate pascal filter kernel by kernel size.

    Args:
        kernel_size: Height and width of the kernel.
        norm: If to normalize the kernel or not. Default: True.
        device: Tensor device desired to create the kernel.
        dtype: Tensor dtype desired to create the kernel.

    Returns:
        If :param:`kernel_size` is an integer the kernel will be shaped as
        :math:`(kernel_size, kernel_size)` otherwise the kernel will be shaped
        as :math: `kernel_size`

    Examples:
    >>> get_pascal_kernel2d(1)
    tensor([[1.]])
    >>> get_pascal_kernel2d(4)
    tensor([[0.0156, 0.0469, 0.0469, 0.0156],
            [0.0469, 0.1406, 0.1406, 0.0469],
            [0.0469, 0.1406, 0.1406, 0.0469],
            [0.0156, 0.0469, 0.0469, 0.0156]])
    >>> get_pascal_kernel2d(4, norm=False)
    tensor([[1., 3., 3., 1.],
            [3., 9., 9., 3.],
            [3., 9., 9., 3.],
            [1., 3., 3., 1.]])
    """
    ky, kx = nn.to_2d_kernel_size(kernel_size=kernel_size)
    ax     = get_pascal_kernel1d(kernel_size=kx, device=device, dtype=dtype)
    ay     = get_pascal_kernel1d(kernel_size=ky, device=device, dtype=dtype)
    filt   = ay[:, None] * ax[None, :]
    if norm:
        filt = filt / torch.sum(filt)
    return filt

# endregion
