#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements Laplacian filters.
"""

from __future__ import annotations

__all__ = [

]

from typing import Any

import torch

import mon.core
from mon import nn
from mon.nn import _size_2_t

console = mon.foundation.console


# region Kernel

def laplacian1d(
    window_size: int,
    *,
    device     : Any = None,
    dtype      : Any = None,
) -> torch.Tensor:
    """One could also use the Laplacian of Gaussian formula to design the filter.
    """
    filter_1d = torch.ones(window_size, device=device, dtype=dtype)
    middle    = window_size // 2
    filter_1d[middle] = 1 - window_size
    return filter_1d


def get_laplacian_kernel1d(
    kernel_size: int,
    *,
    device     : Any = None,
    dtype      : Any = None,
) -> torch.Tensor:
    r"""Function that returns the coefficients of a 1D Laplacian filter.

    Args:
        kernel_size: Filter size. It should be odd and positive.
        device: Tensor device desired to create the kernel.
        dtype: Tensor dtype desired to create the kernel.

    Returns:
        1D tensor with laplacian filter coefficients.

    Shape:
        - Output: math:`(\text{kernel_size})`

    Examples:
        >>> get_laplacian_kernel1d(3)
        tensor([ 1., -2.,  1.])
        >>> get_laplacian_kernel1d(5)
        tensor([ 1.,  1., -4.,  1.,  1.])
    """
    nn.check_kernel_size(kernel_size=kernel_size)
    return laplacian1d(window_size=kernel_size, device=device, dtype=dtype)


def get_laplacian_kernel2d(
    kernel_size: _size_2_t,
    *,
    device     : Any = None,
    dtype      : Any = None,
) -> torch.Tensor:
    r"""Function that returns Gaussian filter matrix coefficients.

    Args:
        kernel_size: Filter size should be odd.
        device: Tensor device desired to create the kernel.
        dtype: Tensor dtype desired to create the kernel.

    Returns:
        2D tensor with laplacian filter matrix coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`

    Examples:
        >>> get_laplacian_kernel2d(3)
        tensor([[ 1.,  1.,  1.],
                [ 1., -8.,  1.],
                [ 1.,  1.,  1.]])
        >>> get_laplacian_kernel2d(5)
        tensor([[  1.,   1.,   1.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.],
                [  1.,   1., -24.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.]])
    """
    ky, kx = nn.to_2d_kernel_size(kernel_size)
    nn.check_kernel_size((ky, kx))
    kernel = torch.ones((ky, kx), device=device, dtype=dtype)
    mid_x  = kx // 2
    mid_y  = ky // 2
    kernel[mid_y, mid_x] = 1 - kernel.sum()
    return kernel

# endregion
