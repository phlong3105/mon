#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements Sobel filters.
"""

from __future__ import annotations

__all__ = [

]

from typing import Any

import torch

import mon.core

console = mon.foundation.console


# region Kernel

def get_diff_kernel_3x3(
    *,
    device: Any = None,
    dtype : Any = None,
) -> torch.Tensor:
    """Utility function that returns a first order derivative kernel of 3x3."""
    return torch.tensor(
        [
            [-0.0, 0.0, 0.0],
            [-1.0, 0.0, 1.0],
            [-0.0, 0.0, 0.0]
        ],
        device = device,
        dtype  = dtype,
    )


def get_diff_kernel2d(
    *,
    device: Any = None,
    dtype : Any = None,
) -> torch.Tensor:
    kernel_x = get_diff_kernel_3x3(device=device, dtype=dtype)
    kernel_y = kernel_x.transpose(0, 1)
    return torch.stack([kernel_x, kernel_y])


def get_diff_kernel2d_2nd_order(
    *,
    device: Any = None,
    dtype : Any = None,
) -> torch.Tensor:
    gxx = torch.tensor([[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]], device=device, dtype=dtype)
    gyy = gxx.transpose(0, 1)
    gxy = torch.tensor([[-1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, -1.0]], device=device, dtype=dtype)
    return torch.stack([gxx, gxy, gyy])


def get_diff_kernel3d(
    device: Any = None,
    dtype : Any = None,
) -> torch.Tensor:
    """Utility function that returns a first order derivative kernel of 3x3x3."""
    kernel = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [-0.5, 0.0, 0.5], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0,  0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]],
                [[0.0,  0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0,  0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0,  0.5, 0.0], [0.0, 0.0, 0.0]],
            ],
        ],
        device = device,
        dtype  = dtype,
    )
    return kernel[:, None, ...]


def get_diff_kernel3d_2nd_order(
    device: Any = None,
    dtype : Any = None,
) -> torch.Tensor:
    """Utility function that returns a first order derivative kernel of 3x3x3.
    """
    kernel = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.0,  0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0,  0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0,  0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0,  0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0,  1.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0,  1.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0,  0.0], [0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0]],
                [[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]],
                [[0.0, 0.0,  0.0], [0.0, 0.0, 0.0], [ 0.0, 0.0, 0.0]],
            ],
            [
                [[0.0,  1.0, 0.0], [0.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
                [[0.0,  0.0, 0.0], [0.0, 0.0, 0.0], [0.0,  0.0, 0.0]],
                [[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0,  1.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [ 1.0, 0.0, -1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [ 0.0, 0.0,  0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [-1.0, 0.0,  1.0], [0.0, 0.0, 0.0]],
            ],
        ],
        device = device,
        dtype  = dtype,
    )
    return kernel[:, None, ...]


def get_sobel_kernel_3x3(
    *,
    device: Any = None,
    dtype : Any = None,
) -> torch.Tensor:
    """Utility function that returns a sobel kernel of 3x3."""
    return torch.tensor(
        [
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0]
        ],
        device = device,
        dtype  = dtype,
    )


def get_sobel_kernel_5x5_2nd_order(
    *,
    device: Any = None,
    dtype : Any = None,
) -> torch.Tensor:
    """Utility function that returns a 2nd order sobel kernel of 5x5."""
    return torch.tensor(
        [
            [-1.0, 0.0,  2.0, 0.0, -1.0],
            [-4.0, 0.0,  8.0, 0.0, -4.0],
            [-6.0, 0.0, 12.0, 0.0, -6.0],
            [-4.0, 0.0,  8.0, 0.0, -4.0],
            [-1.0, 0.0,  2.0, 0.0, -1.0],
        ],
        device = device,
        dtype  = dtype,
    )

def get_sobel_kernel_5x5_2nd_order_xy(
    *,
    device: Any = None,
    dtype : Any = None,
) -> torch.Tensor:
    """Utility function that returns a 2nd order sobel kernel of 5x5."""
    return torch.tensor(
        [
            [-1.0, -2.0, 0.0,  2.0,  1.0],
            [-2.0, -4.0, 0.0,  4.0,  2.0],
            [ 0.0,  0.0, 0.0,  0.0,  0.0],
            [ 2.0,  4.0, 0.0, -4.0, -2.0],
            [ 1.0,  2.0, 0.0, -2.0, -1.0],
        ],
        device = device,
        dtype  = dtype,
    )


def get_sobel_kernel2d(
     *,
    device: Any = None,
    dtype : Any = None,
) -> torch.Tensor:
    kernel_x = get_sobel_kernel_3x3(device=device, dtype=dtype)
    kernel_y = kernel_x.transpose(0, 1)
    return torch.stack([kernel_x, kernel_y])


def get_sobel_kernel2d_2nd_order(
    *,
    device: Any = None,
    dtype : Any = None,
) -> torch.Tensor:
    gxx = get_sobel_kernel_5x5_2nd_order(device=device, dtype=dtype)
    gyy = gxx.transpose(0, 1)
    gxy = get_sobel_kernel_5x5_2nd_order_xy(device=device, dtype=dtype)
    return torch.stack([gxx, gxy, gyy])


def get_spatial_gradient_kernel2d(
    mode  : str,
    order : int,
    *,
    device: Any = None,
    dtype : Any = None,
) -> torch.Tensor:
    r"""Function that returns kernel for 1st or 2nd order image gradients,
    using one of the following operators: sobel, diff.
    """
    assert mode.lower() in ["sobel", "diff"], f"`mode` should be `sobel` or `diff`. Got {mode}."
    assert order in [1, 2], f"Order should be 1 or 2. Got {order}."
    
    if mode == "sobel" and order == 1:
        kernel: torch.Tensor = get_sobel_kernel2d(device=device, dtype=dtype)
    elif mode == "sobel" and order == 2:
        kernel = get_sobel_kernel2d_2nd_order(device=device, dtype=dtype)
    elif mode == "diff" and order == 1:
        kernel = get_diff_kernel2d(device=device, dtype=dtype)
    elif mode == "diff" and order == 2:
        kernel = get_diff_kernel2d_2nd_order(device=device, dtype=dtype)
    else:
        raise NotImplementedError(
            f"Not implemented 2d gradient kernel for `order` {order} on `mode` {mode}."
        )
    return kernel


def get_spatial_gradient_kernel3d(
    mode  : str,
    order : int,
    *,
    device: Any = None,
    dtype : Any = None,
) -> torch.Tensor:
    r"""Function that returns kernel for 1st or 2nd order scale pyramid
    gradients, using one of the following operators: sobel, diff.
    """
    assert mode.lower() in ["sobel", "diff"], f"`mode` should be `sobel` or `diff`. Got {mode}."
    assert order in [1, 2], f"Order should be 1 or 2. Got {order}."
    
    if mode == "diff" and order == 1:
        kernel = get_diff_kernel3d(device=device, dtype=dtype)
    elif mode == "diff" and order == 2:
        kernel = get_diff_kernel3d_2nd_order(device=device, dtype=dtype)
    else:
        raise NotImplementedError(
            f"Not implemented 3d gradient kernel for `order` {order} on `mode` {mode}."
        )
    return kernel

# endregion
