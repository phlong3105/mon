#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements box filters.
"""

from __future__ import annotations

__all__ = [
    "BoxBlur", "box_blur", "get_box_kernel1d", "get_box_kernel2d",
]

from typing import Any

import torch

from mon import nn
from mon.globals import LAYERS
from mon.nn.typing import _size_2_t
from mon.vision.filter import core


# region Kernel

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
    scale = torch.tensor(1.0 / kernel_size, device=device, dtype=dtype)
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
    scale  = torch.tensor(1.0 / (kx * ky), device=device, dtype=dtype)
    return scale.expand(1, ky, kx)

# endregion


# region Box Blur

def box_blur(
    input      : torch.Tensor,
    kernel_size: _size_2_t,
    border_type: str  = "reflect",
    separable  : bool = False
) -> torch.Tensor:
    r"""Blur an image using the box filter.
    
    The function smooths an image using the kernel:
    .. math::
        K = \frac{1}{\text{kernel_size}_x * \text{kernel_size}_y}
        \begin{bmatrix}
            1 & 1 & 1 & \cdots & 1 & 1 \\
            1 & 1 & 1 & \cdots & 1 & 1 \\
            \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
            1 & 1 & 1 & \cdots & 1 & 1 \\
        \end{bmatrix}

    Args:
        input: The image to blur with shape :math:`[B, C, H, W]`.
        kernel_size: The blurring kernel size.
        border_type: The padding mode to be applied before convolving. The
            expected modes are: ``'constant'``, ``'reflect'``, ``'replicate'``
            or ``'circular'``.
        separable: Run as composition of two 1d-convolutions.

    Returns:
        The blurred image with shape :math:`[B, C, H, W]`.

    Notes:
       See a working example `here
       <https://kornia-tutorials.readthedocs.io/en/latest/filtering_operators.html>`__.
    """
    if separable:
        ky, kx   = nn.to_2d_kernel_size(kernel_size=kernel_size, behaviour="conv")
        kernel_y = get_box_kernel1d(ky, device=input.device, dtype=input.dtype)
        kernel_x = get_box_kernel1d(kx, device=input.device, dtype=input.dtype)
        output   = core.filter2d_separable(
            input       = input,
            kernel_x    = kernel_x,
            kernel_y    = kernel_y,
            border_type = border_type,
        )
    else:
        kernel = get_box_kernel2d(kernel_size, device=input.device, dtype=input.dtype)
        output = core.filter2d(input=input, kernel=kernel, border_type=border_type)

    return output


@LAYERS.register()
class BoxBlur(nn.Module):
    r"""Blur an image using the box filter.

    The function smooths an image using the kernel:
    .. math::
        K = \frac{1}{\text{kernel_size}_x * \text{kernel_size}_y}
        \begin{bmatrix}
            1 & 1 & 1 & \cdots & 1 & 1 \\
            1 & 1 & 1 & \cdots & 1 & 1 \\
            \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
            1 & 1 & 1 & \cdots & 1 & 1 \\
        \end{bmatrix}

    Args:
        kernel_size: The blurring kernel size.
        border_type: The padding mode to be applied before convolving. The
            expected modes are: ``'constant'``, ``'reflect'``, ``'replicate'``
            or ``'circular'``.
        separable: Run as composition of two 1d-convolutions.

    Returns:
        The blurred input tensor.
    
    Example:
        >>> input  = torch.rand(2, 4, 5, 7)
        >>> blur   = BoxBlur((3, 3))
        >>> output = blur(input)  # 2x4x5x7
        >>> output.shape
        torch.Size([2, 4, 5, 7])
    """

    def __init__(
        self,
        kernel_size: _size_2_t,
        border_type: str  = "reflect",
        separable  : bool = False
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.border_type = border_type
        self.separable   = separable

        if separable:
            ky, kx = nn.to_2d_kernel_size(kernel_size=self.kernel_size, behaviour="conv")
            self.register_buffer("kernel_y", get_box_kernel1d(ky))
            self.register_buffer("kernel_x", get_box_kernel1d(kx))
            self.kernel_y: torch.Tensor
            self.kernel_x: torch.Tensor
        else:
            self.register_buffer("kernel", get_box_kernel2d(kernel_size))
            self.kernel: torch.Tensor

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(kernel_size={self.kernel_size}, "
            f"border_type={self.border_type}, "
            f"separable={self.separable})"
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.separable:
            return core.filter2d_separable(
                input       = input,
                kernel_x    = self.kernel_x,
                kernel_y    = self.kernel_y,
                border_type = self.border_type,
            )
        return core.filter2d(
            input       = input,
            kernel      = self.kernel,
            border_type = self.border_type
        )
    
# endregion
