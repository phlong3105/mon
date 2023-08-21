#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements base filter operations.
"""

from __future__ import annotations

__all__ = [
    "filter2d",
    "filter2d_separable",
    "filter3d",
    "normalize_kernel2d",
]

import torch

from mon import nn
from mon.nn import functional as F


_VALID_BORDERS   = ["constant", "reflect", "replicate", "circular"]
_VALID_PADDING   = ["valid", "same"]
_VALID_BEHAVIOUR = ["conv", "corr"]


# region Kernel

def normalize_kernel2d(kernel: torch.Tensor) -> torch.Tensor:
    r"""Normalize both derivative and smoothing kernel."""
    nn.check_shape(input=kernel, shape=["*", "H", "W"])
    norm = kernel.abs().sum(dim=-1).sum(dim=-1)
    return kernel / (norm[..., None, None])

# endregion


# region Basic Filter

def filter2d(
    input      : torch.Tensor,
    kernel     : torch.Tensor,
    border_type: str  = "reflect",
    normalized : bool = False,
    padding    : str  = "same",
    behaviour  : str  = "corr",
) -> torch.Tensor:
    r"""Convolve a tensor with a 2-D kernel.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so that
    the output remains in the same shape.

    Args:
        input: The input tensor with shape of :math:`[B, C, H, W]`.
        kernel: The kernel to be convolved with the input tensor. The kernel
            shape must be :math:`[1, kH, kW]` or :math:`[B, kH, kW]`.
        border_type: The padding mode to be applied before convolving. The
            expected modes are: ``'constant'``, ``'reflect'``, ``'replicate'``,
            or ``'circular'``.
        normalized: If ``True``, kernel will be L1 normalized.
        padding: This defines the type of padding. 2 modes available ``'same'``
            or ``'valid'``.
        behaviour: Defines the convolution mode -- correlation (default), using
            PyTorch :class:`torch.nn.Conv2d`, or true convolution (kernel is
            flipped). 2 modes available ``'corr'`` or ``'conv'``.

    Return:
        The convolved tensor of the same size and numbers of channels as the
        input with shape :math:`[B, C, H, W]`.

    Example:
        >>> input = torch.tensor([[[
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 5., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],]]])
        >>> kernel = torch.ones(1, 3, 3)
        >>> filter2d(input, kernel, padding='same')
        tensor([[[[0., 0., 0., 0., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 0., 0., 0., 0.]]]])
    """
    nn.check_shape(input=input,  shape=["B", "C", "H", "W"])
    nn.check_shape(input=kernel, shape=["B", "H", "W"])
    
    assert str(border_type).lower() in _VALID_BORDERS, \
        f"Border must be one of {_VALID_BORDERS}, but got {border_type}."
    assert str(padding).lower() in _VALID_PADDING, \
        f"Padding mode must be one of {_VALID_PADDING}, but got {padding}."
    assert str(behaviour).lower() in _VALID_BEHAVIOUR, \
        f"Behaviour must be one of {_VALID_BEHAVIOUR}, but got {behaviour}."
    
    # Prepare kernel
    b, c, h, w = input.shape
    if str(behaviour).lower() == "conv":
        tmp_kernel = kernel.flip((-2, -1))[:, None, ...].to(device=input.device, dtype=input.dtype)
    else:
        tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)
    
    if normalized:
        tmp_kernel = normalize_kernel2d(kernel=tmp_kernel)
    
    tmp_kernel    = tmp_kernel.expand(-1, c, -1, -1)
    height, width = tmp_kernel.shape[-2:]
    
    # Pad the input tensor
    if padding == "same":
        padding_shape = nn.get_padding(kernel_size=[height, width])
        input         = F.pad(input=input, pad=padding_shape, mode=border_type)

    # Kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input      = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # Convolve the tensor with the kernel.
    output = F.conv2d(
        input   = input,
        weight  = tmp_kernel,
        stride  = 1,
        padding = 0,
        groups  = tmp_kernel.size(0),
    )
    
    if padding == "same":
        output = output.view(b, c, h, w)
    else:
        output = output.view(b, c, h - height + 1, w - width + 1)
    return output


def filter2d_separable(
    input      : torch.Tensor,
    kernel_x   : torch.Tensor,
    kernel_y   : torch.Tensor,
    border_type: str  = "reflect",
    normalized : bool = False,
    padding    : str  = "same",
    behaviour  : str  = "corr",
) -> torch.Tensor:
    r"""Convolve a tensor with two 1-D kernels, in x and y directions.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input: the input tensor with shape of :math:`[B, C, H, W]`.
        kernel_x: The kernel to be convolved with the input tensor. The kernel
            shape must be :math:`[1, kW]` or :math:`[B, kW]`.
        kernel_y: The kernel to be convolved with the input tensor. The kernel
            shape must be :math:`[1, kH]` or :math:`[B, kH]`.
        border_type: The padding mode to be applied before convolving. The
            expected modes are: ``'constant'``, ``'reflect'``, ``'replicate'``,
            or ``'circular'``.
        normalized: If ``True``, kernel will be L1 normalized.
        padding: This defines the type of padding. 2 modes available ``'same'``
            or ``'valid'``.
        behaviour: Defines the convolution mode -- correlation (default), using
            PyTorch :class:`torch.nn.Conv2d`, or true convolution (kernel is
            flipped). 2 modes available ``'corr'`` or ``'conv'``.
        
    Return:
        The convolved tensor of the same size and numbers of channels as the
        input with shape :math:`[B, C, H, W]`.

    Example:
        >>> input = torch.tensor([[[
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 5., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],]]])
        >>> kernel = torch.ones(1, 3)
        >>> filter2d_separable(input, kernel, kernel, padding='same')
        tensor([[[[0., 0., 0., 0., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 0., 0., 0., 0.]]]])
    """
    out_x = filter2d(
        input       = input,
        kernel      = kernel_x[..., None, : ],
        border_type = border_type,
        normalized  = normalized,
        padding     = padding,
        behaviour   = behaviour,
    )
    out = filter2d(
        input       = out_x,
        kernel      = kernel_y[..., None],
        border_type = border_type,
        normalized  = normalized,
        padding     = padding,
        behaviour   = behaviour,
    )
    return out


def filter3d(
    input      : torch.Tensor,
    kernel     : torch.Tensor,
    border_type: str  = "replicate",
    normalized : bool = False,
) -> torch.Tensor:
    r"""Convolve a tensor with a 3-D kernel.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so that
    the output remains in the same shape.

    Args:
        input: The input tensor with shape of :math:`[B, C, D, H, W]`.
        kernel: The kernel to be convolved with the input tensor. The kernel
            shape must be :math:`[1, kD, kH, kW]` or :math:`[B, kD, kH, kW]`.
        border_type: The padding mode to be applied before convolving. The
            expected modes are: ``'constant'``, ``'replicate'``, or
            ``'circular'``.
        normalized: If ``True``, kernel will be L1 normalized.

    Return:
        The convolved tensor of same size and numbers of channels as the input
        with shape :math:`[B, C, D, H, W]`.

    Example:
        >>> input = torch.tensor([[[
        ...    [[0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.]],
        ...    [[0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 5., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.]],
        ...    [[0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.]]
        ... ]]])
        >>> kernel = torch.ones(1, 3, 3, 3)
        >>> filter3d(input, kernel)
        tensor([[[[[0., 0., 0., 0., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 0., 0., 0., 0.]],
        <BLANKLINE>
                  [[0., 0., 0., 0., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 0., 0., 0., 0.]],
        <BLANKLINE>
                  [[0., 0., 0., 0., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 0., 0., 0., 0.]]]]])
    """
    nn.check_shape(input=input,  shape=["B", "C", "D", "H", "W"])
    nn.check_shape(input=kernel, shape=["B", "D", "H", "W"])
    
    assert str(border_type).lower() in _VALID_BORDERS, \
        f"Border must be one of {_VALID_BORDERS}, but got {border_type}."

    # Prepare kernel
    b, c, d, h, w = input.shape
    tmp_kernel    = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    if normalized:
        bk, dk, hk, wk = kernel.shape
        tmp_kernel     = normalize_kernel2d(tmp_kernel.view(bk, dk, hk * wk)).view_as(tmp_kernel)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1, -1)

    # Pad the input tensor
    depth, height, width = tmp_kernel.shape[-3:]
    padding_shape        = nn.get_padding(kernel_size=[depth, height, width])
    input_pad            = F.pad(input=input, pad=padding_shape, mode=border_type)
    
    # Kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, depth, height, width)
    input_pad  = input_pad.view(-1, tmp_kernel.size(0), input_pad.size(-3), input_pad.size(-2), input_pad.size(-1))

    # Convolve the tensor with the kernel.
    output = F.conv3d(
        input   = input_pad,
        weight  = tmp_kernel,
        stride  = 1,
        padding = 0,
        groups  = tmp_kernel.size(0),
    )
    
    return output.view(b, c, d, h, w)

# endregion
