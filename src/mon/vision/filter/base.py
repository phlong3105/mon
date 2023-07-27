#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements base filter operations.
"""

from __future__ import annotations

__all__ = [

]

from mon import nn
from mon.nn import functional as F


_VALID_BORDERS   = ["constant", "reflect", "replicate", "circular"]
_VALID_PADDING   = ["valid", "same"]
_VALID_BEHAVIOUR = ["conv", "corr"]


# region Kernel

def normalize_kernel2d(input: torch.Tensor) -> torch.Tensor:
    r"""Normalize both derivative and smoothing kernel."""
    nn.check_shape(input, ["*", "H", "W"])
    norm = input.abs().sum(dim=-1).sum(dim=-1)
    return input / (norm[..., None, None])

# endregion


# region Filter

def filter2d(
    input      : torch.Tensor,
    kernel     : torch.Tensor,
    border_type: str  = "reflect",
    normalized : bool = False,
    padding    : str  = "same",
    behaviour  : str  = "corr",
) -> torch.Tensor:
    r"""Convolve a tensor with a 2d kernel.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input: The input tensor with shape of :math:`(B, C, H, W)`.
        kernel: The kernel to be convolved with the input tensor. The kernel
            shape must be :math:`(1, kH, kW)` or :math:`(B, kH, kW)`.
        border_type: The padding mode to be applied before convolving. The
            expected modes are: ``'constant'``, ``'reflect'``, ``'replicate'``,
            or ``'circular'``.
        normalized: If True, kernel will be L1 normalized.
        padding: This defines the type of padding. 2 modes available ``'same'``
            or ``'valid'``.
        behaviour: Defines the convolution mode -- correlation (default), using
            pytorch conv2d, or true convolution (kernel is flipped). 2 modes
            available ``'corr'`` or ``'conv'``.

    Return:
        The convolved tensor of same size and numbers of channels as the input
        with shape :math:`[B, C, H, W]`.

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
    KORNIA_CHECK_IS_TENSOR(input)
    KORNIA_CHECK_SHAPE(input, ['B', 'C', 'H', 'W'])
    KORNIA_CHECK_IS_TENSOR(kernel)
    KORNIA_CHECK_SHAPE(kernel, ['B', 'H', 'W'])

    KORNIA_CHECK(
        str(border_type).lower() in _VALID_BORDERS,
        f'Invalid border, gotcha {border_type}. Expected one of {_VALID_BORDERS}',
    )
    KORNIA_CHECK(
        str(padding).lower() in _VALID_PADDING,
        f'Invalid padding mode, gotcha {padding}. Expected one of {_VALID_PADDING}',
    )
    KORNIA_CHECK(
        str(behaviour).lower() in _VALID_BEHAVIOUR,
        f'Invalid padding mode, gotcha {behaviour}. Expected one of {_VALID_BEHAVIOUR}',
    )
    # Prepare kernel
    b, c, h, w = input.shape
    if str(behaviour).lower() == "conv":
        tmp_kernel = kernel.flip((-2, -1))[:, None, ...].to(device=input.device, dtype=input.dtype)
    else:
        tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)
    
    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)
    
    tmp_kernel    = tmp_kernel.expand(-1, c, -1, -1)
    height, width = tmp_kernel.shape[-2:]
    
    # Pad the input tensor
    if padding == "same":
        padding_shape: list[int] = nn.get_padding(kernel_size=[height, width])
        input = F.pad(input=input, pad=padding_shape, mode=border_type)

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

# endregion
