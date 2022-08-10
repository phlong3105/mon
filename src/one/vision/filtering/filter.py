#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basic filters.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from one.core import assert_collection_contain_item
from one.core import assert_str
from one.core import assert_tensor_of_ndim
from one.core import BorderType
from one.core import BorderType_
from one.vision.filtering.kernel import normalize_kernel2d


def _compute_padding(kernel_size: list[int]) -> list[int]:
    """
    Compute padding tuple.
    """
    # 4 or 6 ints: (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # For even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]
        pad_front    = computed_tmp // 2
        pad_rear     = computed_tmp - pad_front
        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def filter2d(
    input      : Tensor,
    kernel     : Tensor,
    border_type: BorderType_ = "reflect",
    normalized : bool        = False,
    padding    : str         = "same",
) -> Tensor:
    """
    Convolve a tensor with a 2d kernel.
    
    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.
    
    Args:
        input (Tensor): The input tensor with shape of [B, C, H, W].
        kernel (Tensor): The kernel to be convolved with the input tensor.
            The kernel shape must be [1, kH, kW] or [B, kH, kW].
        border_type (BorderType_): The padding mode to be applied before
            convolving. One of: [`constant`, `reflect`, `replicate`, `circular`].
            Defaults to reflect.
        normalized (bool): If True, kernel will be L1 normalized.
            Defaults to False.
        padding (str): This defines the type of padding.
            One of: [`same`, `valid`]. Defaults to same.
    
    Return:
        The convolved tensor of same size and numbers of channels as the input
        with shape [B, C, H, W].
    
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
    assert_tensor_of_ndim(input, 4)
    assert_tensor_of_ndim(kernel, 3)
    if not ((kernel.shape[0] == 0) or (kernel.shape[0] == input.shape[0])):
        raise ValueError(
            f"`kernel` must have same of [1, H, W] or [B, H, W]. "
            f"But got: {kernel.shape}"
        )

    border_type = BorderType.from_value(border_type)
    assert_str(padding)
    assert_collection_contain_item(["valid", "same"], padding)
    
    # Prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel.unsqueeze(1).to(input)

    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)

    tmp_kernel    = tmp_kernel.expand(-1, c, -1, -1)
    height, width = tmp_kernel.shape[-2:]

    # Pad the input tensor
    if padding == "same":
        padding_shape = _compute_padding([height, width])
        input         = F.pad(
            input = input,
            pad   = padding_shape,
            mode  = border_type.value
        )

    # Kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input      = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # Convolve the tensor with the kernel.
    output = F.conv2d(
        input   = input,
        weight  = tmp_kernel,
        groups  = tmp_kernel.size(0),
        padding = 0,
        stride  = 1
    )

    if padding == "same":
        out = output.view(b, c, h, w)
    else:
        out = output.view(b, c, h - height + 1, w - width + 1)

    return out


def filter2d_separable(
    input      : Tensor,
    kernel_x   : Tensor,
    kernel_y   : Tensor,
    border_type: str    = "reflect",
    normalized : bool   = False,
    padding    : str    = "same",
) -> Tensor:
    """
    Convolve a tensor with two 1d kernels, in x and y directions.
    
    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.
    
    Args:
        input (Tensor): The input tensor with shape of [B, C, H, W].
        kernel_x (Tensor): The kernel to be convolved with the input tensor.
            The kernel shape must be [1, kW] or [B, kW].
        kernel_y (Tensor): The kernel to be convolved with the input tensor.
            The kernel shape must be [1, kH] or [B, kH].
        border_type (BorderType_): The padding mode to be applied before
            convolving. One of: [`constant`, `reflect`, `replicate`, `circular`].
            Defaults to reflect.
        normalized (bool): If True, kernel will be L1 normalized.
            Defaults to False.
        padding (str): This defines the type of padding.
            One of: [`same`, `valid`]. Defaults to same.
    
    Return:
        The convolved tensor of same size and numbers of channels as the input
        with shape [B, C, H, W].
    
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
    output_x = filter2d(
        input       = input,
        kernel      = kernel_x.unsqueeze(0),
        border_type = border_type,
        normalized  = normalized,
        padding     = padding
    )
    output = filter2d(
        input       = output_x,
        kernel      = kernel_y.unsqueeze(-1),
        border_type = border_type,
        normalized  = normalized,
        padding     = padding
    )
    return output


def filter3d(
    input      : Tensor,
    kernel     : Tensor,
    border_type: str  = "replicate",
    normalized : bool = False
) -> Tensor:
    """
    Convolve a tensor with a 3d kernel.
    
    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.
    
    Args:
        input (Tensor): The input tensor with shape of [B, C, D, H, W].
        kernel (Tensor): The kernel to be convolved with the input tensor.
            The kernel shape must be [1, kD, kH, kW] or [B, kD, kH, kW].
        border_type (BorderType_): The padding mode to be applied before
            convolving. One of: [`constant`, `reflect`, `replicate`, `circular`].
            Defaults to reflect.
        normalized (bool): If True, kernel will be L1 normalized.
            Defaults to False.
     
    Return:
        The convolved tensor of same size and numbers of channels as the input
        with shape [B, C, D, H, W].
   
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
    assert_tensor_of_ndim(input,  5)
    assert_tensor_of_ndim(kernel, 4)
    if not len(kernel.shape) == 4 and kernel.shape[0] != 1:
        raise ValueError(
            f"Invalid kernel shape, we expect [1, D, H, W]. "
            f"But got: {kernel.shape}"
        )

    border_type = BorderType.from_value(border_type)
    
    # Prepare kernel
    b, c, d, h, w = input.shape
    tmp_kernel    = kernel.unsqueeze(1).to(input)

    if normalized:
        bk, dk, hk, wk = kernel.shape
        tmp_kernel     = normalize_kernel2d(
            tmp_kernel.view(bk, dk, hk * wk)
        ).view_as(tmp_kernel)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1, -1)

    # Pad the input tensor
    depth, height, width = tmp_kernel.shape[-3:]
    padding_shape = _compute_padding([depth, height, width])
    input_pad     = F.pad(
        input = input,
        pad   =  padding_shape,
        mode  = border_type
    )

    # Kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, depth, height, width)
    input_pad  = input_pad.view(-1, tmp_kernel.size(0), input_pad.size(-3), input_pad.size(-2), input_pad.size(-1))

    # Convolve the tensor with the kernel.
    output = F.conv3d(
        input   = input_pad,
        weight  = tmp_kernel,
        groups  = tmp_kernel.size(0),
        padding = 0,
        stride  = 1
    )

    return output.view(b, c, d, h, w)
