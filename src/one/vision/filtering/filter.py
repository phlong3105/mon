#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basic filters.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor

from one.core import assert_collection_contain_item
from one.core import assert_str
from one.core import assert_tensor_of_ndim
from one.core import BorderType
from one.core import BorderType_
from one.vision.filtering.kernel import get_binary_kernel2d
from one.vision.filtering.kernel import get_box_kernel2d
from one.vision.filtering.kernel import get_canny_nms_kernel
from one.vision.filtering.kernel import get_gaussian_kernel1d
from one.vision.filtering.kernel import get_gaussian_kernel2d
from one.vision.filtering.kernel import get_hysteresis_kernel
from one.vision.filtering.kernel import get_laplacian_kernel2d
from one.vision.filtering.kernel import get_pascal_kernel_2d
from one.vision.filtering.kernel import get_spatial_gradient_kernel2d
from one.vision.filtering.kernel import get_spatial_gradient_kernel3d
from one.vision.filtering.kernel import normalize_kernel2d


# H1: - Basic Filter -----------------------------------------------------------

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


def _compute_zero_padding(kernel_size: tuple[int, int]) -> tuple[int, int]:
    """
    Utility function that computes zero padding tuple.
    """
    computed = [(k - 1) // 2 for k in kernel_size]
    return computed[0], computed[1]


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
        pad   = padding_shape,
        mode  = border_type.value
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


# H1: - Blur Filter ------------------------------------------------------------

def _blur_pool_by_kernel2d(input: Tensor, kernel: Tensor, stride: int):
    """
    Compute blur pool by a given [C, C_{out}, N, N] kernel.
    """
    assert_tensor_of_ndim(kernel, 4)
    if not kernel.size(-1) == kernel.size(-2):
        raise ValueError(
            f"Expect `kernel` has shape [C, C_out, N, N]. "
            f"But got: {kernel.shape}."
        )
    padding = _compute_zero_padding((kernel.shape[-2], kernel.shape[-1]))
    return F.conv2d(
        input   = input,
        weight  = kernel,
        padding = padding,
        stride  = stride,
        groups  = input.size(1)
    )


def _max_blur_pool_by_kernel2d(
    input        : Tensor,
    kernel       : Tensor,
    stride       : int,
    max_pool_size: int,
    ceil_mode    : bool
):
    """
    Compute max blur pool by a given [C, C_{out}, N, N] kernel.
    """
    if not kernel.size(-1) == kernel.size(-2):
        raise ValueError(
            f"Expect `kernel` has shape [C, C_out, N, N]. "
            f"But got: {kernel.shape}."
        )
    # Compute local maxima
    input = F.max_pool2d(
        input       = input,
        kernel_size = max_pool_size,
        padding     = 0,
        stride      = 1,
        ceil_mode   = ceil_mode
    )
    # Blur and downsample
    padding = _compute_zero_padding((kernel.shape[-2], kernel.shape[-1]))
    return F.conv2d(
        input   = input,
        weight  = kernel,
        padding = padding,
        stride  = stride,
        groups  = input.size(1)
    )


def blur_pool2d(input: Tensor, kernel_size: int, stride: int = 2):
    """
    Compute blurs and downsample a given feature map.
    
    Args:
        input (Tensor): Image of shape [B, C, H, W].
        kernel_size (int): Kernel size for max pooling.
        stride (int): Stride for pooling.
    
    Returns:
        The transformed image of shape [N, C, H_out, W_out], where:
            H_out = (H + 2  * kernel_size // 2 - kernel_size) / stride + 1
            W_out = (W + 2  * kernel_size // 2 - kernel_size) / stride + 1
   
    Examples:
        >>> input = torch.eye(5)[None, None]
        >>> blur_pool2d(input, 3)
        tensor([[[[0.3125, 0.0625, 0.0000],
                  [0.0625, 0.3750, 0.0625],
                  [0.0000, 0.0625, 0.3125]]]])
    """
    assert_tensor_of_ndim(input, 4)
    kernel = get_pascal_kernel_2d(
        kernel_size=kernel_size, norm=True
    ).repeat((input.size(1), 1, 1, 1)).to(input)
    return _blur_pool_by_kernel2d(input=input, kernel=kernel, stride=stride)


def box_blur(
    input      : Tensor,
    kernel_size: tuple[int, int],
    border_type: BorderType_ = "reflect",
    normalized : bool        = True
) -> Tensor:
    """
    Blur an image using the box filter.
    
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
        input (Tensor): The image of shape [B, C, H, W].
        kernel_size (tuple[int, int]): The blurring kernel size.
        border_type (BorderType_): The padding mode to be applied before
            convolving. One of: [`constant`, `reflect`, `replicate`, `circular`].
            Defaults to reflect.
        normalized (bool): If True, L1 norm of the kernel is set to 1.
            Defaults to True.
    
    Returns:
        The blurred image of shape [B, C, H, W].

    Example:
        >>> input  = torch.rand(2, 4, 5, 7)
        >>> output = box_blur(input, (3, 3))  # 2x4x5x7
        >>> output.shape
        torch.Size([2, 4, 5, 7])
    """
    kernel = get_box_kernel2d(kernel_size)
    if normalized:
        kernel = normalize_kernel2d(kernel)
    return filter2d(input=input, kernel=kernel, border_type=border_type)


def gaussian_blur2d(
    input      : Tensor,
    kernel_size: tuple[int, int],
    sigma      : tuple[float, float],
    border_type: BorderType_ = "reflect",
    separable  : bool        = True,
) -> Tensor:
    """
    Create an operator that blurs a tensor using a Gaussian filter.

    The operator smooths the given tensor with a gaussian kernel by convolving
    it to each channel. It supports batched operation.
    
    Args:
        input (Tensor): The image of shape [B, C, H, W].
        kernel_size (tuple[int, int]): The blurring kernel size.
        sigma (tuple[float, float]): The standard deviation of the kernel.
        border_type (BorderType_): The padding mode to be applied before
            convolving. One of: [`constant`, `reflect`, `replicate`, `circular`].
            Defaults to reflect.
        separable (bool): Run as composition of two 1d-convolutions.
            Defaults to True.
    
    Returns:
        The blurred image of shape [B, C, H, W].
    
    Examples:
        >>> input  = torch.rand(2, 4, 5, 5)
        >>> output = gaussian_blur2d(input, (3, 3), (1.5, 1.5))
        >>> output.shape
        torch.Size([2, 4, 5, 5])
    """
    assert_tensor_of_ndim(input, 4)
    if separable:
        kernel_x = get_gaussian_kernel1d(kernel_size[1], sigma[1])
        kernel_y = get_gaussian_kernel1d(kernel_size[0], sigma[0])
        output   = filter2d_separable(
            input       = input,
            kernel_x    = kernel_x[None],
            kernel_y    = kernel_y[None],
            border_type = border_type
        )
    else:
        kernel = get_gaussian_kernel2d(kernel_size, sigma)
        output = filter2d(
            input       = input,
            kernel      = kernel[None],
            border_type = border_type
        )
    return output


def laplacian(
    input      : Tensor,
    kernel_size: int,
    border_type: BorderType_ = "reflect",
    normalized : bool        = True
) -> Tensor:
    """
    Create an operator that returns a tensor using a Laplacian filter.

    The operator smooths the given tensor with a laplacian kernel by convolving
    it to each channel. It supports batched operation.
   
    Args:
        input (Tensor): The image of shape [B, C, H, W].
        kernel_size (tuple[int, int]): The blurring kernel size.
        border_type (BorderType_): The padding mode to be applied before
            convolving. One of: [`constant`, `reflect`, `replicate`, `circular`].
            Defaults to reflect.
        normalized (bool): If True, L1 norm of the kernel is set to 1.
            Defaults to True.
        
    Return:
        The blurred image of shape [B, C, H, W].
    
    
    Examples:
        >>> input  = torch.rand(2, 4, 5, 5)
        >>> output = laplacian(input, 3)
        >>> output.shape
        torch.Size([2, 4, 5, 5])
    """
    assert_tensor_of_ndim(input, 4)
    kernel = torch.unsqueeze(get_laplacian_kernel2d(kernel_size), dim=0)
    if normalized:
        kernel = normalize_kernel2d(kernel)
    return filter2d(input=input, kernel=kernel, border_type=border_type)


def max_blur_pool2d(
    input        : Tensor,
    kernel_size  : int,
    stride       : int  = 2,
    max_pool_size: int  = 2,
    ceil_mode    : bool = False
) -> Tensor:
    """
    Compute pools and blurs and downsample a given feature map.
    
    Args:
        input (Tensor): Image of shape [B, C, H, W].
        kernel_size (int): Kernel size for max pooling.
        stride (int): Stride for pooling. Defaults to 2.
        max_pool_size (int): Kernel size for max pooling. Defaults to 2.
        ceil_mode (bool): Should be true to match output size of Conv2d with
            same kernel size. Defaults to False.

    Examples:
        >>> input = torch.eye(5)[None, None]
        >>> max_blur_pool2d(input, 3)
        tensor([[[[0.5625, 0.3125],
                  [0.3125, 0.8750]]]])
    """
    assert_tensor_of_ndim(input, 4)
    kernel = get_pascal_kernel_2d(
        kernel_size=kernel_size, norm=True
    ).repeat((input.size(1), 1, 1, 1)).to(input)
    return _max_blur_pool_by_kernel2d(
        input         = input,
        kernel        = kernel,
        stride        = stride,
        max_pool_size = max_pool_size,
        ceil_mode     = ceil_mode
    )


def median_blur(input: Tensor, kernel_size: tuple[int, int]) -> Tensor:
    """
    Blur an image using the median filter.
   
    Args:
        input (Tensor): Image of shape [B, C, H, W].
        kernel_size (tuple[int, int]): Kernel size for max pooling.
    
    Returns:
        The blurred input image of shape [B, C, H, W].

    Example:
        >>> input  = torch.rand(2, 4, 5, 7)
        >>> output = median_blur(input, (3, 3))
        >>> output.shape
        torch.Size([2, 4, 5, 7])
    """
    assert_tensor_of_ndim(input, 4)
    padding = _compute_zero_padding(kernel_size)

    # Prepare kernel
    kernel     = get_binary_kernel2d(kernel_size).to(input)
    b, c, h, w = input.shape

    # Map the local window to single vector
    features = F.conv2d(
        input   = input.reshape(b * c, 1, h, w),
        weight  = kernel,
        padding = padding,
        stride  = 1
    )
    features = features.view(b, c, -1, h, w)  # BxCx(K_h * K_w)xHxW

    # Compute the median along the feature axis
    median = torch.median(features, dim=2)[0]

    return median


def sobel(input: Tensor, normalized: bool = True, eps: float = 1e-6) -> Tensor:
    """
    Compute the Sobel operator and returns the magnitude per channel.

    Args:
        input (Tensor): The image of shape [B, C, H, W].
        normalized (bool): If True, L1 norm of the kernel is set to 1.
            Defaults to True.
        eps (float): Regularization number to avoid NaN during backprop.
            Defaults to 1e-6.
    
    Return:
        The sobel edge gradient magnitudes map of shape [B, C, H, W].

    Example:
        >>> input  = torch.rand(1, 3, 4, 4)
        >>> output = sobel(input)  # 1x3x4x4
        >>> output.shape
        torch.Size([1, 3, 4, 4])
    """
    assert_tensor_of_ndim(input, 4)
    # Compute the x/y gradients
    edges = spatial_gradient(input, normalized=normalized)
    # Unpack the edges
    gx = edges[:, :, 0]
    gy = edges[:, :, 1]
    # Compute gradient magnitude
    magnitude = torch.sqrt(gx * gx + gy * gy + eps)
    return magnitude


def spatial_gradient(
    input     : Tensor,
    mode      : str  = "sobel",
    order     : int  = 1,
    normalized: bool = True
) -> Tensor:
    """
    Compute the first order image derivative in both x and y using a Sobel
    operator.

    Args:
        input (Tensor): The image of shape [B, C, H, W].
        mode (str): Derivatives modality. One of: [`sobel`, diff`].
            Defaults to sobel.
        order (int): The order of the derivatives. Defaults to 1.
        normalized (bool): Whether the output is normalized. Defaults to True.
    
    Return:
        The derivatives of the input feature map of shape [B, C, 2, H, W].
   
    Examples:
        >>> input  = torch.rand(1, 3, 4, 4)
        >>> output = spatial_gradient(input)  # 1x3x2x4x4
        >>> output.shape
        torch.Size([1, 3, 2, 4, 4])
    """
    assert_tensor_of_ndim(input, 4)
    # Allocate kernel
    kernel = get_spatial_gradient_kernel2d(mode=mode, order=order)
    if normalized:
        kernel = normalize_kernel2d(kernel)

    # Prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel.to(input).detach()
    tmp_kernel = tmp_kernel.unsqueeze(1).unsqueeze(1)

    # Convolve input tensor with sobel kernel
    kernel_flip = tmp_kernel.flip(-3)

    # Pad with "replicate for spatial dims, but with zeros for channel
    spatial_pad  = [kernel.size(1) // 2, kernel.size(1) // 2,
                    kernel.size(2) // 2, kernel.size(2) // 2]
    out_channels = 3 if order == 2 else 2
    padded_inp = F.pad(
        input = input.reshape(b * c, 1, h, w),
        pad   = spatial_pad,
        mode  = "replicate"
    )[:, :, None]
    
    return F.conv3d(
        input   = padded_inp,
        weight  = kernel_flip,
        padding = 0
    ).view(b, c, out_channels, h, w)


def spatial_gradient3d(
    input: Tensor,
    mode : str = "diff",
    order: int = 1
) -> Tensor:
    """
    Compute the first and second order volume derivative in x, y and d using
    a diff operator.
    
    Args:
        input (Tensor): Input features tensor of shape [B, C, D, H, W].
        mode (str): Derivatives modality. One of: [`sobel`, diff`].
            Defaults to sobel.
        order (int): The order of the derivatives. Defaults to 1.
   
    Return:
        The spatial gradients of the input feature map of shape
        [B, C, 3, D, H, W] or [B, C, 6, D, H, W].
   
    Examples:
        >>> input  = torch.rand(1, 4, 2, 4, 4)
        >>> output = spatial_gradient3d(input)
        >>> output.shape
        torch.Size([1, 4, 3, 2, 4, 4])
    """
    assert_tensor_of_ndim(input, 5)
    b, c, d, h, w = input.shape
    dev           = input.device
    dtype         = input.dtype
    
    if (mode == "diff") and (order == 1):
        # We go for the special case implementation due to conv3d bad speed
        x      = F.pad(input=input, pad=6 * [1], mode="replicate")
        center = slice(1, -1)
        left   = slice(0, -2)
        right  = slice(2, None)
        output = torch.empty(b, c, 3, d, h, w, device=dev, dtype=dtype)
        output[..., 0, :, :, :] = x[..., center, center, right] - x[..., center, center, left]
        output[..., 1, :, :, :] = x[..., center, right, center] - x[..., center, left, center]
        output[..., 2, :, :, :] = x[..., right, center, center] - x[..., left, center, center]
        output = 0.5 * output
    else:
        # Prepare kernel
        # Allocate kernel
        kernel     = get_spatial_gradient_kernel3d(mode=mode, order=order)
        tmp_kernel = kernel.to(input).detach()
        tmp_kernel = tmp_kernel.repeat(c, 1, 1, 1, 1)

        # Convolve input tensor with grad kernel
        kernel_flip = tmp_kernel.flip(-3)

        # Pad with "replicate for spatial dims, but with zeros for channel
        spatial_pad = [
            kernel.size(2) // 2,
            kernel.size(2) // 2,
            kernel.size(3) // 2,
            kernel.size(3) // 2,
            kernel.size(4) // 2,
            kernel.size(4) // 2,
        ]
        out_ch = 6 if order == 2 else 3
        output = F.conv3d(
            input   = F.pad(input=input, pad=spatial_pad, mode="replicate"),
            weight  = kernel_flip,
            padding = 0,
            groups  = c
        ).view(b, c, out_ch, d, h, w)
    return output


class BlurPool2D(nn.Module):
    """
    Compute blur (anti-aliasing) and downsample a given feature map.

    Args:
        kernel_size (int): Kernel size for max pooling.
        stride (int): Stride for pooling. Defaults to 2.
    """

    def __init__(self, kernel_size: int, stride: int = 2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride      = stride
        self.register_buffer(
            name   = "kernel",
            tensor = get_pascal_kernel_2d(kernel_size=kernel_size, norm=True)
        )

    def forward(self, input: Tensor) -> Tensor:
        # To align the logic with the whole lib
        kernel = torch.as_tensor(
            data   = self.kernel,
            device = input.device,
            dtype  = input.dtype
        )
        return _blur_pool_by_kernel2d(
            input  = input,
            kernel = kernel.repeat((input.size(1), 1, 1, 1)),
            stride = self.stride
        )


class BoxBlur(nn.Module):
    """
    Blur an image using the box filter. The function smooths an image using the
    kernel:
    
    Args:
        kernel_size (tuple[int, int]): The blurring kernel size.
        border_type (BorderType_): The padding mode to be applied before
            convolving. One of: [`constant`, `reflect`, `replicate`, `circular`].
            Defaults to reflect.
        normalized (bool): If True, L1 norm of the kernel is set to 1.
            Defaults to True.
    """

    def __init__(
        self, 
        kernel_size: tuple[int, int],
        border_type: BorderType_ = "reflect",
        normalized : bool        = True
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.border_type = border_type
        self.normalized  = normalized

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + '(kernel_size='
            + str(self.kernel_size)
            + ', '
            + 'normalized='
            + str(self.normalized)
            + ', '
            + 'border_type='
            + self.border_type
            + ')'
        )

    def forward(self, input: Tensor) -> Tensor:
        return box_blur(
            input       = input,
            kernel_size = self.kernel_size,
            border_type = self.border_type,
            normalized  = self.normalized
        )


class GaussianBlur2d(nn.Module):
    """
    Create an operator that blurs a tensor using a Gaussian filter.
    The operator smooths the given tensor with a gaussian kernel by convolving
    it to each channel. It supports batched operation.
    
    Arguments:
        kernel_size (tuple[int, int]): The blurring kernel size.
        sigma (tuple[float, float]): The standard deviation of the kernel.
        border_type (BorderType_): The padding mode to be applied before
            convolving. One of: [`constant`, `reflect`, `replicate`, `circular`].
            Defaults to reflect.
        separable (bool): Run as composition of two 1d-convolutions.
            Defaults to True.
    """

    def __init__(
        self,
        kernel_size: tuple[int, int],
        sigma      : tuple[float, float],
        border_type: BorderType_ = "reflect",
        separable  : bool        = True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma       = sigma
        self.border_type = border_type
        self.separable   = separable

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + '(kernel_size='
            + str(self.kernel_size)
            + ', '
            + 'sigma='
            + str(self.sigma)
            + ', '
            + 'border_type='
            + self.border_type
            + 'separable='
            + str(self.separable)
            + ')'
        )

    def forward(self, input: Tensor) -> Tensor:
        return gaussian_blur2d(
            input       = input,
            kernel_size = self.kernel_size,
            sigma       = self.sigma,
            border_type = self.border_type,
            separable   = self.separable
        )


class Laplacian(nn.Module):
    """
    Create an operator that returns a tensor using a Laplacian filter.
    The operator smooths the given tensor with a laplacian kernel by convolving
    it to each channel. It supports batched operation.
    
    Args:
        kernel_size (tuple[int, int]): The blurring kernel size.
        border_type (BorderType_): The padding mode to be applied before
            convolving. One of: [`constant`, `reflect`, `replicate`, `circular`].
            Defaults to reflect.
        normalized (bool): If True, L1 norm of the kernel is set to 1.
            Defaults to True.
    """

    def __init__(
        self,
        kernel_size: int,
        border_type: BorderType_ = "reflect",
        normalized : bool        = True
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.border_type = border_type
        self.normalized  = normalized

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + '(kernel_size='
            + str(self.kernel_size)
            + ', '
            + 'normalized='
            + str(self.normalized)
            + ', '
            + 'border_type='
            + self.border_type
            + ')'
        )

    def forward(self, input: Tensor) -> Tensor:
        return laplacian(
            input       = input,
            kernel_size = self.kernel_size,
            border_type = self.border_type,
            normalized  = self.normalized
        )
    
    
class MaxBlurPool2D(nn.Module):
    """
    Compute pools and blurs and downsample a given feature map.
    Equivalent to `nn.Sequential(nn.MaxPool2d(...), BlurPool2D(...))`
   
    Args:
        kernel_size (int): Kernel size for max pooling.
        stride (int): Stride for pooling. Defaults to 2.
        max_pool_size (int): Kernel size for max pooling. Defaults to 2.
        ceil_mode (bool): Should be true to match output size of Conv2d with
            same kernel size. Defaults to False.
    """

    def __init__(
        self,
        kernel_size  : int,
        stride       : int  = 2,
        max_pool_size: int  = 2,
        ceil_mode    : bool = False
    ):
        super().__init__()
        self.kernel_size   = kernel_size
        self.stride        = stride
        self.max_pool_size = max_pool_size
        self.ceil_mode     = ceil_mode
        self.register_buffer(
            name   = "kernel",
            tensor = get_pascal_kernel_2d(kernel_size, norm=True)
        )

    def forward(self, input: Tensor) -> Tensor:
        # To align the logic with the whole lib
        kernel = torch.as_tensor(
            data   = self.kernel,
            device = input.device,
            dtype  = input.dtype
        )
        return _max_blur_pool_by_kernel2d(
            input         = input,
            kernel        = kernel.repeat((input.size(1), 1, 1, 1)),
            stride        = self.stride,
            max_pool_size = self.max_pool_size,
            ceil_mode     = self.ceil_mode
        )


class MedianBlur(nn.Module):
    """
    Blur an image using the median filter.
    
    Args:
        kernel_size (tuple[int, int]): Kernel size for max pooling.
    """

    def __init__(self, kernel_size: tuple[int, int]):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, input: Tensor) -> Tensor:
        return median_blur(input=input, kernel_size=self.kernel_size)


class Sobel(nn.Module):
    """
    Compute the Sobel operator and returns the magnitude per channel.
    
    Args:
        normalized (bool): If True, L1 norm of the kernel is set to 1.
            Defaults to True.
        eps (float): Regularization number to avoid NaN during backprop.
            Defaults to 1e-6.
    """

    def __init__(self, normalized: bool = True, eps: float = 1e-6):
        super().__init__()
        self.normalized = normalized
        self.eps        = eps

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + '(' 'normalized='
            + str(self.normalized)
            + ')'
        )

    def forward(self, input: Tensor) -> Tensor:
        return sobel(input=input, normalized=self.normalized, eps=self.eps)


class SpatialGradient(nn.Module):
    """
    Compute the first order image derivative in both x and y using a Sobel
    operator.
    
    Args:
        mode (str): Derivatives modality. One of: [`sobel`, diff`].
            Defaults to sobel.
        order (int): The order of the derivatives. Defaults to 1.
        normalized (bool): Whether the output is normalized. Defaults to True.
    """

    def __init__(
        self,
        mode      : str  = "sobel",
        order     : int  = 1,
        normalized: bool = True
    ):
        super().__init__()
        self.normalized = normalized
        self.order      = order
        self.mode       = mode

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + '(' 'order='
            +  str(self.order)
            +  ', '
            + 'normalized='
            + str(self.normalized)
            + ', '
            + 'mode='
            + self.mode
            + ')'
        )

    def forward(self, input: Tensor) -> Tensor:
        return spatial_gradient(
            input      = input,
            mode       = self.mode,
            order      = self.order,
            normalized = self.normalized
        )


class SpatialGradient3d(nn.Module):
    """
    Compute the first and second order volume derivative in x, y and d using a diff
    operator.
    
    Args:
        mode (str): Derivatives modality. One of: [`sobel`, diff`].
            Defaults to sobel.
        order (int): The order of the derivatives. Defaults to 1.
    """

    def __init__(self, mode: str = "diff", order: int = 1):
        super().__init__()
        self.order  = order
        self.mode   = mode
        self.kernel = get_spatial_gradient_kernel3d(mode, order)

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + '(' 'order='
            + str(self.order)
            + ', '
            + 'mode='
            + self.mode
            + ')'
        )

    def forward(self, input: Tensor) -> Tensor:
        return spatial_gradient3d(
            input = input,
            mode  = self.mode,
            order = self.order
        )


# H1: - Edge Filter ------------------------------------------------------------

def canny(
    input         : Tensor,
    low_threshold : float               = 0.1,
    high_threshold: float               = 0.2,
    kernel_size   : tuple[int, int]     = (5, 5),
    sigma         : tuple[float, float] = (1, 1),
    hysteresis    : bool                = True,
    eps           : float               = 1e-6,
) -> tuple[Tensor, Tensor]:
    """
    Find edges of the input image and filters them using the Canny algorithm.

    Args:
        input (Tensor): Input image of shape [B, C, H, W].
        low_threshold (float): Lower threshold for the hysteresis procedure.
            Defaults to 0.1.
        high_threshold (float): Upper threshold for the hysteresis procedure.
            Defaults to 0.2.
        kernel_size (tuple[int, int]): Kernel size for the gaussian blur.
            Defaults to (5, 5).
        sigma (tuple[float, float]): Standard deviation of the kernel for the
            gaussian blur. Defaults to (1, 1).
        hysteresis: If True, applies the hysteresis edge tracking. Otherwise,
            the edges are divided between weak (0.5) and strong (1) edges.
            Defaults to True.
        eps (float): Regularization number to avoid NaN during backprop.
            Defaults to 1e-6.
    
    Returns:
        - The canny edge magnitudes map of shape [B, 1, H, W]
        - The canny edge detection filtered by thresholds and hysteresis of
            shape [B, 1, H, W].
    """
    assert_tensor_of_ndim(input, 4)

    if low_threshold > high_threshold:
        raise ValueError(
            f"Expect `low_threshold` < `high_threshold`. "
            f"But got: {low_threshold} > {high_threshold}."
        )
    if 0.0 > low_threshold > 1.0:
        raise ValueError(
            f"`low_threshold` must be in range (0, 1). "
            f"But got: {low_threshold}."
        )
    if 0.0 > high_threshold > 1.0:
        raise ValueError(
            f"`high_threshold` must be in range (0, 1). "
            f"But got: {high_threshold}."
        )

    device = input.device
    dtype  = input.dtype

    # To Grayscale
    if input.shape[1] == 3:
        from one.vision.transformation import rgb_to_grayscale
        input = rgb_to_grayscale(input)

    # Gaussian filter
    blurred   = gaussian_blur2d(input, kernel_size, sigma)

    # Compute the gradients
    gradients = spatial_gradient(blurred, normalized=False)

    # Unpack the edges
    gx = gradients[:, :, 0]
    gy = gradients[:, :, 1]

    # Compute gradient magnitude and angle
    magnitude = torch.sqrt(gx * gx + gy * gy + eps)
    angle     = torch.atan2(gy, gx)

    # Radians to Degrees
    angle = 180.0 * angle / math.pi

    # Round angle to the nearest 45 degree
    angle = torch.round(angle / 45) * 45

    # Non-maximal suppression
    nms_kernels   = get_canny_nms_kernel(device, dtype)
    nms_magnitude = F.conv2d(magnitude, nms_kernels, padding=nms_kernels.shape[-1] // 2)

    # Get the indices for both directions
    positive_idx = (angle / 45) % 8
    positive_idx = positive_idx.long()

    negative_idx = ((angle / 45) + 4) % 8
    negative_idx = negative_idx.long()

    # Apply the non-maximum suppression to the different directions
    channel_select_filtered_positive = torch.gather(nms_magnitude, 1, positive_idx)
    channel_select_filtered_negative = torch.gather(nms_magnitude, 1, negative_idx)

    channel_select_filtered = torch.stack(
        [channel_select_filtered_positive, channel_select_filtered_negative], 1
    )

    is_max    = channel_select_filtered.min(dim=1)[0] > 0.0
    magnitude = magnitude * is_max

    # Threshold
    edges = F.threshold(magnitude, low_threshold, 0.0)
    low   = magnitude > low_threshold
    high  = magnitude > high_threshold
    edges = low * 0.5 + high * 0.5
    edges = edges.to(dtype)

    # Hysteresis
    if hysteresis:
        edges_old          = -torch.ones(edges.shape, device=edges.device, dtype=dtype)
        hysteresis_kernels = get_hysteresis_kernel(device, dtype)

        while ((edges_old - edges).abs() != 0).any():
            weak   = (edges == 0.5).float()
            strong = (edges == 1).float()

            hysteresis_magnitude = F.conv2d(
                input   = edges,
                weight  = hysteresis_kernels,
                padding = hysteresis_kernels.shape[-1] // 2
            )
            hysteresis_magnitude = (hysteresis_magnitude == 1).any(1, keepdim=True).to(dtype)
            hysteresis_magnitude = hysteresis_magnitude * weak + strong

            edges_old = edges.clone()
            edges     = hysteresis_magnitude + (hysteresis_magnitude == 0) * weak * 0.5

        edges = hysteresis_magnitude

    return magnitude, edges


class Canny(nn.Module):
    """
    Module that finds edges of the input image and filters them using the Canny
    algorithm.
    
    Args:
        low_threshold (float): Lower threshold for the hysteresis procedure.
            Defaults to 0.1.
        high_threshold (float): Upper threshold for the hysteresis procedure.
            Defaults to 0.2.
        kernel_size (tuple[int, int]): Kernel size for the gaussian blur.
            Defaults to (5, 5).
        sigma (tuple[float, float]): Standard deviation of the kernel for the
            gaussian blur. Defaults to (1, 1).
        hysteresis: If True, applies the hysteresis edge tracking. Otherwise,
            the edges are divided between weak (0.5) and strong (1) edges.
            Defaults to True.
        eps (float): Regularization number to avoid NaN during backprop.
            Defaults to 1e-6.
    """

    def __init__(
        self,
        low_threshold : float               = 0.1,
        high_threshold: float               = 0.2,
        kernel_size   : tuple[int, int]     = (5, 5),
        sigma         : tuple[float, float] = (1, 1),
        hysteresis    : bool                = True,
        eps           : float               = 1e-6,
    ):
        super().__init__()
        if low_threshold > high_threshold:
            raise ValueError(
                f"Expect `low_threshold` < `high_threshold`. "
                f"But got: {low_threshold} > {high_threshold}."
            )
        if 0.0 > low_threshold > 1.0:
            raise ValueError(
                f"`low_threshold` must be in range (0, 1). "
                f"But got: {low_threshold}."
            )
        if 0.0 > high_threshold > 1.0:
            raise ValueError(
                f"`high_threshold` must be in range (0, 1). "
                f"But got: {high_threshold}."
            )
        # Gaussian blur parameters
        self.kernel_size = kernel_size
        self.sigma       = sigma
        # Double threshold
        self.low_threshold  = low_threshold
        self.high_threshold = high_threshold
        # Hysteresis
        self.hysteresis = hysteresis
        self.eps        = eps

    def __repr__(self) -> str:
        return ''.join(
            (
                f'{type(self).__name__}(',
                ', '.join(
                    f'{name}={getattr(self, name)}' for name in sorted(self.__dict__) if not name.startswith('_')
                ),
                ')',
            )
        )

    def forward(self, input: Tensor) -> tuple[Tensor, Tensor]:
        return canny(
            input          = input,
            low_threshold  = self.low_threshold,
            high_threshold = self.high_threshold,
            kernel_size    = self.kernel_size,
            sigma          = self.sigma,
            hysteresis     = self.hysteresis,
            eps            = self.eps
        )
    
    
# H1: - Sharp Filter -----------------------------------------------------------

def unsharp_mask(
    input      : Tensor,
    kernel_size: tuple[int, int],
    sigma      : tuple[float, float],
    border_type: BorderType_ = "reflect"
) -> Tensor:
    """
    Create an operator that sharpens a tensor by applying operation
    out = 2 * image - gaussian_blur2d(image).
   
    Args:
        input (Tensor): Input image of shape [B, C, H, W].
        kernel_size (tuple[int, int]): The size of the kernel.
        sigma (tuple[float, float]): The standard deviation of the kernel.
        border_type (BorderType_): The padding mode to be applied before
            convolving. One of: [`constant`, `reflect`, `replicate`, `circular`].
            Defaults to reflect.
    
    Returns:
        The sharpened image of shape [B, C, H, W].
        
    Examples:
        >>> input  = torch.rand(2, 4, 5, 5)
        >>> output = unsharp_mask(input, (3, 3), (1.5, 1.5))
        >>> output.shape
        torch.Size([2, 4, 5, 5])
    """
    data_blur      = gaussian_blur2d(input, kernel_size, sigma, border_type)
    data_sharpened = input + (input - data_blur)
    return data_sharpened


class UnsharpMask(nn.Module):
    """
    Create an operator that sharpens image with: out = 2 * image - gaussian_blur2d(image).
    
    Args:
        kernel_size (tuple[int, int]): The size of the kernel.
        sigma (tuple[float, float]): The standard deviation of the kernel.
        border_type (BorderType_): The padding mode to be applied before
            convolving. One of: [`constant`, `reflect`, `replicate`, `circular`].
            Defaults to reflect.
    """

    def __init__(
        self,
        kernel_size: tuple[int, int],
        sigma      : tuple[float, float],
        border_type: BorderType_ = "reflect"
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma       = sigma
        self.border_type = border_type

    def forward(self, input: Tensor) -> Tensor:
        return unsharp_mask(
            input       = input,
            kernel_size = self.kernel_size,
            sigma       = self.sigma,
            border_type = self.border_type
        )
