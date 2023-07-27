#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements Gaussian filters.
"""

from __future__ import annotations

__all__ = [
    "gaussian",
    "gaussian_blur2d",
    "get_gaussian_kernel1d",
    "get_gaussian_kernel2d",
    "get_gaussian_kernel3d",
]

from typing import Any

import torch

import mon.core
from mon import nn
from mon.nn import _size_2_t, _size_3_t

console = mon.foundation.console


# region Kernel

def gaussian(
    window_size: int,
    sigma      : torch.Tensor | float,
    *,
    device     : Any = None,
    dtype      : Any = None,
) -> torch.Tensor:
    """Compute the gaussian values based on the window and sigma values.

    Args:
        window_size: The size which drives the filter amount.
        sigma: Gaussian standard deviation. If a tensor, should be in a shape
            :math:`(B, 1)`.
        device: This value will be used if sigma is a float. Device desired to
            compute.
        dtype: This value will be used if sigma is a float. Dtype desired for
            compute.
    
    Returns:
        A tensor with shape :math:`(B, \text{kernel_size})`, with Gaussian values.
    """
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]], device=device, dtype=dtype)

    batch_size = sigma.shape[0]
    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x += 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))
    return gauss / gauss.sum(-1, keepdim=True)


def get_gaussian_kernel1d(
    kernel_size: int,
    sigma      : float | torch.Tensor,
    force_even : bool = False,
    *,
    device     : Any  = None,
    dtype      : Any  = None,
) -> torch.Tensor:
    r"""Function that returns Gaussian filter coefficients.

    Args:
        kernel_size: Filter size. It should be odd and positive.
        sigma: Gaussian standard deviation.
        force_even: Overrides requirement for odd kernel size.
        device: This value will be used if sigma is a float. Device desired to
            compute.
        dtype: This value will be used if sigma is a float. Dtype desired for
            compute.

    Returns:
        Gaussian filter coefficients with shape :math:`(B, \text{kernel_size})`.

    Examples:
        >>> get_gaussian_kernel1d(3, 2.5)
        tensor([[0.3243, 0.3513, 0.3243]])
        >>> get_gaussian_kernel1d(5, 1.5)
        tensor([[0.1201, 0.2339, 0.2921, 0.2339, 0.1201]])
        >>> get_gaussian_kernel1d(5, torch.tensor([[1.5], [0.7]]))
        tensor([[0.1201, 0.2339, 0.2921, 0.2339, 0.1201],
                [0.0096, 0.2054, 0.5699, 0.2054, 0.0096]])
    """
    nn.check_kernel_size(kernel_size=kernel_size, allow_even=force_even)
    return gaussian(window_size=kernel_size, sigma=sigma, device=device, dtype=dtype)


def get_gaussian_kernel2d(
    kernel_size: _size_2_t,
    sigma      : tuple[float, float] | torch.Tensor,
    force_even : bool = False,
    *,
    device     : Any  = None,
    dtype      : Any  = None,
) -> torch.Tensor:
    r"""Function that returns Gaussian filter matrix coefficients.

    Args:
        kernel_size: Filter sizes in the y and x direction. Sizes should be odd
            and positive.
        sigma: Gaussian standard deviation in the y and x.
        force_even: Overrides requirement for odd kernel size.
        device: This value will be used if sigma is a float. Device desired to
            compute.
        dtype: This value will be used if sigma is a float. Dtype desired for
            compute.

    Returns:
        2D tensor with gaussian filter matrix coefficients.

    Shape:
        - Output: :math:`(B, \text{kernel_size}_x, \text{kernel_size}_y)`

    Examples:
        >>> get_gaussian_kernel2d((5, 5), (1.5, 1.5))
        tensor([[[0.0144, 0.0281, 0.0351, 0.0281, 0.0144],
                 [0.0281, 0.0547, 0.0683, 0.0547, 0.0281],
                 [0.0351, 0.0683, 0.0853, 0.0683, 0.0351],
                 [0.0281, 0.0547, 0.0683, 0.0547, 0.0281],
                 [0.0144, 0.0281, 0.0351, 0.0281, 0.0144]]])
        >>> get_gaussian_kernel2d((3, 5), (1.5, 1.5))
        tensor([[[0.0370, 0.0720, 0.0899, 0.0720, 0.0370],
                 [0.0462, 0.0899, 0.1123, 0.0899, 0.0462],
                 [0.0370, 0.0720, 0.0899, 0.0720, 0.0370]]])
        >>> get_gaussian_kernel2d((5, 5), torch.tensor([[1.5, 1.5]]))
        tensor([[[0.0144, 0.0281, 0.0351, 0.0281, 0.0144],
                 [0.0281, 0.0547, 0.0683, 0.0547, 0.0281],
                 [0.0351, 0.0683, 0.0853, 0.0683, 0.0351],
                 [0.0281, 0.0547, 0.0683, 0.0547, 0.0281],
                 [0.0144, 0.0281, 0.0351, 0.0281, 0.0144]]])
    """
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], device=device, dtype=dtype)
    
    ksize_y, ksize_x = nn.to_2d_kernel_size(kernel_size=kernel_size)
    sigma_y, sigma_x = sigma[:, 0, None], sigma[:, 1, None]

    kernel_y = get_gaussian_kernel1d(
        kernel_size = ksize_y,
        sigma       = sigma_y,
        force_even  = force_even,
        device      = device,
        dtype       = dtype,
    )[..., None]
    kernel_x = get_gaussian_kernel1d(
        kernel_size = ksize_x,
        sigma       = sigma_x,
        force_even  = force_even,
        device      = device,
        dtype       = dtype,
    )[..., None]
    
    return kernel_y * kernel_x.view(-1, 1, ksize_x)


def get_gaussian_kernel3d(
    kernel_size: _size_3_t,
    sigma      : tuple[float, float, float] | torch.Tensor,
    force_even : bool = False,
    *,
    device     : Any  = None,
    dtype      : Any  = None,
) -> torch.Tensor:
    r"""Function that returns Gaussian filter matrix coefficients.

    Args:
        kernel_size: Filter sizes in the z, y and x direction. Sizes should be
            odd and positive.
        sigma: Gaussian standard deviation in the z, y and x direction.
        force_even: Overrides requirement for odd kernel size.
        device: This value will be used if sigma is a float. Device desired to
            compute.
        dtype: This value will be used if sigma is a float. Dtype desired for
            compute.

    Returns:
        3D tensor with gaussian filter matrix coefficients.

    Shape:
        - Output: :math:`(B, \text{kernel_size}_x, \text{kernel_size}_y, \text{kernel_size}_z)`

    Examples:
        >>> get_gaussian_kernel3d((3, 3, 3), (1.5, 1.5, 1.5))
        tensor([[[[0.0292, 0.0364, 0.0292],
                  [0.0364, 0.0455, 0.0364],
                  [0.0292, 0.0364, 0.0292]],
        <BLANKLINE>
                 [[0.0364, 0.0455, 0.0364],
                  [0.0455, 0.0568, 0.0455],
                  [0.0364, 0.0455, 0.0364]],
        <BLANKLINE>
                 [[0.0292, 0.0364, 0.0292],
                  [0.0364, 0.0455, 0.0364],
                  [0.0292, 0.0364, 0.0292]]]])
        >>> get_gaussian_kernel3d((3, 3, 3), (1.5, 1.5, 1.5)).sum()
        tensor(1.)
        >>> get_gaussian_kernel3d((3, 3, 3), (1.5, 1.5, 1.5)).shape
        torch.Size([1, 3, 3, 3])
        >>> get_gaussian_kernel3d((3, 7, 5), torch.tensor([[1.5, 1.5, 1.5]])).shape
        torch.Size([1, 3, 7, 5])
    """
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], device=device, dtype=dtype)

    ksize_z, ksize_y, ksize_x = nn.to_3d_kernel_size(kernel_size)
    sigma_z, sigma_y, sigma_x = sigma[:, 0, None], sigma[:, 1, None], sigma[:, 2, None]

    kernel_z = get_gaussian_kernel1d(
        kernel_size = ksize_z,
        sigma       = sigma_z,
        force_even  = force_even,
        device      = device,
        dtype       = dtype,
    )
    kernel_y = get_gaussian_kernel1d(
        kernel_size = ksize_y,
        sigma       = sigma_y,
        force_even  = force_even,
        device      = device,
        dtype       = dtype,
    )
    kernel_x = get_gaussian_kernel1d(
        kernel_size = ksize_x,
        sigma       = sigma_x,
        force_even  = force_even,
        device      = device,
        dtype       = dtype,
    )

    return kernel_z.view(-1, ksize_z, 1, 1) \
        * kernel_y.view(-1, 1, ksize_y, 1) \
        * kernel_x.view(-1, 1, 1, ksize_x)

# endregion


# region Gaussian Filter

def gaussian_blur2d(
    input      : torch.Tensor,
    kernel_size: _size_2_t,
    sigma      : tuple[float, float] | torch.Tensor,
    border_type: str  = "reflect",
    separable  : bool = True,
) -> torch.Tensor:
    r"""Create an operator that blurs a tensor using a Gaussian filter. The
    operator smooths the given tensor with a Gaussian kernel by convolving it to
    each channel. It supports batched operation.
    
    Arguments:
        input: The input tensor with shape :math:`[B, C, H, W]`.
        kernel_size: The size of the kernel.
        sigma: The standard deviation of the kernel.
        border_type: The padding mode to be applied before convolving.
            The expected modes are: ``'constant'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        separable: Run as composition of two 1d-convolutions.

    Returns:
        The blurred tensor with shape :math:`[B, C, H, W]`.

    Notes:
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       gaussian_blur.html>`__.

    Examples:
        >>> input  = torch.rand(2, 4, 5, 5)
        >>> output = gaussian_blur2d(input, (3, 3), (1.5, 1.5))
        >>> output.shape
        torch.Size([2, 4, 5, 5])

        >>> output = gaussian_blur2d(input, (3, 3), torch.tensor([[1.5, 1.5]]))
        >>> output.shape
        torch.Size([2, 4, 5, 5])
    """
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], device=input.device, dtype=input.dtype)
    else:
        sigma = sigma.to(device=input.device, dtype=input.dtype)

    if separable:
        ky, kx   = nn.to_2d_kernel_size(kernel_size=kernel_size)
        bs       = sigma.shape[0]
        kernel_x = get_gaussian_kernel1d(kernel_size=kx, sigma=sigma[:, 1].view(bs, 1))
        kernel_y = get_gaussian_kernel1d(kernel_size=ky, sigma=sigma[:, 0].view(bs, 1))
        output   = filter2d_separable(input, kernel_x, kernel_y, border_type)
    else:
        kernel   = get_gaussian_kernel2d(kernel_size=kernel_size, sigma=sigma)
        output   = filter2d(input, kernel, border_type)

    return output

# endregion
