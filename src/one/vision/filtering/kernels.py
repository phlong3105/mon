#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor

from one.core import Int2T
from one.core import ListOrTuple2T
from one.core import to_2tuple

__all__ = [
    "gaussian",
    "gaussian_discrete",
    "gaussian_discrete_erf",
    "get_binary_kernel2d",
    "get_box_kernel2d",
    "get_canny_nms_kernel",
    "get_diff_kernel2d",
    "get_diff_kernel2d_2nd_order",
    "get_diff_kernel3d",
    "get_diff_kernel3d_2nd_order",
    "get_diff_kernel_3x3",
    "get_gaussian_discrete_kernel1d",
    "get_gaussian_erf_kernel1d",
    "get_gaussian_kernel1d",
    "get_gaussian_kernel2d",
    "get_hysteresis_kernel",
    "get_laplacian_kernel1d",
    "get_laplacian_kernel2d",
    "get_pascal_kernel_1d",
    "get_pascal_kernel_2d",
    "get_sobel_kernel2d",
    "get_sobel_kernel2d_2nd_order",
    "get_sobel_kernel_3x3",
    "get_sobel_kernel_5x5_2nd_order",
    "get_spatial_gradient_kernel2d",
    "get_spatial_gradient_kernel3d",
    "laplacian_1d",
    "normalize_kernel2d"
]


# MARK: - Functional

def get_binary_kernel2d(window_size: Int2T) -> Tensor:
    """Create a binary kernel to extract the patches. If the window size is
    [H, W] will create a (H*W)xHxW kernel.
    """
    window_size  = to_2tuple(window_size)
    window_range = window_size[0] * window_size[1]
    kernel       = torch.zeros(window_range, window_range)
    for i in range(window_range):
        kernel[i, i] += 1.0
    return kernel.view(window_range, 1, window_size[0], window_size[1])


def get_box_kernel2d(kernel_size: Int2T) -> Tensor:
    """Utility function that returns a box filter."""
    kernel_size = to_2tuple(kernel_size)
    kx          = float(kernel_size[0])
    ky          = float(kernel_size[1])
    scale       = torch.tensor(1.0) / torch.tensor([kx * ky])
    tmp_kernel  = torch.ones(1, kernel_size[0], kernel_size[1])
    return scale.to(tmp_kernel.dtype) * tmp_kernel


def get_canny_nms_kernel(device=torch.device("cpu"), dtype=torch.float) -> Tensor:
    """Utility function that returns 3x3 kernels for the Canny Non-maximal
    suppression.
    """
    kernel = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        device = device,
        dtype  = dtype,
    )
    return kernel.unsqueeze(1)


def gaussian(window_size: int, sigma: float) -> Tensor:
    device, dtype = None, None
    if isinstance(sigma, Tensor):
        device, dtype = sigma.device, sigma.dtype
    x = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    if window_size % 2 == 0:
        x += 0.5
    gauss = torch.exp(-x.pow(2.0) / (2 * sigma ** 2))
    return gauss / gauss.sum()


def gaussian_discrete_erf(window_size: int, sigma) -> Tensor:
    """Discrete Gaussian by interpolating the error function. Adapted from:
    https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py
    """
    device = sigma.device if isinstance(sigma, Tensor) else None
    sigma  = torch.as_tensor(sigma, dtype=torch.float, device=device)
    x      = torch.arange(window_size).float() - window_size // 2
    t      = 0.70710678 / torch.abs(sigma)
    gauss  = 0.5 * ((t * (x + 0.5)).erf() - (t * (x - 0.5)).erf())
    gauss  = gauss.clamp(min=0)
    return gauss / gauss.sum()


def _modified_bessel_0(x: Tensor) -> Tensor:
    """Adapted from:
    https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py
    """
    if torch.abs(x) < 3.75:
        y = (x / 3.75) * (x / 3.75)
        return 1.0 + y * (
            3.5156229 + y * (
                3.0899424 + y * (
                    1.2067492 + y *
                        (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2))
                )
            )
        )
    ax   = torch.abs(x)
    y    = 3.75 / ax
    ans  = 0.916281e-2 + y * (
        -0.2057706e-1 + y * (
            0.2635537e-1 + y * (-0.1647633e-1 + y * 0.392377e-2)
        )
    )
    coef = 0.39894228 + y * (
        0.1328592e-1 + y * (0.225319e-2 + y * (-0.157565e-2 + y * ans))
    )
    return (torch.exp(ax) / torch.sqrt(ax)) * coef


def _modified_bessel_1(x: Tensor) -> Tensor:
    """Adapted from:
    https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py
    """
    if torch.abs(x) < 3.75:
        y   = (x / 3.75) * (x / 3.75)
        ans = 0.51498869 + y * (
            0.15084934 + y * (
                0.2658733e-1 + y * (0.301532e-2 + y * 0.32411e-3)
            )
        )
        return torch.abs(x) * (0.5 + y * (0.87890594 + y * ans))
    ax  = torch.abs(x)
    y   = 3.75 / ax
    ans = 0.2282967e-1 + y * (
        -0.2895312e-1 + y * (0.1787654e-1 - y * 0.420059e-2)
    )
    ans = 0.39894228 + y * (
        -0.3988024e-1 + y * (
            -0.362018e-2 + y * (0.163801e-2 + y * (-0.1031555e-1 + y * ans))
        )
    )
    ans = ans * torch.exp(ax) / torch.sqrt(ax)
    return -ans if x < 0.0 else ans


def _modified_bessel_i(n: int, x: Tensor) -> Tensor:
    """Adapted from:
    https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py
    """
    if n < 2:
        raise ValueError("n must be greater than 1.")
    if x == 0.0:
        return x
    device = x.device
    tox = 2.0 / torch.abs(x)
    ans = torch.tensor(0.0, device=device)
    bip = torch.tensor(0.0, device=device)
    bi  = torch.tensor(1.0, device=device)
    m   = int(2 * (n + int(sqrt(40.0 * n))))
    for j in range(m, 0, -1):
        bim = bip + float(j) * tox * bi
        bip = bi
        bi  = bim
        if abs(bi) > 1.0e10:
            ans *= 1.0e-10
            bi  *= 1.0e-10
            bip *= 1.0e-10
        if j == n:
            ans = bip
    ans = ans * _modified_bessel_0(x) / bi
    return -ans if x < 0.0 and (n % 2) == 1 else ans


def gaussian_discrete(window_size, sigma) -> Tensor:
    """Discrete Gaussian kernel based on the modified Bessel functions.
    Adapted from:
    https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py
    """
    device     = sigma.device if isinstance(sigma, Tensor) else None
    sigma      = torch.as_tensor(sigma, dtype=torch.float, device=device)
    sigma2     = sigma * sigma
    tail       = int(window_size // 2)
    out_pos    = [None] * (tail + 1)
    out_pos[0] = _modified_bessel_0(sigma2)
    out_pos[1] = _modified_bessel_1(sigma2)
    for k in range(2, len(out_pos)):
        out_pos[k] = _modified_bessel_i(k, sigma2)
    out = out_pos[:0:-1]
    out.extend(out_pos)
    out = torch.stack(out) * torch.exp(sigma2)  # type: ignore
    return out / out.sum()  # type: ignore


def get_gaussian_discrete_kernel1d(
    kernel_size: int, sigma: float, force_even: bool = False
) -> Tensor:
    """Function that returns Gaussian filter coefficients based on the modified
    Bessel functions. Adapted from:
    https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py

    Args:
        kernel_size (int):
            Filter size. It should be odd and positive.
        sigma (float):
            Gaussian standard deviation.
        force_even (bool):
            Overrides requirement for odd kernel size.

    Returns:
        windown_1d (Tensor):
            1D image with gaussian filter coefficients.

    Examples:
        >>> get_gaussian_discrete_kernel1d(3, 2.5)
        image([0.3235, 0.3531, 0.3235])

        >>> get_gaussian_discrete_kernel1d(5, 1.5)
        image([0.1096, 0.2323, 0.3161, 0.2323, 0.1096])
    """
    if (
        not isinstance(kernel_size, int) or
        ((kernel_size % 2 == 0) and not force_even) or
        (kernel_size <= 0)
    ):
        raise TypeError(f"kernel_size must be an odd positive integer. "
                        f"Got: {kernel_size}")
    window_1d = gaussian_discrete(kernel_size, sigma)
    return window_1d


def get_gaussian_erf_kernel1d(
    kernel_size: int, sigma: float, force_even: bool = False
) -> Tensor:
    """Function that returns Gaussian filter coefficients by interpolating the
    error function, adapted from:
    https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py

    Args:
        kernel_size (int):
            Filter size. It should be odd and positive.
        sigma (float):
            Gaussian standard deviation.
        force_even (bool):
            Overrides requirement for odd kernel size.

    Returns:
        window_1d (Tensor):
            1D image with gaussian filter coefficients.

    Examples:
        >>> get_gaussian_erf_kernel1d(3, 2.5)
        image([0.3245, 0.3511, 0.3245])

        >>> get_gaussian_erf_kernel1d(5, 1.5)
        image([0.1226, 0.2331, 0.2887, 0.2331, 0.1226])
    """
    if (
        not isinstance(kernel_size, int) or
        ((kernel_size % 2 == 0) and not force_even) or
        (kernel_size <= 0)
    ):
        raise TypeError(f"kernel_size must be an odd positive integer. "
                        f"Got: {kernel_size}")
    window_1d = gaussian_discrete_erf(kernel_size, sigma)
    return window_1d


def get_gaussian_kernel1d(
    kernel_size: int, sigma: float, force_even: bool = False
) -> Tensor:
    """Function that returns Gaussian filter coefficients.

    Args:
        kernel_size (int):
            Filter size. It should be odd and positive.
        sigma (float):
            Gaussian standard deviation.
        force_even (bool):
            Overrides requirement for odd kernel size.

    Returns:
        window_1d (Tensor):
            1D image with gaussian filter coefficients.

    Examples:
        >>> get_gaussian_kernel1d(3, 2.5)
        image([0.3243, 0.3513, 0.3243])

        >>> get_gaussian_kernel1d(5, 1.5)
        image([0.1201, 0.2339, 0.2921, 0.2339, 0.1201])
    """
    if (
        not isinstance(kernel_size, int) or
        ((kernel_size % 2 == 0) and not force_even) or
        (kernel_size <= 0)
    ):
        raise TypeError(f"kernel_size must be an odd positive integer."
                        f"Got: {kernel_size}")
    window_1d = gaussian(kernel_size, sigma)
    return window_1d


def get_gaussian_kernel2d(
    kernel_size: Int2T, sigma: ListOrTuple2T[float], force_even: bool = False
) -> Tensor:
    """Function that returns Gaussian filter matrix coefficients.

    Args:
        kernel_size (Int2T):
            Filter sizes in the x and y direction. Sizes should be odd and
            positive.
        sigma (ListOrTuple2T[float]):
            Gaussian standard deviation in the x and y direction.
        force_even (bool):
            Overrides requirement for odd kernel size.

    Returns:
         kernel_2d (Tensor):
            2D image with gaussian filter matrix coefficients.

    Examples:
        >>> get_gaussian_kernel2d((3, 3), (1.5, 1.5))
        image([[0.0947, 0.1183, 0.0947],
                [0.1183, 0.1478, 0.1183],
                [0.0947, 0.1183, 0.0947]])
        >>> get_gaussian_kernel2d((3, 5), (1.5, 1.5))
        image([[0.0370, 0.0720, 0.0899, 0.0720, 0.0370],
                [0.0462, 0.0899, 0.1123, 0.0899, 0.0462],
                [0.0370, 0.0720, 0.0899, 0.0720, 0.0370]])
    """
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise TypeError(f"kernel_size must be a tuple of length two. "
                        f"Got: {kernel_size}")
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError(f"sigma must be a tuple of length two. Got: {sigma}")
    kernel_size      = to_2tuple(kernel_size)
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma
    kernel_x  = get_gaussian_kernel1d(ksize_x, sigma_x, force_even)
    kernel_y  = get_gaussian_kernel1d(ksize_y, sigma_y, force_even)
    kernel_2d = torch.matmul(kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
    return kernel_2d


def get_hysteresis_kernel(device=torch.device("cpu"), dtype=torch.float) -> Tensor:
    """Utility function that returns the 3x3 kernels for the Canny hysteresis.
    """
    kernel = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        device=device,
        dtype=dtype,
    )
    return kernel.unsqueeze(1)


def laplacian_1d(window_size) -> Tensor:
    """One could also use the Laplacian of Gaussian formula to design the
    filter.
    """
    filter_1d                   = torch.ones(window_size)
    filter_1d[window_size // 2] = 1 - window_size
    _laplacian_1d               = filter_1d
    return _laplacian_1d


def get_laplacian_kernel1d(kernel_size: int) -> Tensor:
    """Function that returns the coefficients of a 1D Laplacian filter.

    Args:
        kernel_size (int):
            Filter size. It should be odd and positive.

    Returns:
        window_1d (Tensor):
            1D image with laplacian filter coefficients.

    Examples:
        >>> get_laplacian_kernel1d(3)
        image([ 1., -2.,  1.])
        >>> get_laplacian_kernel1d(5)
        image([ 1.,  1., -4.,  1.,  1.])
    """
    if (not isinstance(kernel_size, int) or
        kernel_size % 2 == 0 or
        kernel_size <= 0):
        raise TypeError(f"ksize must be an odd positive integer. "
                        f"Got: {kernel_size}")
    window_1d = laplacian_1d(kernel_size)
    return window_1d


def get_laplacian_kernel2d(kernel_size: int) -> Tensor:
    """Function that returns Gaussian filter matrix coefficients.

    Args:
        kernel_size (int):
            Filter size. It should be odd and positive.

    Returns:
        window_1d (Tensor):
            2D image with laplacian filter matrix coefficients.

    Examples:
        >>> get_laplacian_kernel2d(3)
        image([[ 1.,  1.,  1.],
                [ 1., -8.,  1.],
                [ 1.,  1.,  1.]])
        >>> get_laplacian_kernel2d(5)
        image([[  1.,   1.,   1.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.],
                [  1.,   1., -24.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.]])
    """
    if (not isinstance(kernel_size, int) or
        kernel_size % 2 == 0 or
        kernel_size <= 0):
        raise TypeError(f"ksize must be an odd positive integer. "
                        f"Got: {kernel_size}")
    
    kernel           = torch.ones((kernel_size, kernel_size))
    mid              = kernel_size // 2
    kernel[mid, mid] = 1 - kernel_size ** 2
    kernel_2d        = kernel
    return kernel_2d


def normalize_kernel2d(image: Tensor) -> Tensor:
    """Normalize both derivative and smoothing kernel."""
    if len(image.size()) < 2:
        raise TypeError(f"input should be at least 2D image. "
                        f"Got: {image.size()}")
    
    norm = image.abs().sum(dim=-1).sum(dim=-1)
    return image / (norm.unsqueeze(-1).unsqueeze(-1))


def get_pascal_kernel_2d(kernel_size: int, norm: bool = True) -> Tensor:
    """Generate pascal filter kernel by kernel size.

    Args:
        kernel_size (int):
            Height and width of the kernel.
        norm (bool):
            If to normalize the kernel or not. Default: `True`.

    Returns:
        filt (Tensor):
            Kernel shaped as [kernel_size, kernel_size].

    Examples:
    >>> get_pascal_kernel_2d(1)
    image([[1.]])
    >>> get_pascal_kernel_2d(4)
    image([[0.0156, 0.0469, 0.0469, 0.0156],
            [0.0469, 0.1406, 0.1406, 0.0469],
            [0.0469, 0.1406, 0.1406, 0.0469],
            [0.0156, 0.0469, 0.0469, 0.0156]])
    >>> get_pascal_kernel_2d(4, norm=False)
    image([[1., 3., 3., 1.],
            [3., 9., 9., 3.],
            [3., 9., 9., 3.],
            [1., 3., 3., 1.]])
    """
    a    = get_pascal_kernel_1d(kernel_size)
    filt = a[:, None] * a[None, :]
    if norm:
        filt /= torch.sum(filt)
    return filt


def get_pascal_kernel_1d(kernel_size: int, norm: bool = False) -> Tensor:
    """Generate Yang Hui triangle (Pascal"s triangle) by a given number.

    Args:
        kernel_size (int):
            Height and width of the kernel.
        norm (bool):
            If to normalize the kernel or not. Default: `False`.

    Returns:
        out (Tensor):
            Kernel shaped as [kernel_size, ].

    Examples:
    >>> get_pascal_kernel_1d(1)
    image([1.])
    >>> get_pascal_kernel_1d(2)
    image([1., 1.])
    >>> get_pascal_kernel_1d(3)
    image([1., 2., 1.])
    >>> get_pascal_kernel_1d(4)
    image([1., 3., 3., 1.])
    >>> get_pascal_kernel_1d(5)
    image([1., 4., 6., 4., 1.])
    >>> get_pascal_kernel_1d(6)
    image([ 1.,  5., 10., 10.,  5.,  1.])
    """
    pre = []
    cur = []
    for i in range(kernel_size):
        cur = [1.0] * (i + 1)

        for j in range(1, i // 2 + 1):
            value  = pre[j - 1] + pre[j]
            cur[j] = value
            if i != 2 * j:
                cur[-j - 1] = value
        pre = cur

    out = torch.as_tensor(cur)
    if norm:
        out /= torch.sum(out)
    return out


def get_sobel_kernel_3x3() -> Tensor:
    """Utility function that returns a sobel kernel of [3, 3]."""
    return torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])


def get_sobel_kernel_5x5_2nd_order() -> Tensor:
    """Utility function that returns a 2nd order sobel kernel of [5, 5]."""
    return torch.tensor(
        [
            [-1.0, 0.0,  2.0, 0.0, -1.0],
            [-4.0, 0.0,  8.0, 0.0, -4.0],
            [-6.0, 0.0, 12.0, 0.0, -6.0],
            [-4.0, 0.0,  8.0, 0.0, -4.0],
            [-1.0, 0.0,  2.0, 0.0, -1.0],
        ]
    )


def _get_sobel_kernel_5x5_2nd_order_xy() -> Tensor:
    """Utility function that returns a 2nd order sobel kernel of [5, 5]."""
    return torch.tensor(
        [
            [-1.0, -2.0, 0.0,  2.0,  1.0],
            [-2.0, -4.0, 0.0,  4.0,  2.0],
            [ 0.0,  0.0, 0.0,  0.0,  0.0],
            [ 2.0,  4.0, 0.0, -4.0, -2.0],
            [ 1.0,  2.0, 0.0, -2.0, -1.0],
        ]
    )


def get_sobel_kernel2d() -> Tensor:
    kernel_x = get_sobel_kernel_3x3()
    kernel_y = kernel_x.transpose(0, 1)
    return torch.stack([kernel_x, kernel_y])


def get_sobel_kernel2d_2nd_order() -> Tensor:
    gxx = get_sobel_kernel_5x5_2nd_order()
    gyy = gxx.transpose(0, 1)
    gxy = _get_sobel_kernel_5x5_2nd_order_xy()
    return torch.stack([gxx, gxy, gyy])


def get_diff_kernel_3x3() -> Tensor:
    """Utility function that returns a first order derivative kernel of
    [3, 3].
    """
    return torch.tensor([[-0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [-0.0, 0.0, 0.0]])


def get_diff_kernel3d(device=torch.device("cpu"), dtype=torch.float) -> Tensor:
    """Utility function that returns a first order derivative kernel of
    [3, 3, 3].
    """
    kernel = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [-0.5, 0.0, 0.5], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]],
            ],
        ],
        device=device,
        dtype=dtype,
    )
    return kernel.unsqueeze(1)


def get_diff_kernel3d_2nd_order(
    device=torch.device("cpu"), dtype=torch.float
) -> Tensor:
    """Utility function that returns a first order derivative kernel of
    [3, 3, 3].
    """
    kernel = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, -1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
            ],
        ],
        device=device,
        dtype=dtype,
    )
    return kernel.unsqueeze(1)


def get_diff_kernel2d() -> Tensor:
    kernel_x = get_diff_kernel_3x3()
    kernel_y = kernel_x.transpose(0, 1)
    return torch.stack([kernel_x, kernel_y])


def get_diff_kernel2d_2nd_order() -> Tensor:
    gxx = torch.tensor([[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]])
    gyy = gxx.transpose(0, 1)
    gxy = torch.tensor([[-1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, -1.0]])
    return torch.stack([gxx, gxy, gyy])


def get_spatial_gradient_kernel2d(mode: str, order: int) -> Tensor:
    """Function that returns kernel for 1st or 2nd order image gradients,
    using one of the following operators: `sobel`, `diff`.
    """
    if mode not in ["sobel", "diff"]:
        raise TypeError(f"mode should be either sobel or diff. Got: {mode}")
    if order not in [1, 2]:
        raise TypeError(f"order should be either 1 or 2. Got: {order}")
    if mode == "sobel" and order == 1:
        kernel = get_sobel_kernel2d()
    elif mode == "sobel" and order == 2:
        kernel = get_sobel_kernel2d_2nd_order()
    elif mode == "diff" and order == 1:
        kernel = get_diff_kernel2d()
    elif mode == "diff" and order == 2:
        kernel = get_diff_kernel2d_2nd_order()
    else:
        raise NotImplementedError("")
    return kernel


def get_spatial_gradient_kernel3d(
    mode: str, order: int, device=torch.device("cpu"), dtype=torch.float
) -> Tensor:
    """Function that returns kernel for 1st or 2nd order scale pyramid
    gradients, using one of the following operators: `sobel`, `diff`.
    """
    if mode not in ["sobel", "diff"]:
        raise TypeError(f"mode should be either sobel or diff. Got: {mode}")
    if order not in [1, 2]:
        raise TypeError(f"order should be either 1 or 2. Got: {order}")
    if mode == "sobel":
        raise NotImplementedError("Sobel kernel for 3d gradient is not "
                                  "implemented yet")
    if mode == "diff" and order == 1:
        kernel = get_diff_kernel3d(device, dtype)
    elif mode == "diff" and order == 2:
        kernel = get_diff_kernel3d_2nd_order(device, dtype)
    else:
        raise NotImplementedError("")
    return kernel
