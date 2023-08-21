#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements utility functions for :mod:`mon.nn`. It extends Python
:mod:`torch.nn.utils` module.
"""

from __future__ import annotations

from typing import Sequence

import multipledispatch
import numpy as np
import torch
# noinspection PyUnresolvedReferences
from torch.nn.utils import *

from mon.core import console
from mon.nn.typing import _size_2_t, _size_3_t, _size_any_t


# region Assert

def check_kernel_size(
    kernel_size: _size_any_t,
    min_value  : int  = 0,
    allow_even : bool = False,
):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,)
    
    fmt = "even or odd" if allow_even else "odd"
    for size in kernel_size:
        assert isinstance(size, int) and (((size % 2 == 1) or allow_even) and size > min_value), \
            f"`kernel_size` must be an {fmt} integer bigger than {min_value}. " \
            f"Gotcha {size} on {kernel_size}."


def check_same_device(
    x     : torch.Tensor,
    y     : torch.Tensor,
    raises: bool = True
) -> bool:
    """Check whether two tensors in the same device.

    Args:
        x: First tensor to evaluate.
        y: Sencod tensor to evaluate.
        raises: Whether an exception should be raised upon failure.

    Raises:
        TypeException: If the two tensors are not in the same device and raises
            is ``True``.

    Example:
        >>> x1 = torch.rand(2, 3, 3)
        >>> x2 = torch.rand(1, 3, 1)
        >>> check_same_device(x1, x2)
        True
    """
    if x.device != y.device:
        if raises:
            raise TypeError(
                f"Not same device for tensors, but got: {x.device} != {y.device}"
            )
        return False
    return True


def check_same_devices(input: list[torch.Tensor], raises: bool = True) -> bool:
    """Check whether a :class:`list` provided tensors live in the same device.

    Args:
        input: A :class:`list` of tensors.
        raises: Whether an exception should be raised upon failure.

    Raises:
        Exception: If all the tensors are not in the same device and raises is
            ``True``.

    Example:
        >>> x1 = torch.rand(2, 3, 3)
        >>> x2 = torch.rand(1, 3, 1)
        >>> check_same_devices([x1, x2], "Tensors not in the same device")
        True
    """
    assert isinstance(input, list) and len(input) >= 1
    if not all(input[0].device == x.device for x in tensors):
        if raises:
            raise Exception(
                f"Not same device for tensors, but got: "
                f"{[x.device for x in tensors]}.\n{msg}"
            )
        return False
    return True

def check_shape(
    input : torch.Tensor | np.ndarray,
    shape : Sequence[str],
    raises: bool = True
) -> bool:
    """Check whether an :param:`input` has a specified shape.

    The shape can be specified with a implicit or explicit :class:`list` of
    :class:`str`. The guard also check whether the variable is a valid type.

    Args:
        input: The input to evaluate.
        shape: A :class:`Sequence` with :class:`str` with the expected shape.
        raises: A bool indicating whether an exception should be raised upon
            failure. Default: ``True``.

    Raises:
        Exception: If the input tensor is has not the expected shape and raises
            is ``True``.

    Example:
        >>> x = torch.rand(2, 3, 4, 4)
        >>> check_shape(x, ["B", "C", "H", "W"])  # implicit
        True
    
        >>> x = torch.rand(2, 3, 4, 4)
        >>> check_shape(x, ["2", "3", "H", "W"])  # explicit
        True
    """
    x = input
    if "*" == shape[0]:
        shape_to_check   = shape[1:]
        x_shape_to_check = x.shape[-len(shape) + 1 :]
    elif "*" == shape[-1]:
        shape_to_check   = shape[:-1]
        x_shape_to_check = x.shape[: len(shape) - 1]
    else:
        shape_to_check   = shape
        x_shape_to_check = x.shape

    if len(x_shape_to_check) != len(shape_to_check):
        if raises:
            raise TypeError(f"{x} shape must be [{shape}], but got {x.shape}.")
        else:
            return False

    for i in range(len(x_shape_to_check)):
        dim_: str = shape_to_check[i]
        if not dim_.isnumeric():
            continue
        dim = int(dim_)
        if x_shape_to_check[i] != dim:
            if raises:
                raise TypeError(
                    f"{x} shape must be [{shape}], but got {x.shape}."
                )
            else:
                return False
    return True

# endregion


# region Convert

def get_padding(kernel_size: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    """Compute padding as a :class:`tuple` of 4 or 6 :class:`int`:
    ``(padding_left, padding_right, padding_top, padding_bottom)``.
    
    References:
        `<https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad>`__
    """
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]
    
    # For even kernels we need to do asymmetric padding
    padding  = 2 * len(kernel_size) * [0]
    
    for i in range(len(kernel_size)):
        computed_tmp       = computed[-(i + 1)]
        pad_front          = computed_tmp // 2
        pad_rear           = computed_tmp - pad_front
        padding[2 * i + 0] = pad_front
        padding[2 * i + 1] = pad_rear

    return tuple(padding)


def to_2d_kernel_size(
    kernel_size: _size_2_t,
    behaviour  : str  = "corr",
) -> tuple[int, int]:
    """Return a 2-D kernel size.
    
    Args:
        kernel_size: The kernel size.
        behaviour: Defines the convolution mode -- correlation (default), using
            PyTorch :class:`torch.nn.Conv2d`, or true convolution (kernel is
            flipped). 2 modes available ``'corr'`` or ``'conv'``.
    """
    if isinstance(kernel_size, int):
        ky = kx = kernel_size
    else:
        assert len(kernel_size) == 2, \
            (f"2-D kernel size should have a length of 2, "
             f"but got: { len(kernel_size)}.")
        ky, kx = kernel_size
    
    ky = int(ky)
    kx = int(kx)
    
    if str(behaviour).lower() == "corr":
        return kx, ky
    else:
        return ky, kx


def to_3d_kernel_size(
    kernel_size: _size_3_t,
    behaviour  : str  = "corr",
) -> tuple[int, int, int]:
    """Return a 3-D kernel size.
    
    Args:
        kernel_size: The kernel size.
        behaviour: Defines the convolution mode -- correlation (default), using
            PyTorch :class:`torch.nn.Conv2d`, or true convolution (kernel is
            flipped). 2 modes available ``'corr'`` or ``'conv'``.
    """
    if isinstance(kernel_size, int):
        kz = ky = kx = kernel_size
    else:
        assert len(kernel_size) == 3, \
            (f"2-D kernel size should have a length of 3, "
             f"but got: { len(kernel_size)}.")
        kz, ky, kx = kernel_size
    
    kz = int(kz)
    ky = int(ky)
    kx = int(kx)
    
    if str(behaviour).lower() == "corr":
        return kx, ky, kz
    else:
        return kz, ky, kx


def upcast(
    input    : torch.Tensor | np.ndarray,
    keep_type: bool = False
) -> torch.Tensor | np.ndarray:
    """Protect from numerical overflows in multiplications by upcasting to the
    equivalent higher type.
    
    Args:
        input: An input of type :class:`numpy.ndarray` or :class:`torch.Tensor`.
        keep_type: If True, keep the same type (int32  -> int64). Else upcast to
            a higher type (int32 -> float32).
            
    Return:
        An image of higher type.
    """
    if input.dtype is torch.float16:
        return input.to(torch.float32)
    elif input.dtype is torch.float32:
        return input  # x.to(torch.float64)
    elif input.dtype is torch.int8:
        return input.to(torch.int16) if keep_type else input.to(torch.float16)
    elif input.dtype is torch.int16:
        return input.to(torch.int32) if keep_type else input.to(torch.float32)
    elif input.dtype is torch.int32:
        return input  # x.to(torch.int64) if keep_type else x.to(torch.float64)
    elif type(input) is np.float16:
        return input.astype(np.float32)
    elif type(input) is np.float32:
        return input  # x.astype(np.float64)
    elif type(input) is np.int16:
        return input.astype(np.int32) if keep_type else input.astype(np.float32)
    elif type(input) is np.int32:
        return input  # x.astype(np.int64) if keep_type else x.astype(np.int64)
    return input

# endregion


# region Create

@multipledispatch.dispatch(int, torch.Tensor)
def eye_like(n: int, input: torch.Tensor) -> torch.Tensor:
    """Create a tensor of shape :math:`[n, n]` with ones on the diagonal and
    zeros everywhere else, and then repeats it along the batch dimension to
    match the shape of the input tensor.

    Args:
        n: The number of rows and columns in the output tensor.
        input: An input tensor.

    Return:
        A tensor of shape :math:`[input.shape[0], n, n]`.
    """
    if not input.ndim >= 1:
        raise ValueError(
            f"x's number of dimensions must be >= 1, but got {input.ndim}."
        )
    if not n > 0:
        raise ValueError(f"n must be > 0, but got {n}.")
    identity = torch.eye(n, device=input.device, dtype=input.dtype)
    return identity[None].repeat(input.shape[0], 1, 1)


@multipledispatch.dispatch(int, torch.Tensor)
def vec_like(n: int, input: torch.Tensor) -> torch.Tensor:
    """Create a vector of zeros with the same shape as the input.
    
    Args:
        n: The number of elements in the vector.
        input: An input tensor.

    Return:
        A tensor of zeros with the same shape as the input tensor.
    """
    if not input.ndim >= 1:
        raise ValueError(
            f"x's number of dimensions must be >= 1, but got {input.ndim}."
        )
    if not n > 0:
        raise ValueError(f"n must be > 0, but got {n}.")
    vec = torch.zeros(n, 1, device=input.device, dtype=input.dtype)
    return vec[None].repeat(input.shape[0], 1, 1)

# endregion
