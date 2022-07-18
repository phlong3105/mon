#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Operations on Tensor.
"""

from __future__ import annotations

import inspect
import sys
from typing import Any

import numpy as np
import torch
from multipledispatch import dispatch
from torch import Tensor
from torch.linalg import solve

from one.core.collection import is_list_of
from one.core.rich import error_console
from one.core.version import torch_version_geq


# MARK: - Functional

def _to_3d(input: Tensor) -> Tensor:
    """Convert Tensor to 3D shape [C, H, W].
    
    Args:
        input (Tensor):
            Tensor of arbitrary dimensions.
            
    Returns:
        input (Tensor):
            Tensor of shape [1, H, W] or [C, H, W].
    """
    if not isinstance(input, Tensor):
        raise TypeError(f"`input` must be a `Tensor`. But got: {type(input)}.")
   
    if input.ndim == 2:  # [H, W] -> [1, H, W]
        input = input.unsqueeze(dim=0)
    elif input.ndim == 3:  # [C, H, W]
        pass
    elif input.ndim == 4 and input.shape[0] == 1:  # [1, C, H, W] -> [C, H, W]
        input = input.squeeze(dim=0)
    else:
        raise ValueError(
            f"Require 2 <= `input.ndim` <= 4. But got: {input.ndim}."
        )
    return input


def _to_4d(input: Tensor) -> Tensor:
    """Convert Tensor to 4D shape [B, C, H, W].
    
    Args:
        input (Tensor):
            Tensor of arbitrary dimensions.
            
    Returns:
        input (Tensor):
            Tensor of shape [1, C, H, W] or [B, C, H, W].
    """
    if not isinstance(input, Tensor):
        raise TypeError(f"`input` must be a `Tensor`. But got: {type(input)}.")
   
    if input.ndim == 2:  # [H, W] -> [1, 1, H, W]
        input = input.unsqueeze(dim=0)
        input = input.unsqueeze(dim=0)
    elif input.ndim == 3:  # [C, H, W] -> [1, C, H, W]
        input = input.unsqueeze(dim=0)
    elif input.ndim == 4:  # [B, C, H, W]
        pass
    elif input.ndim == 5 and input.shape[0] == 1:
        input = input.squeeze(dim=0)
    else:
        raise ValueError(f"Require 2 <= `input.ndim` <= 5. But got: {input.ndim}.")
    return input


def _to_5d(input: Tensor) -> Tensor:
    """Convert Tensor to 5D shape [*, B, C, H, W].
    
    Args:
        input (Tensor):
            Tensor of arbitrary dimensions.
            
    Returns:
        input (Tensor):
            Tensor of shape [1, B, C, H, W] or [*, B, C, H, W].
    """
    if not isinstance(input, Tensor):
        raise TypeError(f"`input` must be a `Tensor`. But got: {type(input)}.")
    
    if input.ndim == 2:  # [H, W] -> [1, 1, 1, H, W]
        input = input.unsqueeze(dim=0)
        input = input.unsqueeze(dim=0)
        input = input.unsqueeze(dim=0)
    if input.ndim == 3:  # [C, H, W] -> [1, 1, C, H, W]
        input = input.unsqueeze(dim=0)
        input = input.unsqueeze(dim=0)
    if input.ndim == 4:  # [B, C, H, W] -> [1, B, C, H, W]
        input = input.unsqueeze(dim=0)
    if input.ndim == 5:  # [*, B, C, H, W]
        pass
    if input.ndim == 6 and input.shape[0] == 1:
        input = input.squeeze(dim=0)
    else:
        raise ValueError(f"Require 2 <= `input.ndim` <= 6. But got: {input.ndim}.")
    return input


def eye_like(n: int, input: Tensor) -> Tensor:
    """Return a 2-D image with ones on the diagonal and zeros elsewhere with
    the same batch size as the input.

    Args:
        n (int):
            Number of rows [N].
        input (Tensor):
            Tensor that will determine the batch size of the output matrix.
            The expected shape is [B, *].

    Returns:
        identity (Tensor):
            The identity matrix with the same batch size as the input [B, N, N].

    """
    if not n > 0:
        raise ValueError(f"Require `n` > 0. But got: {n}.")
    if not input.ndim >= 1:
        raise ValueError(f"Require `input.ndim` >= 1. But got: {input.ndim}.")

    identity = torch.eye(n, device=input.device, dtype=input.dtype)
    return identity[None].repeat(input.shape[0], 1, 1)


def histc_cast(input: Tensor, bins: int, min: int, max: int) -> Tensor:
    """Helper function to make torch.histc work with other than fp32/64.

    The function torch.histc is only implemented for fp32/64 which makes
    impossible to be used by fp16 or others. What this function does, is cast
    input data type to fp32, apply torch.inverse, and cast back to the input
    dtype.
    """
    if not isinstance(input, Tensor):
        raise TypeError(f"`input` must be a `Tensor`. But got: {type(input)}.")
    dtype = input.dtype
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float32
    return torch.histc(input.to(dtype), bins, min, max).to(input.dtype)


def inverse_cast(input: Tensor) -> Tensor:
    """Helper function to make torch.inverse work with other than fp32/64.

	The function torch.inverse is only implemented for fp32/64 which makes
	impossible to be used by fp16 or others. What this function does, is cast
	input data type to fp32, apply torch.inverse, and cast back to the input
	dtype.
	"""
    if not isinstance(input, Tensor):
        raise TypeError(f"`input` must be a `Tensor`. But got: {type(input)}.")
    dtype = input.dtype
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float32
    return torch.inverse(input.to(dtype)).to(input.dtype)


def safe_solve_with_mask(B: Tensor, A: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Helper function, which avoids crashing because of singular matrix input
    and outputs the mask of valid solution.
    """
    if not torch_version_geq(1, 10):
        sol, lu = solve_cast(B, A)
        error_console.log("PyTorch version < 1.10, solve validness mask maybe "
                          "not correct", RuntimeWarning)
        return sol, lu, torch.ones(len(A), dtype=torch.bool, device=A.device)
    # Based on https://github.com/pytorch/pytorch/issues/31546#issuecomment
    # -694135622
    if not isinstance(B, Tensor):
        raise TypeError(f"`B` must be a `Tensor`. But got: {type(B)}.")
    
    dtype = B.dtype
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float32
    A_LU, pivots, info = torch.lu(A.to(dtype), get_infos=True)
    valid_mask         = info == 0
    X                  = torch.lu_solve(B.to(dtype), A_LU, pivots)
    return X.to(B.dtype), A_LU.to(A.dtype), valid_mask


def safe_inverse_with_mask(A: Tensor) -> tuple[Tensor, Tensor]:
    r"""Helper function, which avoids crashing because of non-invertable matrix
    input and outputs the mask of valid solution.
    """
    # Based on https://github.com/pytorch/pytorch/issues/31546#issuecomment
    # -694135622
    if not torch_version_geq(1, 9):
        inv = inverse_cast(A)
        error_console.log("PyTorch version < 1.9, inverse validness mask maybe "
                          "not correct", RuntimeWarning)
        return inv, torch.ones(len(A), dtype=torch.bool, device=A.device)
    if isinstance(A, Tensor):
        raise TypeError(f"`A` must be a `Tensor`. But got: {type(A)}.")
   
    dtype_original = A.dtype
    if dtype_original not in (torch.float32, torch.float64):
        dtype = torch.float32
    else:
        dtype = dtype_original
    from torch.linalg import inv_ex  # type: ignore # (not available in 1.8.1)
    inverse, info = inv_ex(A.to(dtype))
    mask          = info == 0
    return inverse.to(dtype_original), mask


def solve_cast(input: Tensor, A: Tensor) -> tuple[Tensor, Tensor]:
    """Helper function to make torch.solve work with other than fp32/64.

    The function torch.solve is only implemented for fp32/64 which makes
    impossible to be used by fp16 or others. What this function does, is cast
    input data type to fp32, apply torch.svd, and cast back to the input dtype.
    """
    if not isinstance(input, Tensor):
        raise TypeError(f"`input` must be a `Tensor`. But got: {type(input)}.")
    dtype = input.dtype
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float32
    
    out = solve(A.to(dtype), input.to(dtype))
    return out.to(input.dtype), out


def svd_cast(input: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Helper function to make torch.svd work with other than fp32/64.

    The function torch.svd is only implemented for fp32/64 which makes
    impossible to be used by fp16 or others. What this function does, is cast
    input data type to fp32, apply torch.svd, and cast back to the input dtype.

    NOTE: in torch 1.8.1 this function is recommended to use as torch.linalg.svd
    """
    if not isinstance(input, Tensor):
        raise TypeError(f"`input` must be a `Tensor`. But got: {type(input)}.")
    dtype = input.dtype
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float32
    
    out1, out2, out3 = torch.svd(input.to(dtype))
    return out1.to(input.dtype), out2.to(input.dtype), out3.to(input.dtype)


def to_3d_tensor_list(input: Any) -> list[Tensor]:
    """Convert to a list of 3D Tensors.
   
    Args:
        input (Any):
            Input of arbitrary type.
            
    Returns:
        input (list[Tensor]):
            List of 3D Tensors of shape [C, H, W].
    """
    if isinstance(input, dict):
        input = list(input.values())
    if isinstance(input, tuple):
        input = list(input)
    if isinstance(input, np.ndarray):
        input = torch.from_numpy(input)
    if isinstance(input, Tensor):
        if input.ndim == 3:
            input = [input.unsqueeze(dim=0)]
        elif input.ndim == 4:
            input = list(input)
        else:
            raise ValueError(f"Require 3 <= `input.ndim` <= 4. But got: {input.ndim}.")
    if isinstance(input, list) and is_list_of(input, np.ndarray):
        input = [torch.from_numpy(i) for i in input]
    if isinstance(input, list) and is_list_of(input, Tensor):
        if all(i.ndim == 3 for i in input):
            return input
        else:
            raise ValueError(f"Require all `input.ndim` == 3.")
    raise TypeError(f"`input` must be a `Tensor`. But got: {type(input)}.")


def to_4d_tensor(input: Any) -> Tensor:
    """Convert to a 4D Tensor.
   
    Args:
        input (Any):
            Input of arbitrary type.
            
    Returns:
        input (Tensor):
            A 4D Tensor of shape [B, C, H, W].
    """
    if isinstance(input, dict):
        input = list(input.values())
    if isinstance(input, tuple):
        input = list(input)
    if isinstance(input, list) and is_list_of(input, np.ndarray):
        input = [torch.from_numpy(i) for i in input]
    if isinstance(input, list) and is_list_of(input, Tensor):
        if all(i.ndim == 2 for i in input):
            input = [i.unsqueeze(dim=0) for i in input]
        elif all(i.ndim == 3 for i in input):
            input = torch.stack(input, dim=0)
        else:
            raise ValueError(f"Require 2 <= `input.ndim` <= 3.")
    if isinstance(input, np.ndarray):
        input = torch.from_numpy(input)
    if isinstance(input, Tensor):
        return _to_4d(input)
    raise TypeError(f"`input` must be a `Tensor`. But got: {type(input)}.")
    

def to_4d_tensor_list(input) -> list[Tensor]:
    """Convert to a list of 4D Tensors.
   
    Args:
        input (Any):
            Input of arbitrary type.
            
    Returns:
        input (list[Tensor]):
            List of 4D Tensors of shape [B, C, H, W].
    """
    if isinstance(input, dict):
        input = list(input.values())
    if isinstance(input, tuple):
        input = list(input)
    if isinstance(input, np.ndarray):
        input = torch.from_numpy(input)
    if isinstance(input, Tensor):
        if input.ndim == 3:
            input = [input.unsqueeze(dim=0)]
        elif input.ndim == 4:
            input = [input]
        elif input.ndim == 5:
            input = list(input)
        else:
            raise ValueError(f"Require 3 <= `input.ndim` <= 5.")
    if isinstance(input, list) and is_list_of(input, np.ndarray):
        input = [torch.from_numpy(i) for i in input]
    if isinstance(input, list) and is_list_of(input, Tensor):
        if all(i.ndim == 3 for i in input):
            return [torch.stack(input, dim=0)]
        elif all(i.ndim == 4 for i in input):
            return input
        else:
            raise ValueError(f"Require 3 <= `input.ndim` <= 4.")
    raise TypeError(f"`input` must be a `Tensor`. But got: {type(input)}.")


def to_5d_tensor(input) -> Tensor:
    """Convert to a 5D Tensor.
   
    Args:
        input (Any):
            Input of arbitrary type.
            
    Returns:
        input (Tensor):
            5D Tensor of shape [*, B, C, H, W].
    """
    if isinstance(input, dict):
        input = list(input.values())
    if isinstance(input, tuple):
        input = list(input)
    if isinstance(input, list) and is_list_of(input, np.ndarray):
        input = [torch.from_numpy(i) for i in input]
    if isinstance(input, list) and is_list_of(input, Tensor):
        if all(i.ndim == 2 for i in input):
            input = [i.unsqueeze(dim=0) for i in input]
        if all(3 <= i.ndim <= 4 for i in input):
            input = torch.stack(input, dim=0)
        else:
            raise ValueError(f"Require 2 <= `input.ndim` <= 4.")
    if isinstance(input, np.ndarray):
        input = torch.from_numpy(input)
    if isinstance(input, Tensor):
        return _to_5d(input)
    raise TypeError(f"`input` must be a `Tensor`. But got: {type(input)}.")


@dispatch(Tensor)
def upcast(input: Tensor) -> Tensor:
    """Protects from numerical overflows in multiplications by upcasting to
    the equivalent higher type.
    
    Args:
        input (Tensor):
            Tensor of arbitrary type.
    
    Returns:
        input (Tensor):
            Tensor of higher type.
    """
    if not isinstance(input, Tensor):
        raise TypeError(f"`input` must be a `Tensor`. But got: {type(input)}.")
    if input.dtype in (torch.float16, torch.float32, torch.float64):
        return input.to(torch.float)
    if input.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        return input.to(torch.int)
    return input
    

def vec_like(n: int, input: Tensor):
    """Return a 2D image with a vector containing zeros with the same batch
    size as the input.

    Args:
        n (int):
            Number of rows [N].
        input (Tensor):
            Tensor that will determine the batch size of the output matrix.
            The expected shape is [B, *].

    Returns:
        (Tensor):
            The vector with the same batch size as the input [B, N, 1].

    """
    if not n > 0:
        raise ValueError(f"Require `n` > 0. But got: {n}.")
    if not input.ndim >= 1:
        raise ValueError(f"Require `input.ndim` >= 1. But got: {input.ndim}.")

    vec = torch.zeros(n, 1, device=input.device, dtype=input.dtype)
    return vec[None].repeat(input.shape[0], 1, 1)


# MARK: - Main

__all__ = [
    name for name, value in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__
    )
]
