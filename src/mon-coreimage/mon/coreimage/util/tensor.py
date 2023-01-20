#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module extends the functionalities of :class:`torch.Tensor` that are
meant to use in images.

Notes:
    We put these functions here instead of :mod:`mon.foundation` because we may
    develop similar functions for, let's say, :mod:`mon.audio` or
    :mod:`mon.text` package with slightly different operation.
"""

from __future__ import annotations

__all__ = [
    "eye_like", "is_same_shape", "to_3d_tensor", "to_4d_tensor", "to_5d_tensor",
    "to_list_of_3d_tensor", "to_list_of_4d_tensor", "upcast", "vec_like",
]

from typing import Any

import multipledispatch
import numpy as np
import torch


# region Non-mutating Operations

@multipledispatch.dispatch(torch.Tensor, torch.Tensor)
def is_same_shape(input1: torch.Tensor, input2: torch.Tensor) -> bool:
    """Return True if two tensors have the same shape. Otherwise, return False.
    """
    return input1.shape == input2.shape

# endregion


# region Mutating Operations

def _to_3d_tensor(input: torch.Tensor) -> torch.Tensor:
    """Convert a 2D or 4D tensor to a 3D tensor.
    
    If the input is a 2D tensor, add a new axis at the beginning.
    If the input is a 4D tensor with the first dimension being 1, remove the
    first dimension.
    
    Args:
        input: An input tensor.
    
    Return:
        A 3D tensor of shape [C, H, W].
    """
    assert isinstance(input, torch.Tensor)
    ndim = input.ndim
    assert 2 <= ndim <= 4
    if ndim == 2:    # [H, W] -> [1, H, W]
        input = input.unsqueeze(dim=0)
    elif ndim == 3:  # [C, H, W]
        pass
    elif ndim == 4 and input.shape[0] == 1:  # [1, C, H, W] -> [C, H, W]
        input = input.squeeze(dim=0)
    return input


def _to_4d_tensor(input: torch.Tensor) -> torch.Tensor:
    """Convert a 2D, 3D, or 5D tensor to a 4D tensor.
    
    If the input is a 2D tensor, add 2 new axes at the beginning.
    If the input is a 3D tensor, add a new axis at the beginning.
    If the input is a 5D tensor with the first dimension being 1, remove the
    first dimension.
    
    Args:
        input: An input tensor.
    
    Return:
        A 4D tensor of shape [B, C, H, W].
    """
    assert isinstance(input, torch.Tensor)
    ndim = input.ndim
    assert 2 <= ndim <= 4
    if ndim == 2:    # [H, W] -> [1, 1, H, W]
        input = input.unsqueeze(dim=0)
        input = input.unsqueeze(dim=0)
    elif ndim == 3:  # [C, H, W] -> [1, C, H, W]
        input = input.unsqueeze(dim=0)
    elif ndim == 4:  # [B, C, H, W]
        pass
    elif ndim == 5 and input.shape[0] == 1:
        input = input.squeeze(dim=0)  # [1, C, B, H, W] -> [B, C, H, W]
    return input


def _to_5d_tensor(input: torch.Tensor) -> torch.Tensor:
    """Convert a 2D, 3D, 4D, or 6D tensor to a 5D tensor.
    
    If the input is a 2D tensor, add 3 new axes at the beginning.
    If the input is a 3D tensor, add 2 new axes at the beginning.
    If the input is a 4D tensor, add a new axis at the beginning.
    If the input is a 6D tensor with the first dimension being 1, remove the
    first dimension.
    
    Args:
        input: An input tensor.
    
    Return:
        A 5D tensor of shape [*, B, C, H, W].
    """
    assert isinstance(input, torch.Tensor)
    ndim = input.ndim
    assert 2 <= ndim <= 6
    if ndim == 2:    # [H, W] -> [1, 1, 1, H, W]
        input = input.unsqueeze(dim=0)
        input = input.unsqueeze(dim=0)
        input = input.unsqueeze(dim=0)
    elif ndim == 3:  # [C, H, W] -> [1, 1, C, H, W]
        input = input.unsqueeze(dim=0)
        input = input.unsqueeze(dim=0)
    elif ndim == 4:  # [B, C, H, W] -> [1, B, C, H, W]
        input = input.unsqueeze(dim=0)
    elif ndim == 5:  # [*, B, C, H, W]
        pass
    elif ndim == 6 and input.shape[0] == 1:
        input = input.squeeze(dim=0)  # [1, *, B, C, H, W] -> [*, B, C, H, W]
    return input


def eye_like(n: int, input: torch.Tensor) -> torch.Tensor:
    """Create a tensor of shape `(n, n)` with ones on the diagonal and zeros
    everywhere else, and then repeats it along the batch dimension to match the
    shape of the input tensor.
    
    Args:
        n: The number of rows and columns in the output tensor.
        input: An input tensor.
    
    Return:
        A tensor of shape (input.shape[0], n, n).
    """
    if not n > 0:
        raise ValueError(f"Expect `n` > 0. But got: {n}.")
    assert isinstance(input, torch.Tensor) and input.ndim >= 1
    identity = torch.eye(n, device=input.device, dtype=input.dtype)
    return identity[None].repeat(input.shape[0], 1, 1)


def to_3d_tensor(input: Any) -> torch.Tensor:
    """Convert an arbitrary input to a 3D tensor.
   
    Args:
        input: An input of arbitrary type.
        
    Return:
        A 3D tensor of shape [C, H, W].
    """
    if isinstance(input, dict):
        input = list(input.values())                                            # dict -> list[Tensor | np.ndarray]
    if isinstance(input, tuple):
        input = list(input)                                                     # tuple -> list[Tensor | np.ndarray]
    if isinstance(input, list) and all(isinstance(i, np.ndarray)   for i in input):
        input = [torch.from_numpy(i) for i in input]                            # list[np.ndarray] -> list[Tensor]
    if isinstance(input, list) and all(isinstance(i, torch.Tensor) for i in input):
        if all(i.ndim == 2 for i in input):
            input = torch.stack(input, dim=0)                                   # list[2D Tensor] -> 3D Tensor
        else:
            raise ValueError(f"Expect 2 <= `input.ndim` <= 3.")
    if isinstance(input, np.ndarray):
        input = torch.from_numpy(input)                                         # np.ndarray -> Tensor any dimensions
    if isinstance(input, torch.Tensor):
        return _to_3d_tensor(input)                                             # Tensor any dimensions -> 3D Tensor
    raise TypeError(
        f"`input` must be a `torch.Tensor`. But got: {type(input)}."
    )


def to_4d_tensor(input: Any) -> torch.Tensor:
    """Convert an arbitrary input to a 4D tensor.
   
    Args:
        input: An input of arbitrary type.
        
    Return:
        A 4D tensor of shape [B, C, H, W].
    """
    if isinstance(input, dict):
        input = list(input.values())                                            # dict -> list[Tensor | np.ndarray]
    if isinstance(input, tuple):
        input = list(input)                                                     # tuple -> list[Tensor | np.ndarray]
    if isinstance(input, list) and all(isinstance(i, np.ndarray)   for i in input):
        input = [torch.from_numpy(i) for i in input]                            # list[np.ndarray] -> list[Tensor]
    if isinstance(input, list) and all(isinstance(i, torch.Tensor) for i in input):
        if all(i.ndim == 2 for i in input):
            input = [i.unsqueeze(dim=0) for i in input]                         # list[2D Tensor] -> list[3D Tensor]
        if all(i.ndim == 3 for i in input):
            input = torch.stack(input, dim=0)                                   # list[3D Tensor] -> 4D Tensor
        else:
            raise ValueError(f"Expect 2 <= `input.ndim` <= 3.")
    if isinstance(input, np.ndarray):
        input = torch.from_numpy(input)                                         # np.ndarray -> Tensor any dimensions
    if isinstance(input, torch.Tensor):
        return _to_4d_tensor(input)                                             # Tensor any dimensions -> 4D Tensor
    raise TypeError(
        f"`input` must be a `torch.Tensor`. But got: {type(input)}."
    )


def to_5d_tensor(input: Any) -> torch.Tensor:
    """Convert an arbitrary input to a 5D tensor.
   
    Args:
        input: An input of arbitrary type.
        
    Return:
        A 5D tensor of shape [*, B, C, H, W].
    """
    if isinstance(input, dict):
        input = list(input.values())                                            # dict -> list[Tensor | np.ndarray]
    if isinstance(input, tuple):
        input = list(input)                                                     # tuple -> list[Tensor | np.ndarray]
    if isinstance(input, list) and all(isinstance(i, np.ndarray)   for i in input):
        input = [torch.from_numpy(i) for i in input]                            # list[np.ndarray] -> list[Tensor]
    if isinstance(input, list) and all(isinstance(i, torch.Tensor) for i in input):
        if all(i.ndim == 2 for i in input):
            input = [i.unsqueeze(dim=0) for i in input]                         # list[2D Tensor] -> list[3D Tensor]
        if all(3 <= i.ndim <= 4 for i in input):
            input = torch.stack(input, dim=0)                                   # list[3D Tensor] -> 4D Tensor
        else:
            raise ValueError(f"Expect 2 <= `input.ndim` <= 4.")
    if isinstance(input, np.ndarray):
        input = torch.from_numpy(input)                                         # np.ndarray -> Tensor any dimensions
    if isinstance(input, torch.Tensor):
        return _to_5d_tensor(input)                                             # Tensor any dimensions -> 5D Tensor
    raise TypeError(
        f"`input` must be a `torch.Tensor`. But got: {type(input)}."
    )


def to_list_of_3d_tensor(input: Any) -> list[torch.Tensor]:
    """Convert arbitrary input to a list of 3D tensors.
   
    Args:
        input: An input of arbitrary type.
        
    Return:
        A list of 3D tensors of shape [C, H, W].
    """
    if isinstance(input, dict):
        input = list(input.values())                                            # dict -> list[Tensor | np.ndarray]
    if isinstance(input, tuple):
        input = list(input)                                                     # tuple -> list[Tensor | np.ndarray]
    if isinstance(input, np.ndarray):
        input = torch.from_numpy(input)                                         # np.ndarray -> Tensor any dimension
    if isinstance(input, torch.Tensor):
        if input.ndim == 3:
            input = [input]                                                     # Tensor -> list[3D Tensor]
        elif input.ndim == 4:
            input = list(input)                                                 # Tensor -> list[3D Tensor]
        else:
            raise ValueError(
                f"Expect 3 <= `input.ndim` <= 4. But got: {input.ndim}."
            )
    if isinstance(input, list) and all(isinstance(i, np.ndarray)   for i in input):
        input = [torch.from_numpy(i) for i in input]                            # list[np.ndarray] -> list[Tensor any dimensions]
    if isinstance(input, list) and all(isinstance(i, torch.Tensor) for i in input):
        return [_to_3d_tensor(i) for i in input]                                # list[Tensor any dimensions] -> list[3D Tensor]
    raise TypeError(f"Cannot convert `input` to a list of 3D tensor.")


def to_list_of_4d_tensor(input: Any) -> list[torch.Tensor]:
    """Convert arbitrary input to a list of 4D tensors.
   
    Args:
        input: An input of arbitrary type.
        
    Return:
        A list of 3D tensors of shape [B, C, H, W].
    """
    if isinstance(input, dict):
        input = list(input.values())                                            # dict -> list[Tensor | np.ndarray]
    if isinstance(input, tuple):
        input = list(input)                                                     # tuple -> list[Tensor | np.ndarray]
    if isinstance(input, np.ndarray):
        input = torch.from_numpy(input)                                         # np.ndarray -> Tensor any dimension
    if isinstance(input, torch.Tensor):
        if input.ndim == 3:
            input = [input.unsqueeze(dim=0)]                                    # Tensor -> list[4D Tensor]
        elif input.ndim == 4:
            input = [input]                                                     # Tensor -> list[4D Tensor]
        elif input.ndim == 5:
            input = list(input)                                                 # Tensor -> list[4D Tensor]
        else:
            raise ValueError(
                f"Expect 3 <= `input.ndim` <= 5. But got: {input.ndim}."
            )
    if isinstance(input, list) and all(isinstance(i, np.ndarray)   for i in input):
        input = [torch.from_numpy(i) for i in input]                            # list[np.ndarray] -> list[Tensor any dimensions]
    if isinstance(input, list) and all(isinstance(i, torch.Tensor) for i in input):
        return [_to_4d_tensor(i) for i in input]                                # list[Tensor any dimensions] -> list[3D Tensor]
    raise TypeError(f"Cannot convert `input` to a list of 4D tensor.")


def upcast(input: torch.Tensor) -> torch.Tensor:
    """Protect from numerical overflows in multiplications by upcasting to the
    equivalent higher type.

    Args:
        input: A tensor of arbitrary type.

    Return:
        A tensor of higher type.
    """
    assert isinstance(input, torch.Tensor)
    if input.dtype in (torch.float16, torch.float32, torch.float64):
        return input.to(torch.float)
    if input.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        return input.to(torch.int)
    return input


def vec_like(n: int, input: torch.Tensor) -> torch.Tensor:
    """Create a vector of zeros with the same shape as the input.

    Args:
        n: The number of elements in the vector.
        input: An input tensor.
    
    Return:
        A tensor of zeros with the same shape as the input tensor.
    """
    if not n > 0:
        raise ValueError(f"Expect `n` > 0. But got: {n}.")
    assert isinstance(input, torch.Tensor) and input.ndim >= 1
    vec = torch.zeros(n, 1, device=input.device, dtype=input.dtype)
    return vec[None].repeat(input.shape[0], 1, 1)

# endregion
