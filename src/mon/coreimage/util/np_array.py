#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module extends the functionalities of :class:`numpy.ndarray` that are
meant to use in images.
"""

from __future__ import annotations

__all__ = [
    "is_same_shape", "to_3d_array", "to_4d_array", "to_5d_array",
    "to_list_of_3d_array", "to_list_of_4d_array", "upcast",
]

from typing import Any

import multipledispatch
import numpy as np
import torch


# region Non-mutating Operations

@multipledispatch.dispatch(np.ndarray, np.ndarray)
def is_same_shape(input1: np.ndarray, input2: np.ndarray) -> bool:
    """Return True if two arrays have the same shape. Otherwise, return False.
    """
    return input1.shape == input2.shape

# endregion


# region Mutating Operations

def _to_3d_array(input: np.ndarray) -> np.ndarray:
    """Convert a 2D or 4D array to a 3D array.
    
    If the input is a 2D array, add a new axis at the beginning.
    If the input is a 4D array with the first dimension being 1, remove the
    first dimension.
    
    Args:
        input: An input array.
    
    Return:
        A 3D array of shape [H, W, C].
    """
    assert isinstance(input, np.ndarray)
    ndim = input
    assert 2 <= ndim <= 4
    if ndim == 2:    # [H, W] -> [1, H, W]
        input = np.expand_dims(input, axis=0)
    elif ndim == 3:  # [H, W, C]
        pass
    elif ndim == 4 and input.shape[0] == 1:  # [1, H, W, C] -> [H, W, C]
        input = np.squeeze(input, axis=0)
    return input


def _to_4d_array(input: np.ndarray) -> np.ndarray:
    """Convert a 2D, 3D, or 5D array to a 4D array.
    
    If the input is a 2D array, add 2 new axes at the beginning.
    If the input is a 3D array, add a new axis at the beginning.
    If the input is a 5D array with the first dimension being 1, remove the
    first dimension.
    
    Args:
        input: An input array.
    
    Return:
        A 4D array of shape [B, H, W, C].
    """
    assert isinstance(input, np.ndarray)
    ndim = input.ndim
    assert 2 <= ndim <= 5
    if input.ndim == 2:    # [H, W] -> [1, 1, H, W]
        input = np.expand_dims(input, axis=0)
        input = np.expand_dims(input, axis=0)
    elif input.ndim == 3:  # [H, W, C] -> [1, H, W, C]
        input = np.expand_dims(input, axis=0)
    elif input.ndim == 4:  # [B, H, W, C]
        pass
    elif input.ndim == 5 and input.shape[0] == 1:
        input = np.squeeze(input, axis=0)  # [1, B, H, W, C] -> [B, H, W, C]
    return input


def _to_5d_array(input: np.ndarray) -> np.ndarray:
    """Convert a 2D, 3D, 4D, or 6D array to a 5D array.
    
    If the input is a 2D array, add 3 new axes at the beginning.
    If the input is a 3D array, add 2 new axes at the beginning.
    If the input is a 4D array, add a new axis at the beginning.
    If the input is a 6D array with the first dimension being 1, remove the
    first dimension.
    
    Args:
        input: An input array.
    
    Return:
        A 5D array of shape [*, B, H, W, C].
    """
    assert isinstance(input, np.ndarray)
    ndim = input.ndim
    assert 2 <= ndim <= 6
    if input.ndim == 2:    # [H, W] -> [1, 1, 1, H, W]
        input = np.expand_dims(input, axis=0)
    elif input.ndim == 3:  # [H, W, C] -> [1, 1, H, W, C]
        input = np.expand_dims(input, axis=0)
    elif input.ndim == 4:  # [B, H, W, C] -> [1, B, H, W, C]
        input = np.expand_dims(input, axis=0)
    elif input.ndim == 5:  # [*, B, H, W, C]
        pass
    elif input.ndim == 6 and input.shape[0] == 1:
        input = np.squeeze(input, axis=0)  # [1, *, B, H, W, C] -> [*, B, H, W, C]
    return input


def to_3d_array(input: Any) -> np.ndarray:
    """Convert an arbitrary input to a 3D array.
   
    Args:
        input: An input of arbitrary type.
        
    Return:
        A 3D array of shape [H, W, C].
    """
    if isinstance(input, dict):
        input = list(input.values())                                            # dict -> list[Tensor | np.ndarray]
    if isinstance(input, tuple):
        input = list(input)                                                     # tuple -> list[Tensor | np.ndarray]
    if isinstance(input, list) and all(isinstance(i, torch.Tensor) for i in input):
        input = [i.detach().cpu().numpy() for i in input]                       # list[Tensor] -> list[np.ndarray]
    if isinstance(input, list) and all(isinstance(i, np.ndarray)   for i in input):
        if all(i.ndim == 2 for i in input):
            input = np.stack(input)                                             # list[2D np.ndarray] -> 3D np.ndarray
        else:
            raise ValueError(f"Expect `input.ndim` == 2.")
    if isinstance(input, torch.Tensor):
        input = input.detach().cpu().numpy()                                    # Tensor -> np.ndarray any dimensions
    if isinstance(input, np.ndarray):
        return _to_3d_array(input)                                              # np.ndarray any dimensions -> 3D np.ndarray
    raise TypeError(
        f"`input` must be a `np.ndarray`. But got: {type(input)}."
    )


def to_4d_array(input: Any) -> np.ndarray:
    """Convert an arbitrary input to a 4D array.
   
    Args:
        input: An input of arbitrary type.
        
    Return:
        A 4D array of shape [B, H, W, C].
    """
    if isinstance(input, dict):
        input = list(input.values())                                            # dict -> list[Tensor | np.ndarray]
    if isinstance(input, tuple):
        input = list(input)                                                     # tuple -> list[Tensor | np.ndarray]
    if isinstance(input, list) and all(isinstance(i, torch.Tensor) for i in input):
        input = [i.detach().cpu().numpy() for i in input]                       # list[Tensor] -> list[np.ndarray]
    if isinstance(input, list) and all(isinstance(i, np.ndarray)   for i in input):
        if all(i.ndim == 2 for i in input):
            input = [np.expand_dims(i, axis=0) for i in input]                  # list[2D np.ndarray] -> list[3D np.ndarray]
        if all(i.ndim == 3 for i in input):
            input = np.stack(input)                                             # list[3D np.ndarray] -> 4D np.ndarray
        else:
            raise ValueError(f"Expect 2 <= `input.ndim` <= 3.")
    if isinstance(input, torch.Tensor):
        input = input.detach().cpu().numpy()                                    # Tensor -> np.ndarray any dimensions
    if isinstance(input, np.ndarray):
        return _to_4d_array(input)                                              # np.ndarray any dimensions -> 4D np.ndarray
    raise TypeError(
        f"`input` must be a `np.ndarray`. But got: {type(input)}."
    )


def to_5d_array(input) -> np.ndarray:
    """Convert an arbitrary input to a 5D array.
   
    Args:
        input: An input of arbitrary type.
        
    Return:
        A 5D array of shape [*, B, H, W, C].
    """
    if isinstance(input, dict):
        input = list(input.values())                                            # dict -> list[Tensor | np.ndarray]
    if isinstance(input, tuple):
        input = list(input)                                                     # tuple -> list[Tensor | np.ndarray]
    if isinstance(input, list) and all(isinstance(i, torch.Tensor) for i in input):
        input = [i.detach().cpu().numpy() for i in input]                       # list[Tensor] -> list[np.ndarray]
    if isinstance(input, list) and all(isinstance(i, np.ndarray)   for i in input):
        if all(i.ndim == 2 for i in input):
            input = [np.expand_dims(i, axis=0) for i in input]                  # list[2D np.ndarray] -> list[3D np.ndarray]
        if all(3 <= i.ndim <= 4 for i in input):
            input = np.stack(input)                                             # list[3D np.ndarray] -> 4D np.ndarray
        else:
            raise ValueError(f"Expect 2 <= `input.ndim` <= 4.")
    if isinstance(input, torch.Tensor):
        input = input.detach().cpu().numpy()                                    # Tensor -> np.ndarray any dimensions
    if isinstance(input, np.ndarray):
        return _to_5d_array(input)                                              # np.ndarray any dimensions -> 5D np.ndarray
    raise TypeError(
        f"`input` must be a `np.ndarray`. But got: {type(input)}."
    )


def to_list_of_3d_array(input: Any) -> list[np.ndarray]:
    """Convert arbitrary input to a list of 3D arrays.
   
    Args:
        input: An Input of arbitrary type.
        
    Return:
        A list of 3D arrays of shape [H, W, C].
    """
    if isinstance(input, dict):
        input = list(input.values())                                            # list[Tensor | np.ndarray]
    if isinstance(input, tuple):
        input = list(input)                                                     # list[Tensor | np.ndarray]
    if isinstance(input, torch.Tensor):
        input = input.detach().cpu().numpy()                                    # Tensor -> np.ndarray
    if isinstance(input, np.ndarray):
        if input.ndim == 3:
            input = [input]                                                     # np.ndarray -> list[3D np.ndarray]
        elif input.ndim == 4:
            input = list(input)                                                 # np.ndarray -> list[3D np.ndarray]
        else:
            raise ValueError(
                f"Expect 3 <= `input.ndim` <= 4. But got: {input.ndim}."
            )
    if isinstance(input, list) and all(isinstance(i, torch.Tensor) for i in input):
        input = [i.detach().cpu().numpy() for i in input]                       # list[Tensor any dimensions] -> list[np.ndarray any dimensions]
        
    if isinstance(input, list) and all(isinstance(i, np.ndarray)   for i in input):
        return input                                                            # list[np.ndarray any dimensions] -> list[3D np.ndarray]
    raise TypeError(
        f"`input` must be a `np.ndarray`. But got: {type(input)}."
    )


def to_list_of_4d_array(input) -> list[np.ndarray]:
    """Convert arbitrary input to a list of 4D arrays.
   
    Args:
        input: An input of arbitrary type.
        
    Return:
       A list of 4D arrays of shape [B, H, W, C].
    """
    if isinstance(input, dict):
        input = list(input.values())                                            # list[Tensor | np.ndarray]
    if isinstance(input, tuple):
        input = list(input)                                                     # list[Tensor | np.ndarray]
    if isinstance(input, torch.Tensor):
        input = input.detach().cpu().numpy()                                    # Tensor -> np.ndarray
    if isinstance(input, np.ndarray):
        if input.ndim == 3:
            input = [np.expand_dims(input, axis=0)]                             # np.ndarray -> list[4D np.ndarray]
        elif input.ndim == 4:
            input = [input]                                                     # np.ndarray -> list[4D np.ndarray]
        elif input.ndim == 5:
            input = list(input)                                                 # np.ndarray -> list[4D np.ndarray]
        else:
            raise ValueError(f"Expect 3 <= `input.ndim` <= 5.")
    if isinstance(input, list) and all(isinstance(i, torch.Tensor) for i in input):
        input = [i.detach().cpu().numpy() for i in input]                       # list[Tensor] -> list[np.ndarray any dimensions]
    
    if isinstance(input, list) and all(isinstance(i, np.ndarray)   for i in input):
        return [_to_4d_array(i) for i in input]                                 # list[np.ndarray any dimensions] -> list[4D np.ndarray]
    raise TypeError(
        f"`input` must be a `np.ndarray`. But got: {type(input)}."
    )


def upcast(input: np.ndarray) -> np.ndarray:
    """Protects from numerical overflows in multiplications by upcasting to
    the equivalent higher type.
    
    Args:
        input (Tensor): Array of arbitrary type.
    
    Return:
        Array of higher type.
    """
    if not isinstance(input, np.ndarray):
        raise TypeError(f"`input` must be a `np.ndarray`. But got: {type(input)}.")
    if type(input) in (np.float16, np.float32, np.float64):
        return input.astype(float)
    if type(input) in (np.int16, np.int32, np.int64):
        return input.astype(int)
    return input

# endregion
