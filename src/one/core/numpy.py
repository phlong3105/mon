#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Operations on Numpy Array.
"""

from __future__ import annotations

import inspect
import sys
from typing import Any

import numpy as np
from multipledispatch import dispatch
from torch import Tensor

from one.core.collection import is_list_of


# MARK: - Functional

def _to_3d(input: np.ndarray) -> np.ndarray:
    """Convert the tensor or array to 3D shape [C, H, W] or [H, W, C]."""
    if not isinstance(input, np.ndarray):
        raise TypeError(f"`input` must be a `np.ndarray`. But got: {type(input)}.")
    if not 2 <= input.ndim <= 4:
        raise ValueError(f"Require 2 <= `input.ndim` <= 4. But got: {input.ndim}.")
    
    if input.ndim == 2:  # [H, W] -> [1, H, W]
        input = np.expand_dims(input, axis=0)
    if input.ndim == 3:  # [C, H, W]
        input = np.expand_dims(input, axis=0)
    if input.ndim == 4 and input.shape[0] == 1:  # [1, C, H, W] -> [C, H, W]
        input = np.squeeze(input, axis=0)
    return input


def _to_4d(input: np.ndarray) -> np.ndarray:
    """Convert the tensor or array to 4D shape [B, C, H, W] or [B, H, W, C]."""
    if not isinstance(input, np.ndarray):
        raise TypeError(f"`input` must be a `np.ndarray`. But got: {type(input)}.")
    if not 2 <= input.ndim <= 5:
        raise ValueError(f"Require 2 <= `input.ndim` <= 5. But got: {input.ndim}.")
    
    if input.ndim == 2:  # [H, W] -> [1, H, W]
        input = np.expand_dims(input, axis=0)
    if input.ndim == 3:  # [C, H, W] -> [1, C, H, W]
        input = np.expand_dims(input, axis=0)
    if input.ndim == 4:  # [B, C, H, W]
        pass
    if input.ndim == 5 and input.shape[0] == 1:
        input = np.squeeze(input, axis=0)
    return input


def _to_5d(input: np.ndarray) -> np.ndarray:
    """Convert the tensor or array to 5D shape [*, B, C, H, W] or [*, B, H, W, C].
    """
    if not isinstance(input, np.ndarray):
        raise TypeError(f"`input` must be a `np.ndarray`. But got: {type(input)}.")
    if not 2 <= input.ndim <= 6:
        raise ValueError(f"Require 2 <= `input.ndim` <= 6. But got: {input.ndim}.")

    if input.ndim == 2:  # [H, W] -> [1, H, W]
        input = np.expand_dims(input, axis=0)
    if input.ndim == 3:  # [C, H, W] -> [1, C, H, W]
        input = np.expand_dims(input, axis=0)
    if input.ndim == 4:  # [B, C, H, W] -> [1, B, C, H, W]
        input = np.expand_dims(input, axis=0)
    if input.ndim == 5:  # [*, B, C, H, W]
        pass
    if input.ndim == 6 and input.shape[0] == 1:
        input = np.squeeze(input, axis=0)
    return input


def to_3d_array_list(input: Any) -> list[np.ndarray]:
    """Convert to a 3D-array list."""
    if isinstance(input, dict):
        input = list(input.values())
    if isinstance(input, tuple):
        input = list(input)
    if isinstance(input, Tensor):
        input = input.detach().cpu().numpy()
    if isinstance(input, np.ndarray):
        if input.ndim == 3:
            input = [input]
        elif input.ndim == 4:
            input = list(input)
        else:
            raise ValueError(f"Require 3 <= `input.ndim` <= 4. But got: {input.ndim}.")
    if isinstance(input, list) and is_list_of(input, Tensor):
        input = [i.detach().cpu().numpy() for i in input]
    if isinstance(input, list) and is_list_of(input, np.ndarray):
        if all(i.ndim == 3 for i in input):
            return input
        else:
            raise ValueError(f"Require all `input.ndim` == 3.")
    raise TypeError(f"`input` must be a `np.ndarray`. But got: {type(input)}.")
    

def to_4d_array(input) -> np.ndarray:
    """Convert to a 4D-array. The output will be:
		- Single 3D-array will be expanded to a single 4D-array.
		- Single 4D-array will remain the same.
		- Sequence of 3D-arrays will be stacked into a 4D-array.
		- Sequence of 4D-arrays will remain the same.
	"""
    if isinstance(input, dict):
        input = list(input.values())
    if isinstance(input, tuple):
        input = list(input)
    if isinstance(input, list) and is_list_of(input, Tensor):
        input = [i.detach().cpu().numpy() for i in input]
    if isinstance(input, list) and is_list_of(input, np.ndarray):
        if all(i.ndim == 2 for i in input):
            input = [np.expand_dims(i, axis=0) for i in input]
        elif all(i.ndim == 3 for i in input):
            input = np.stack(input)
        else:
            raise ValueError(f"Require 2 <= `input.ndim` <= 3.")
    if isinstance(input, Tensor):
        input = input.detach().cpu().numpy()
    if isinstance(input, np.ndarray):
        return _to_4d(input)
    raise TypeError(f"`input` must be a `np.ndarray`. But got: {type(input)}.")
    

def to_4d_array_list(input) -> list[np.ndarray]:
    """Convert to a 4D-array list."""
    if isinstance(input, dict):
        input = list(input.values())
    if isinstance(input, tuple):
        input = list(input)
    if isinstance(input, Tensor):
        input = input.detach().cpu().numpy()
    if isinstance(input, np.ndarray):
        if input.ndim == 3:
            input = [np.expand_dims(input, axis=0)]
        elif input.ndim == 4:
            input = [input]
        elif input.ndim == 5:
            input = list(input)
        else:
            raise ValueError(f"Require 3 <= `input.ndim` <= 5.")
    if isinstance(input, list) and is_list_of(input, Tensor):
        input = [i.detach().cpu().numpy() for i in input]
    if isinstance(input, list) and is_list_of(input, np.ndarray):
        if all(i.ndim == 3 for i in input):
            return [np.stack(input, axis=0)]
        elif all(i.ndim == 4 for i in input):
            return input
        else:
            raise ValueError(f"Require 3 <= `input.ndim` <= 4.")
    raise TypeError(f"`input` must be a `np.ndarray`. But got: {type(input)}.")


def to_5d_array(input) -> np.ndarray:
    """Convert to a 5D-array."""
    if isinstance(input, dict):
        input = list(input.values())
    if isinstance(input, tuple):
        input = list(input)
    if isinstance(input, list) and is_list_of(input, Tensor):
        input = [i.detach().cpu().numpy() for i in input]
    if isinstance(input, list) and is_list_of(input, np.ndarray):
        if all(i.ndim == 2 for i in input):
            input = [np.expand_dims(i, axis=0) for i in input]
        elif all(3 <= i.ndim <= 4 for i in input):
            input = np.stack(input)
        else:
            raise ValueError(f"Require 2 <= `input.ndim` <= 4.")
    if isinstance(input, Tensor):
        input = input.detach().cpu().numpy()
    if isinstance(input, np.ndarray):
        return _to_5d(input)
    raise TypeError(f"`input` must be a `np.ndarray`. But got: {type(input)}.")


@dispatch(np.ndarray)
def upcast(input: np.ndarray) -> np.ndarray:
    """Protects from numerical overflows in multiplications by upcasting to
    the equivalent higher type.
    """
    if not isinstance(input, np.ndarray):
        raise TypeError(f"`input` must be a `np.ndarray`. But got: {type(input)}.")
    if type(input) in (np.float16, np.float32, np.float64):
        return input.astype(float)
    if type(input) in (np.int16, np.int32, np.int64):
        return input.astype(int)
    return input


# MARK: - Main

__all__ = [
    name for name, value in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__
    )
]
