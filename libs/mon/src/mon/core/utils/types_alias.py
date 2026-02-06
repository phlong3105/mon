#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Type aliases.

This module provides type aliases to simplify and standardize type annotations.
"""

from __future__ import annotations

__all__ = [
    "DeviceType",
    "DictType",
    "PathLike",
    "TensorOrArray",
    "float_2_t",
    "float_3_t",
    "int_2_t",
    "int_3_t",
    "str_any_t",
]

import pathlib
from typing import Optional, TypeVar, Union

import box
import numpy as np
import torch
from typing_extensions import TypeAlias as _TypeAlias

T = TypeVar("T")
_scalar_or_tuple_any_t: _TypeAlias = Union[T, tuple[T, ...]]
_scalar_or_tuple_1_t: _TypeAlias = Union[T, tuple[T]]
_scalar_or_tuple_2_t: _TypeAlias = Union[T, tuple[T, T]]
_scalar_or_tuple_3_t: _TypeAlias = Union[T, tuple[T, T, T]]
_scalar_or_tuple_4_t: _TypeAlias = Union[T, tuple[T, T, T, T]]
_scalar_or_tuple_5_t: _TypeAlias = Union[T, tuple[T, T, T, T, T]]
_scalar_or_tuple_6_t: _TypeAlias = Union[T, tuple[T, T, T, T, T, T]]

int_2_t: _TypeAlias = _scalar_or_tuple_2_t[int]
int_3_t: _TypeAlias = _scalar_or_tuple_3_t[int]

float_2_t: _TypeAlias = _scalar_or_tuple_3_t[float]
float_3_t: _TypeAlias = _scalar_or_tuple_3_t[float]

str_any_t: _TypeAlias = _scalar_or_tuple_any_t[str]

DeviceType: _TypeAlias = Optional[Union[torch.device, str, int]]
DictType: _TypeAlias = Union[dict, box.Box]
PathLike: _TypeAlias = Union[str, pathlib.Path]
TensorOrArray: _TypeAlias = Union[torch.Tensor, np.ndarray]
