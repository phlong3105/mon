#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Type aliases.

This module provides type aliases to simplify and standardize type annotations.
"""

from __future__ import annotations

__all__ = [
    "ClassListLike",
    "DeviceType",
    "DictType",
    "SplitType",
    "TensorOrArray",
    "float_2_t",
    "float_3_t",
    "int_2_t",
    "int_3_t",
    "int_any_t",
    "str_any_t",
]

from typing import Optional, TypeVar, Union

import box
import numpy as np
import torch
from typing_extensions import TypeAlias

from mon.core.dtypes import ClassList
from mon.core.enum import Split
from mon.core.pathlib import Path

T = TypeVar("T")
_scalar_or_tuple_any_t: TypeAlias = Union[T, tuple[T, ...]]
_scalar_or_tuple_1_t: TypeAlias = Union[T, tuple[T]]
_scalar_or_tuple_2_t: TypeAlias = Union[T, tuple[T, T]]
_scalar_or_tuple_3_t: TypeAlias = Union[T, tuple[T, T, T]]
_scalar_or_tuple_4_t: TypeAlias = Union[T, tuple[T, T, T, T]]
_scalar_or_tuple_5_t: TypeAlias = Union[T, tuple[T, T, T, T, T]]
_scalar_or_tuple_6_t: TypeAlias = Union[T, tuple[T, T, T, T, T, T]]

int_any_t: TypeAlias = _scalar_or_tuple_any_t[int]
int_2_t: TypeAlias = _scalar_or_tuple_2_t[int]
int_3_t: TypeAlias = _scalar_or_tuple_3_t[int]

float_2_t: TypeAlias = _scalar_or_tuple_3_t[float]
float_3_t: TypeAlias = _scalar_or_tuple_3_t[float]

str_any_t: TypeAlias = _scalar_or_tuple_any_t[str]

ClassListLike: TypeAlias = Union[ClassList, Path, str]
DeviceType: TypeAlias = Optional[Union[torch.device, str, int]]
DictType: TypeAlias = Union[box.Box, dict]
SplitType: TypeAlias = Union[Split, str]
TensorOrArray: TypeAlias = Union[torch.Tensor, np.ndarray]
