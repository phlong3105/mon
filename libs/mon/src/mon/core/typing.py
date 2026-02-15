#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Type Hint & Alias.

This module defines common type hints and aliases used throughout the ``mon``
package.
"""

from __future__ import annotations

__all__ = [
    "BBoxFormatLike",
    "DeviceLike",
    "DictLike",
    "MemoryUnitLike",
    "PathLike",
    "RunModeLike",
    "SplitLike",
    "StrOrList",
    "TaskLike",
    "TensorOrArray",
    "float_2_t",
    "float_3_t",
    "float_4_t",
    "float_any_t",
    "int_2_t",
    "int_3_t",
    "int_4_t",
    "int_any_t",
]

from typing import Any, TypeVar, Union

import torch
from box import Box
from numpy import ndarray
from torch import Tensor
from typing_extensions import TypeAlias

from mon.core.enum import BBoxFormat, MemoryUnit, RunMode, Split, Task
from mon.core.path import Path

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
int_4_t: TypeAlias = _scalar_or_tuple_4_t[int]

float_any_t: TypeAlias = _scalar_or_tuple_any_t[float]
float_2_t: TypeAlias = _scalar_or_tuple_2_t[float]
float_3_t: TypeAlias = _scalar_or_tuple_3_t[float]
float_4_t: TypeAlias = _scalar_or_tuple_4_t[float]

StrOrList: TypeAlias = Union[str, list[str]]

BBoxFormatLike: TypeAlias = Union[BBoxFormat, str]
DeviceLike: TypeAlias = Union[torch.device | str | int]
DictLike: TypeAlias = Union[Box, dict[str, Any]]
MemoryUnitLike: TypeAlias = Union[MemoryUnit, str]
PathLike: TypeAlias = Union[Path, str]
RunModeLike: TypeAlias = Union[RunMode, str]
SplitLike: TypeAlias = Union[Split, str]
TaskLike: TypeAlias = Union[Task, str]
TensorOrArray: TypeAlias = Union[Tensor, ndarray]
