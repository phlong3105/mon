#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Type Hints & Aliases.

This module defines common type hints and aliases used throughout the ``mon``
package.
"""

from __future__ import annotations

__all__ = [
    "BBoxFormatLike",
    "DeviceLike",
    "DictLike",
    "Float2",
    "Float3",
    "Float4",
    "FloatOrTuple",
    "FloatOrTuple2",
    "FloatOrTuple3",
    "FloatOrTuple4",
    "Int2",
    "Int3",
    "Int4",
    "IntOrTuple",
    "IntOrTuple2",
    "IntOrTuple3",
    "IntOrTuple4",
    "MISSING",
    "MemoryUnitLike",
    "PathLike",
    "RunModeLike",
    "SplitLike",
    "StrOrList",
    "TaskLike",
    "TensorOrArray",
]

from typing import Any, TypeVar, Union

import torch
from box import Box
from numpy import ndarray
from torch import Tensor
from typing_extensions import TypeAlias

from .dtype import BBoxFormat, MemoryUnit, RunMode, Split, Task
from .path import Path

T = TypeVar("T")
_scalar_or_tuple_any_t: TypeAlias = Union[T, tuple[T, ...]]
_scalar_or_tuple_1_t: TypeAlias = Union[T, tuple[T]]
_scalar_or_tuple_2_t: TypeAlias = Union[T, tuple[T, T]]
_scalar_or_tuple_3_t: TypeAlias = Union[T, tuple[T, T, T]]
_scalar_or_tuple_4_t: TypeAlias = Union[T, tuple[T, T, T, T]]
_scalar_or_tuple_5_t: TypeAlias = Union[T, tuple[T, T, T, T, T]]
_scalar_or_tuple_6_t: TypeAlias = Union[T, tuple[T, T, T, T, T, T]]

Int2: TypeAlias = tuple[int, int]
Int3: TypeAlias = tuple[int, int, int]
Int4: TypeAlias = tuple[int, int, int, int]

Float2: TypeAlias = tuple[float, float]
Float3: TypeAlias = tuple[float, float, float]
Float4: TypeAlias = tuple[float, float, float, float]

IntOrTuple: TypeAlias = _scalar_or_tuple_any_t[int]
IntOrTuple2: TypeAlias = _scalar_or_tuple_2_t[int]
IntOrTuple3: TypeAlias = _scalar_or_tuple_3_t[int]
IntOrTuple4: TypeAlias = _scalar_or_tuple_4_t[int]

FloatOrTuple: TypeAlias = _scalar_or_tuple_any_t[float]
FloatOrTuple2: TypeAlias = _scalar_or_tuple_2_t[float]
FloatOrTuple3: TypeAlias = _scalar_or_tuple_3_t[float]
FloatOrTuple4: TypeAlias = _scalar_or_tuple_4_t[float]

MISSING = object()

StrOrList: TypeAlias = Union[str, list[str]]

BBoxFormatLike: TypeAlias = Union[BBoxFormat, str]
DeviceLike: TypeAlias = Union[torch.device, str, int]
DictLike: TypeAlias = Union[Box, dict[str, Any]]
MemoryUnitLike: TypeAlias = Union[MemoryUnit, str]
PathLike: TypeAlias = Union[Path, str]
RunModeLike: TypeAlias = Union[RunMode, str]
SplitLike: TypeAlias = Union[Split, str]
TaskLike: TypeAlias = Union[Task, str]
TensorOrArray: TypeAlias = Union[Tensor, ndarray]
