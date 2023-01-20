#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module defines type aliases used throughout :mod:`mon.foundation`
package.
"""

from __future__ import annotations

__all__ = [
    "CallableType", "ConfigType", "DictType",  "Float1T", "Float2T", "Float3T",
    "Float4T", "Float5T", "Float6T", "FloatAnyT", "Floats", "ImageFormatType",
    "Int1T", "Int2T", "Int3T", "Int4T", "Int5T", "Int6T", "IntAnyT", "Ints",
    "MemoryUnitType", "PathType", "PathsType", "Strs", "VideoFormatType",
]

import functools
import types
from typing import Callable, Sequence, TypeAlias, TypeVar

import munch

from mon.foundation import constant, pathlib

T = TypeVar("T")
_ScalarOrTupleAnyT: TypeAlias = T | tuple[T, ...]
_ScalarOrTuple1T  : TypeAlias = T | tuple[T]
_ScalarOrTuple2T  : TypeAlias = T | tuple[T, T]
_ScalarOrTuple3T  : TypeAlias = T | tuple[T, T, T]
_ScalarOrTuple4T  : TypeAlias = T | tuple[T, T, T, T]
_ScalarOrTuple5T  : TypeAlias = T | tuple[T, T, T, T, T]
_ScalarOrTuple6T  : TypeAlias = T | tuple[T, T, T, T, T, T]

Strs              : TypeAlias = str | Sequence[str]

Ints              : TypeAlias = int | Sequence[int]
IntAnyT           : TypeAlias = _ScalarOrTupleAnyT[int]
Int1T             : TypeAlias = _ScalarOrTuple1T[int]
Int2T             : TypeAlias = _ScalarOrTuple2T[int]
Int3T             : TypeAlias = _ScalarOrTuple3T[int]
Int4T             : TypeAlias = _ScalarOrTuple4T[int]
Int5T             : TypeAlias = _ScalarOrTuple5T[int]
Int6T             : TypeAlias = _ScalarOrTuple6T[int]

Floats            : TypeAlias = float | Sequence[float]
FloatAnyT         : TypeAlias = _ScalarOrTupleAnyT[int]
Float1T           : TypeAlias = _ScalarOrTuple1T[int]
Float2T           : TypeAlias = _ScalarOrTuple2T[int]
Float3T           : TypeAlias = _ScalarOrTuple3T[int]
Float4T           : TypeAlias = _ScalarOrTuple4T[int]
Float5T           : TypeAlias = _ScalarOrTuple5T[int]
Float6T           : TypeAlias = _ScalarOrTuple6T[int]
                  
CallableType      : TypeAlias = Callable | types.FunctionType | functools.partial
DictType          : TypeAlias = dict | munch.Munch

ImageFormatType   : TypeAlias = str | int | constant.ImageFormat
MemoryUnitType    : TypeAlias = str | int | constant.MemoryUnit
VideoFormatType   : TypeAlias = str | int | constant.VideoFormat

PathType          : TypeAlias = pathlib.Path | str
PathsType         : TypeAlias = PathType | Sequence[PathType]

ConfigType        : TypeAlias = DictType | PathType
