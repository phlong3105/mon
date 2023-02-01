#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module defines type aliases used throughout :mod:`mon.coreimage`
package.
"""

from __future__ import annotations

__all__ = [
    "AppleRGBType", "BBoxFormatType", "BBoxFormatType", "BasicRGBType",
    "BorderTypeType", "CFAType", "DistanceMetricType", "InterpolationModeType",
    "PaddingModeType", "PaddingModeType", "RGBType", "TensorOrArray",
    "VisionBackendType",
    # Extend :mod:`mon.core.typing`
    "CallableType", "ConfigType", "DictType",  "Float1T", "Float2T", "Float3T",
    "Float4T", "Float5T", "Float6T", "FloatAnyT", "Floats", "ImageFormatType",
    "Int1T", "Int2T", "Int3T", "Int4T", "Int5T", "Int6T", "IntAnyT", "Ints",
    "MemoryUnitType", "Number", "PathType", "PathsType", "Strs",
    "VideoFormatType",
]

from typing import TypeAlias

import numpy
import torch

from mon.core.typing import *
from mon.coreimage import constant

TensorOrArray        : TypeAlias = torch.Tensor | numpy.ndarray  # | Sequence[int | float]
AppleRGBType         : TypeAlias = str | int | constant.AppleRGB
BBoxFormatType       : TypeAlias = str | int | constant.BasicRGB
BasicRGBType         : TypeAlias = str | int | constant.BBoxFormat
BorderTypeType       : TypeAlias = str | int | constant.BorderType
CFAType              : TypeAlias = str | int | constant.CFA
DistanceMetricType   : TypeAlias = str | int | constant.DistanceMetric
InterpolationModeType: TypeAlias = str | int | constant.InterpolationMode
PaddingModeType      : TypeAlias = str | int | constant.PaddingMode
RGBType              : TypeAlias = str | int | constant.RGB
VisionBackendType    : TypeAlias = str | int | constant.VisionBackend
