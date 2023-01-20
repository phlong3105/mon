#!/usr/bin/env python
# -*- coding: utf-8 -*-

""":mod:`mon.vision.typing` is a public API of all the type aliases used
internally in :mod:`mon.coreml`.

This design pattern brings several benefits:
1. It can avoid circular dependency.
2. It simplifies import statements when accessing internal type aliases of the
   package. The user of the API don't have to remember which module defines the
   custom types.
"""

from __future__ import annotations

__all__ = [
    "LogitsType",
    # Extend :mod:`mon.foundation.typing`
    "CallableType", "ConfigType", "DictType",  "Float1T", "Float2T", "Float3T",
    "Float4T", "Float5T", "Float6T", "FloatAnyT", "Floats", "ImageFormatType",
    "Int1T", "Int2T", "Int3T", "Int4T", "Int5T", "Int6T", "IntAnyT", "Ints",
    "MemoryUnitType", "PathType", "PathsType", "Strs", "VideoFormatType",
    # Extend :mod:`mon.coreimage.typing`
    "AppleRGBType", "BBoxFormatType", "BBoxFormatType", "BBoxType",
    "BasicRGBType", "BorderTypeType", "CFAType", "DistanceMetricType", "Image",
    "Images", "InterpolationModeType", "MaskType", "PaddingModeType",
    "PaddingModeType", "PointsType", "RGBType", "VisionBackendType",
    # Extend :mod:`mon.coreml.typing`
    "CallbackType", "CallbacksType", "ClassLabelsType", "EpochOutput",
    "LRSchedulerType", "LRSchedulersType", "LoggerType", "LoggersType",
    "LossType", "LossesType", "MetricType", "MetricsType", "ModelPhaseType",
    "OptimizerType", "OptimizersType", "ReductionType", "StepOutput",
    "TransformType", "TransformsType",
]

from typing import Sequence, TypeAlias

import numpy as np
import torch

from mon.coreimage.typing import *
from mon.coreml.typing import *

LogitsType: TypeAlias = torch.Tensor | np.ndarray | Sequence[int]
