#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module defines type aliases used throughout :mod:`checkout` package."""

from __future__ import annotations

__all__ = [
    "CallableType",
    "ClassLabelsType",
    "ConfigType",
    "DictType",
    "InstancesType",
    "Int3T",
    "Ints",
    "MotionType",
    "MovingStateType",
    "ObjectType",
    "PathType",
    "TensorOrArray",
    "PathsType",
    "PointsType",
    "Strs",
    "UIDType",
    "WeightsType",
]

from typing import Sequence, TypeAlias

import numpy as np

from mon.typing import *
from supr import constant, data, motion

InstancesType  : TypeAlias = data.Instance | Sequence[data.Instance]
MovingStateType: TypeAlias = str | int | constant.MovingState
MotionType     : TypeAlias = DictType | str | CallableType | motion.Motion
ObjectType     : TypeAlias = DictType | str | CallableType | data.Object
PointsType     : TypeAlias = np.ndarray | list | tuple
UIDType        : TypeAlias = int | str
