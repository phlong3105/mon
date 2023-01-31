#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module defines type aliases used throughout :mod:`checkout` package."""

from __future__ import annotations

__all__ = [
    "CallableType",
    "InstancesType",
    "Ints",
    "MotionType",
    "MovingStateType",
    "PathType",
    "PathsType",
    "PointsType",
    "UIDType",
]

from typing import Sequence, TypeAlias

import numpy as np

from mon.typing import *
from supr import constant, data, tracking

InstancesType  : TypeAlias = data.Instance | Sequence[data.Instance]
MovingStateType: TypeAlias = str | int | constant.MovingState
MotionType     : TypeAlias = DictType | str | CallableType | tracking.Motion
PointsType     : TypeAlias = np.ndarray | list | tuple
UIDType        : TypeAlias = int | str
