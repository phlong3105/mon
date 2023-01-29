#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module defines type aliases used throughout :mod:`checkout` package."""

from __future__ import annotations

__all__ = [
    "DetectionsType",
    "Ints",
    "PathType",
    "PathsType",
    "PointsType",
    "UIDType",
]

from typing import Sequence, TypeAlias

import numpy as np

from checkout.data import detection
from mon.typing import *


DetectionsType: TypeAlias = detection.Detection | Sequence[detection.Detection]
PointsType    : TypeAlias = np.ndarray | list | tuple
UIDType       : TypeAlias = int | str
