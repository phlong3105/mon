#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module defines type aliases used throughout :mod:`checkout` package."""

from __future__ import annotations

__all__ = [
    "Ints",
    "PathType",
    "PathsType",
    "PointsType",
    "UIDType",
]

from typing import TypeAlias

import numpy as np

from mon.typing import *

PointsType: TypeAlias = np.ndarray | list | tuple
UIDType   : TypeAlias = int | str


