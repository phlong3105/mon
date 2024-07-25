#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module defines all global constants used across :mod:`openss` package.

Notes:
    - To avoid circular dependency, only define constants of basic/atomic types.
      The same goes for type aliases.
    - The only exception is the enum and factory constants.
"""

from __future__ import annotations

__all__ = [
    "CAMERAS",
    "DATA_DIR",
    "ROOT_DIR",
    "ZOO_DIR",
]

import os

import mon
from mon.globals import *


# region Directory

_current_file = mon.Path(__file__).absolute()
ROOT_DIR      = _current_file.parents[2]
SRC_DIR       = _current_file.parents[1]
PACKAGE_DIR   = _current_file.parents[0]
MON_DIR       = SRC_DIR / "mon"
MON_EXTRA_DIR = SRC_DIR / "mon_extra"

ZOO_DIR = None
for i, parent in enumerate(_current_file.parents):
    if (parent / "zoo").is_dir():
        ZOO_DIR = parent / "zoo"
        break
    if i >= 5:
        break
if ZOO_DIR is None:
    raise Warning(f"Cannot locate the ``zoo`` directory.")

DATA_DIR = mon.Path(os.getenv("DATA_DIR", None))
DATA_DIR = DATA_DIR or mon.Path("/data")
DATA_DIR = DATA_DIR if DATA_DIR.is_dir() else ROOT_DIR / "data"
if not DATA_DIR.is_dir():
    raise Warning(f"Cannot locate the ``data`` directory.")

# endregion


# region Factory

CAMERAS = mon.Factory(name="Cameras")

# endregion
