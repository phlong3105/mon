#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module defines all global constants used across :mod:`checkout` package.
"""

from __future__ import annotations

__all__ = [
    "CAMERAS",
    "DATA_DIR",
    "ROOT_DIR",
    "RUN_DIR",
    "SOURCE_DIR",
    "ZOO_DIR",
]

import os

import mon
from mon.globals import *

# region Factory

CAMERAS = mon.Factory(name="Cameras")

# endregion


# region Directory

_current_file = mon.Path(__file__).absolute()
PACKAGE_DIR   = _current_file.parents[0]
APP_DIR       = _current_file.parents[1]
SOURCE_DIR    = _current_file.parents[2]
ROOT_DIR      = _current_file.parents[3]
BIN_DIR       = ROOT_DIR / "bin"
DOCS_DIR      = ROOT_DIR / "docs"
RUN_DIR       = ROOT_DIR / "run"
TEST_DIR      = ROOT_DIR / "test"

ZOO_DIR = PACKAGE_DIR / "zoo"
if not ZOO_DIR.is_dir():
    ZOO_DIR = SOURCE_DIR / "zoo"
if not ZOO_DIR.is_dir():
    ZOO_DIR = ROOT_DIR / "zoo"

DATA_DIR = os.getenv("DATA_DIR", None)
if DATA_DIR is None:
    DATA_DIR = mon.Path("/data")
else:
    DATA_DIR = mon.Path(DATA_DIR)
if not DATA_DIR.is_dir():
    DATA_DIR = ROOT_DIR / "data"
if not DATA_DIR.is_dir():
    DATA_DIR = ""

# endregion
