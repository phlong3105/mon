#!/usr/bin/env python
# -*- coding: utf-8 -*-

""":mod:`mon.vision.dataset` package implements datasets and labels used in
computer vision.
"""

from __future__ import annotations

import importlib
import pathlib

import mon.vision.dataset.base
from mon.vision.dataset.base import *

current_dir = pathlib.Path(__file__).resolve().parent
files       = list(current_dir.rglob("*.py"))
for f in files:
    module = f.stem
    if module == "__init__":
        continue
    importlib.import_module(f"one.datasets.{module}")
