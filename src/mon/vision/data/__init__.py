#!/usr/bin/env python
# -*- coding: utf-8 -*-

""":mod:`mon.vision.data` package implements datasets, datamodules, and
data augmentation used in computer vision tasks.
"""

from __future__ import annotations

import mon.vision.data.augment
import mon.vision.data.base
import mon.vision.data.dataset
from mon.vision.data.augment import *
from mon.vision.data.base import *
from mon.vision.data.dataset import *

"""
current_dir = pathlib.Path(__file__).resolve().parent
files       = list(current_dir.rglob("*.py"))
for f in files:
    module = f.stem
    if module == "__init__":
        continue
    importlib.import_module(f"mon.vision.dataset.{module}")
"""
