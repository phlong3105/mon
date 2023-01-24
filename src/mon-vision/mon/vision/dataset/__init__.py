#!/usr/bin/env python
# -*- coding: utf-8 -*-

""":mod:`mon.vision.dataset` package implements datasets and labels used in
computer vision.
"""

from __future__ import annotations

import mon.vision.dataset.base
from mon.vision.dataset.a2i2 import *
from mon.vision.dataset.base import *
from mon.vision.dataset.cifar import *
from mon.vision.dataset.cityscapes import *
from mon.vision.dataset.gt_rain import *
from mon.vision.dataset.kodas import *
from mon.vision.dataset.lol import *
from mon.vision.dataset.mnist import *
from mon.vision.dataset.ntire import *
from mon.vision.dataset.rain13k import *
from mon.vision.dataset.reside import *
from mon.vision.dataset.satehaze1k import *
from mon.vision.dataset.sice import *
from mon.vision.dataset.snow100k import *

"""
current_dir = pathlib.Path(__file__).resolve().parent
files       = list(current_dir.rglob("*.py"))
for f in files:
    module = f.stem
    if module == "__init__":
        continue
    importlib.import_module(f"one.datasets.{module}")
"""
