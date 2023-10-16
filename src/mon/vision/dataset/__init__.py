#!/usr/bin/env python
# -*- coding: utf-8 -*-

""":mod:`mon.vision.dataset` package implements datasets and labels used in
computer vision.
"""

from __future__ import annotations

import mon.vision.dataset.a2i2_haze
import mon.vision.dataset.base
import mon.vision.dataset.cifar
import mon.vision.dataset.haze
import mon.vision.dataset.kodas
import mon.vision.dataset.llie
import mon.vision.dataset.mnist
import mon.vision.dataset.rain
import mon.vision.dataset.snow
from mon.vision.dataset.a2i2_haze import *
from mon.vision.dataset.base import *
from mon.vision.dataset.cifar import *
from mon.vision.dataset.haze import *
from mon.vision.dataset.kodas import *
from mon.vision.dataset.llie import *
from mon.vision.dataset.mnist import *
from mon.vision.dataset.rain import *
from mon.vision.dataset.snow import *

"""
current_dir = pathlib.Path(__file__).resolve().parent
files       = list(current_dir.rglob("*.py"))
for f in files:
    module = f.stem
    if module == "__init__":
        continue
    importlib.import_module(f"mon.vision.dataset.{module}")
"""
