#!/usr/bin/env python
# -*- coding: utf-8 -*-

""":mod:`mon.vision.data` package implements datasets and datamodules used in
computer vision tasks.
"""

from __future__ import annotations

import mon.vision.data.a2i2_haze
import mon.vision.data.base
import mon.vision.data.cifar
import mon.vision.data.dehaze
import mon.vision.data.derain
import mon.vision.data.desnow
import mon.vision.data.kodas
import mon.vision.data.les
import mon.vision.data.llie
import mon.vision.data.mnist
from mon.vision.data.a2i2_haze import *
from mon.vision.data.base import *
from mon.vision.data.cifar import *
from mon.vision.data.dehaze import *
from mon.vision.data.derain import *
from mon.vision.data.desnow import *
from mon.vision.data.kodas import *
from mon.vision.data.les import *
from mon.vision.data.llie import *
from mon.vision.data.mnist import *

"""
current_dir = pathlib.Path(__file__).resolve().parent
files       = list(current_dir.rglob("*.py"))
for f in files:
    module = f.stem
    if module == "__init__":
        continue
    importlib.import_module(f"mon.vision.dataset.{module}")
"""
