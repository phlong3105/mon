#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Cityscapes Datasets.

This module implements the Cityscapes dataset.

References:
	https://www.cityscapes-dataset.com/
"""

from __future__ import annotations

import mon.dataset.cityscapes.cityscapes
import mon.dataset.cityscapes.cityscapes_foggy
import mon.dataset.cityscapes.cityscapes_rain
import mon.dataset.cityscapes.cityscapes_snow
from mon.dataset.cityscapes.cityscapes import *
from mon.dataset.cityscapes.cityscapes_foggy import *
from mon.dataset.cityscapes.cityscapes_rain import *
from mon.dataset.cityscapes.cityscapes_snow import *
