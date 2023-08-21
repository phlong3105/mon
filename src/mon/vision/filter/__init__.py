#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package implements filtering functions.

The different between :mod:`mon.vision.filter` and :mod:`mon.vision.enhance` is:
    - Filter functions are used to process or manipulate data by applying
      specific criteria or rules. Filters are often used to extract or select
      relevant information from a dataset, remove unwanted elements, or
      transform data in a particular way.
    - Enhance functions typically refer to operations or techniques that improve
      the quality, clarity, or visibility of an image, audio, or video. These
      functions aim to make the content more perceptually pleasing, informative,
      or aesthetically appealing. Enhancements can involve adjustments to
      brightness, contrast, color balance, sharpness, noise reduction, etc.
"""

from __future__ import annotations

import mon.vision.filter.box
import mon.vision.filter.core
from mon.vision.filter.box import *
from mon.vision.filter.core import *
