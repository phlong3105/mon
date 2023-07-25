#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements layers that are used to build vision deep learning
models.

This module is built on top of :mod:`mon.coreml.layer`.
"""

from __future__ import annotations

import mon.vision.nn.layer.blueprint
import mon.vision.nn.layer.ffconv
import mon.vision.nn.layer.ghost
import mon.vision.nn.layer.mobileone
import mon.vision.nn.layer.srcnn
import mon.vision.nn.layer.unet
# noinspection PyUnresolvedReferences
from mon.coreml.layer import *
from mon.vision.nn.layer.blueprint import *
from mon.vision.nn.layer.ffconv import *
from mon.vision.nn.layer.ghost import *
from mon.vision.nn.layer.mobileone import *
from mon.vision.nn.layer.srcnn import *
from mon.vision.nn.layer.unet import *
