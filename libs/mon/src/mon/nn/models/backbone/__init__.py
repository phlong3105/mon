#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Backbones.

This package contains various neural network backbones. It strictly outputs a
list of feature maps (usually from different stages of the network).
"""

from __future__ import annotations

from .alexnet import *
from .convnext import *
from .densenet import *
from .efficientnet import *
from .mobilenetv2 import *
from .mobilenetv3 import *
from .resnet import *
from .swin import *
from .vgg import *
from .vit import *
