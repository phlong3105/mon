#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package implements attention layers that are used to build deep learning
models.
"""

from __future__ import annotations

import mon.nn.layer.attention.channel
import mon.nn.layer.attention.channelspatial
import mon.nn.layer.attention.pixel
import mon.nn.layer.attention.spatial
import mon.nn.layer.attention.supervised
import mon.nn.layer.attention.transformer
from mon.nn.layer.attention.channel import *
from mon.nn.layer.attention.channelspatial import *
from mon.nn.layer.attention.pixel import *
from mon.nn.layer.attention.spatial import *
from mon.nn.layer.attention.supervised import *
from mon.nn.layer.attention.transformer import *
