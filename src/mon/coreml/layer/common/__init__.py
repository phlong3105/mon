#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package implements the common (widely used and accepted layers) layers
to build deep learning models.
"""

from __future__ import annotations

import mon.coreml.layer.common.activation
import mon.coreml.layer.common.attention
import mon.coreml.layer.common.attn_block
import mon.coreml.layer.common.bottleneck
import mon.coreml.layer.common.conv
import mon.coreml.layer.common.dropout
import mon.coreml.layer.common.head
import mon.coreml.layer.common.linear
import mon.coreml.layer.common.mutating
import mon.coreml.layer.common.normalization
import mon.coreml.layer.common.padding
import mon.coreml.layer.common.pooling
import mon.coreml.layer.common.sampling
from mon.coreml.layer.common.activation import *
from mon.coreml.layer.common.attention import *
from mon.coreml.layer.common.attn_block import *
from mon.coreml.layer.common.bottleneck import *
from mon.coreml.layer.common.conv import *
from mon.coreml.layer.common.dropout import *
from mon.coreml.layer.common.head import *
from mon.coreml.layer.common.linear import *
from mon.coreml.layer.common.mutating import *
from mon.coreml.layer.common.normalization import *
from mon.coreml.layer.common.padding import *
from mon.coreml.layer.common.pooling import *
from mon.coreml.layer.common.sampling import *

"""The basic dependency chart is as follows:
|------------------|------------------------------------------------------------|
| Atomic modules   | activation, dropout, linear, mutating, padding, sampling,  |
|------------------|------------------------------------------------------------|
| 2nd tier modules | conv, normalization, pooling,                              |
|------------------|------------------------------------------------------------|
| 3rd tier modules | attention, bottleneck, head,                               |
|------------------|------------------------------------------------------------|
| 4th tier modules | attn_block                                                 |
|------------------|------------------------------------------------------------|
"""
