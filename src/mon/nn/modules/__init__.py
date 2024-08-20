#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package implements modules that are used to build deep learning models.
The layers are split into two categories: base (atomic layers) and custom
(layers that are specially designed for a model).

In this package, we keep the same naming convention as in :mod:`torch.nn.modules`
for consistency.
"""

from __future__ import annotations

import mon.nn.modules.activation
import mon.nn.modules.attention
import mon.nn.modules.bottleneck
import mon.nn.modules.conv
import mon.nn.modules.dropout
import mon.nn.modules.extract
import mon.nn.modules.flatten
import mon.nn.modules.fold
import mon.nn.modules.linear
import mon.nn.modules.merge
import mon.nn.modules.misc
import mon.nn.modules.mlp
import mon.nn.modules.normalization
import mon.nn.modules.padding
import mon.nn.modules.pooling
import mon.nn.modules.prior
import mon.nn.modules.projection
import mon.nn.modules.sampling
import mon.nn.modules.shuffle
from mon.nn.modules.activation import *
from mon.nn.modules.attention import *
from mon.nn.modules.bottleneck import *
from mon.nn.modules.conv import *
from mon.nn.modules.dropout import *
from mon.nn.modules.extract import *
from mon.nn.modules.flatten import *
from mon.nn.modules.fold import *
from mon.nn.modules.linear import *
from mon.nn.modules.merge import *
from mon.nn.modules.misc import *
from mon.nn.modules.mlp import *
from mon.nn.modules.normalization import *
from mon.nn.modules.padding import *
from mon.nn.modules.pooling import *
from mon.nn.modules.prior import *
from mon.nn.modules.projection import *
from mon.nn.modules.sampling import *
from mon.nn.modules.shuffle import *
