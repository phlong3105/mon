#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package implements modules that are used to build deep learning models.
The layers are split into two categories: base (atomic layers) and custom
(layers that are specially designed for a model).

In this package, we keep the same naming convention as in :obj:`torch.nn.modules`
for consistency.
"""

from __future__ import annotations

import mon.nn.modules.activation
import mon.nn.modules.attention
import mon.nn.modules.conv
import mon.nn.modules.dropout
import mon.nn.modules.flatten
import mon.nn.modules.linear
import mon.nn.modules.misc
import mon.nn.modules.normalization
import mon.nn.modules.padding
import mon.nn.modules.pooling
import mon.nn.modules.prior
import mon.nn.modules.projection
import mon.nn.modules.scale
import mon.nn.modules.shuffle
import mon.nn.modules.siren
from mon.nn.modules.activation import *
from mon.nn.modules.attention import *
from mon.nn.modules.conv import *
from mon.nn.modules.dropout import *
from mon.nn.modules.flatten import *
from mon.nn.modules.linear import *
from mon.nn.modules.misc import *
from mon.nn.modules.normalization import *
from mon.nn.modules.padding import *
from mon.nn.modules.pooling import *
from mon.nn.modules.prior import *
from mon.nn.modules.projection import *
from mon.nn.modules.scale import *
from mon.nn.modules.shuffle import *
from mon.nn.modules.siren import *
