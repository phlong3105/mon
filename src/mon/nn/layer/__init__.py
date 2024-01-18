#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package implements layers that are used to build deep learning models.
The layers are split into two categories: base (atomic layers) and custom
(layers that are specially designed for a model).

In this package, we keep the same naming convention as in
:mod:`torch.nn.modules`.
"""

from __future__ import annotations

import mon.nn.layer.activation
import mon.nn.layer.attention
import mon.nn.layer.base
import mon.nn.layer.bottleneck
import mon.nn.layer.conv
import mon.nn.layer.dropout
import mon.nn.layer.extract
import mon.nn.layer.flatten
import mon.nn.layer.fold
import mon.nn.layer.linear
import mon.nn.layer.merge
import mon.nn.layer.misc
import mon.nn.layer.mlp
import mon.nn.layer.normalization
import mon.nn.layer.padding
import mon.nn.layer.pooling
import mon.nn.layer.projection
import mon.nn.layer.sampling
import mon.nn.layer.shuffle
from mon.nn.layer.activation import *
from mon.nn.layer.attention import *
from mon.nn.layer.base import *
from mon.nn.layer.bottleneck import *
from mon.nn.layer.conv import *
from mon.nn.layer.dropout import *
from mon.nn.layer.extract import *
from mon.nn.layer.flatten import *
from mon.nn.layer.fold import *
from mon.nn.layer.linear import *
from mon.nn.layer.merge import *
from mon.nn.layer.misc import *
from mon.nn.layer.mlp import *
from mon.nn.layer.normalization import *
from mon.nn.layer.padding import *
from mon.nn.layer.pooling import *
from mon.nn.layer.projection import *
from mon.nn.layer.sampling import *
from mon.nn.layer.shuffle import *
