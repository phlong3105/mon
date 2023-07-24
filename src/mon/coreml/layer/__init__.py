#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package implements layers that are used to build deep learning models.
The layers are split into two categories: base (atomic layers) and custom
(layers that are specially designed for a model).

In this package, we keep the same naming convention as in
:mod:`torch.nn.modules`.
"""

from __future__ import annotations

import mon.coreml.layer.activation
import mon.coreml.layer.attention
import mon.coreml.layer.base
import mon.coreml.layer.bottleneck
import mon.coreml.layer.conv
import mon.coreml.layer.dropout
import mon.coreml.layer.head
import mon.coreml.layer.linear
import mon.coreml.layer.mutating
import mon.coreml.layer.normalization
import mon.coreml.layer.padding
import mon.coreml.layer.parsing
import mon.coreml.layer.pooling
import mon.coreml.layer.sampling
import mon.coreml.layer.typing
from mon.coreml.layer.activation import *
from mon.coreml.layer.attention import *
from mon.coreml.layer.base import *
from mon.coreml.layer.bottleneck import *
from mon.coreml.layer.conv import *
from mon.coreml.layer.dropout import *
from mon.coreml.layer.head import *
from mon.coreml.layer.linear import *
from mon.coreml.layer.mutating import *
from mon.coreml.layer.normalization import *
from mon.coreml.layer.padding import *
from mon.coreml.layer.parsing import *
from mon.coreml.layer.pooling import *
from mon.coreml.layer.sampling import *
