#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package implements layers that are used to build deep learning models.
The layers are split into two categories: base (atomic layers) and custom
(layers that are specially designed for a model).

In this package, we keep the same naming convention as in
:mod:`torch.nn.modules`.
"""

from __future__ import annotations

import mon.coreml.layer.base
import mon.coreml.layer.custom
import mon.coreml.layer.parsing
import mon.coreml.layer.typing
from mon.coreml.layer.base import *
from mon.coreml.layer.custom import *
from mon.coreml.layer.parsing import *
