#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package implements layers that are used to build deep learning models.
The layers are split into two categories: common (widely used and accepted
layers) and specific (layers that are specially designed for a model).

In this package, we keep the same naming convention as in
:mod:`torch.nn.modules`.
"""

from __future__ import annotations

import mon.coreml.layer.base
import mon.coreml.layer.common
import mon.coreml.layer.parsing
import mon.coreml.layer.specific
import mon.coreml.layer.typing
from mon.coreml.layer.base import *
from mon.coreml.layer.common import *
from mon.coreml.layer.parsing import *
from mon.coreml.layer.specific import *
