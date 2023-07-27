#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package implements logging mechanisms to record intermediate results
during training, validating, testing, and inferring.
"""

from __future__ import annotations

import mon.nn.logger.base
import mon.nn.logger.tensorboard
from mon.nn.logger.base import *
from mon.nn.logger.tensorboard import *
