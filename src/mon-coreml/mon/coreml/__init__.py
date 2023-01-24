#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package provides the base functionalities for machine learning and
deep learning research. It provides support for high-level machine learning
packages, such as vision, natual language, or speech.

:mod:`mon.coreml` package itself is built on top `PyTorch` and `Lightning`
libraries.
"""

from __future__ import annotations

import mon.coreml.constant
import mon.coreml.typing
from mon.coreml.callback import *
from mon.coreml.constant import *
from mon.coreml.data import *
from mon.coreml.device import *
from mon.coreml.layer import *
from mon.coreml.logger import *
from mon.coreml.loop import *
from mon.coreml.metric import *
from mon.coreml.model import *
from mon.coreml.optimizer import *
from mon.coreml.strategy import *
from mon.coreml.util import *
