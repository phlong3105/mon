#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package implements the basic functionalities for machine learning and
deep learning research. It provides support for high-level machine learning
packages, such as vision, natural language, or speech.

:mod:`mon.nn` package itself is built on top `PyTorch` and `Lightning`
libraries.
"""

from __future__ import annotations

# Interface to :mod:`torch.nn`. We import commonly used components so that
# everything can be accessed under one single import of ``from mon import nn``.
# noinspection PyUnresolvedReferences
from torch.nn import (
    Container, functional, init, Module, ModuleDict,
    ModuleList, ParameterDict, ParameterList, Sequential,
)
# noinspection PyUnresolvedReferences
from torch.nn.parallel import DataParallel as DataParallel
# noinspection PyUnresolvedReferences
from torch.nn.parameter import (
    Parameter as Parameter,
    UninitializedBuffer as UninitializedBuffer,
    UninitializedParameter as UninitializedParameter,
)

# Import
import mon.nn.callback
import mon.nn.factory
import mon.nn.layer
import mon.nn.logger
import mon.nn.loss
import mon.nn.metric
import mon.nn.model
import mon.nn.optimizer
import mon.nn.prior
import mon.nn.runner
import mon.nn.strategy
import mon.nn.thop
import mon.nn.utils
from mon.nn.callback import *
from mon.nn.factory import *
from mon.nn.layer import *
from mon.nn.logger import *
from mon.nn.loss import *
from mon.nn.metric import *
from mon.nn.model import *
from mon.nn.optimizer import *
from mon.nn.prior import *
from mon.nn.runner import *
from mon.nn.strategy import *
from mon.nn.thop import *
from mon.nn.utils import *
