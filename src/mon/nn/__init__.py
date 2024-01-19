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
    Container, functional, init, Module, ModuleDict, ModuleList, ParameterDict,
    ParameterList, Sequential,
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
import mon.nn.data
import mon.nn.device
import mon.nn.factory
import mon.nn.layer
import mon.nn.logger
import mon.nn.loop
import mon.nn.loss
import mon.nn.metric
import mon.nn.model
import mon.nn.optimizer
import mon.nn.parsing
import mon.nn.strategy
import mon.nn.typing
import mon.nn.utils
from mon.nn.callback import *
from mon.nn.data import *
from mon.nn.device import *
from mon.nn.factory import *
from mon.nn.layer import *
from mon.nn.logger import *
from mon.nn.loop import *
from mon.nn.loss import *
from mon.nn.metric import *
from mon.nn.model import *
from mon.nn.optimizer import *
from mon.nn.parsing import *
from mon.nn.strategy import *
from mon.nn.typing import (
    _callable, _ratio_2_t, _ratio_3_t, _ratio_any_t, _scalar_or_tuple_1_t,
    _scalar_or_tuple_2_t, _scalar_or_tuple_3_t, _scalar_or_tuple_4_t,
    _scalar_or_tuple_5_t, _scalar_or_tuple_6_t, _scalar_or_tuple_any_t,
    _size_1_t, _size_2_opt_t, _size_2_t, _size_3_opt_t, _size_3_t, _size_4_t,
    _size_5_t, _size_6_t, _size_any_opt_t, _size_any_t,
)
from mon.nn.utils import *
