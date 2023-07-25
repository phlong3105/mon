#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module is an interface to :mod:`torch.nn`. We import commonly used
components so that everything can be access under one single import of
``from mon import coreml as nn``.
"""

from __future__ import annotations

__all__ = [
    "Container", "DataParallel", "Module", "ModuleDict", "ModuleList",
    "Parameter", "ParameterDict", "ParameterList", "Sequential",
    "UninitializedBuffer", "UninitializedParameter", "functional", "init",
    "utils",
]

# noinspection PyUnresolvedReferences
from torch.nn import (
    Container, functional, init, Module, ModuleDict, ModuleList, ParameterDict,
    ParameterList, Sequential, utils,
)
# noinspection PyUnresolvedReferences
from torch.nn.parallel import DataParallel as DataParallel
# noinspection PyUnresolvedReferences
from torch.nn.parameter import (
    Parameter as Parameter, UninitializedBuffer as UninitializedBuffer,
    UninitializedParameter as UninitializedParameter,
)
