#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for neural network container modules.

This module implements various container modules for neural networks, such as
sequential containers, module lists, and parameter dictionaries. These containers
help organize and manage layers and parameters in a structured way.
"""

__all__ = [
    "Container",
    "Module",
    "ModuleDict",
    "ModuleList",
    "ParameterDict",
    "ParameterList",
    "Sequential",
]

from torch.nn.modules.container import *
from torch.nn.modules.module import *
