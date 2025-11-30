#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for core linear layers.

This module implements various core linear layer components used in neural
networks.
"""

__all__ = [
    "Bilinear",
    "Identity",
    "LazyLinear",
    "Linear",
]

from torch.nn.modules.linear import *
