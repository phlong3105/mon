#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for flattening layers.

This module provides classes for flattening and unflattening tensors in neural
networks.
"""

__all__ = [
    "Flatten",
    "Unflatten",
]

from torch.nn.modules.flatten import *
