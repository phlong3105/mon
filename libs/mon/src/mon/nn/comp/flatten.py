#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Flattening layers.

This module implements various flattening layers commonly used for flattening a
continuous range of dims into a tensor.
"""

__all__ = [
    "Flatten",
    "Unflatten",
]

from torch.nn.modules.flatten import *
