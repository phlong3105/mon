#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Flattening layers.

This module provides various flattening layers commonly used for flattening a
continuous range of dims into a tensor.
"""

from __future__ import annotations

__all__ = [
    "Flatten",
    "Unflatten",
]

from torch.nn.modules.flatten import *
