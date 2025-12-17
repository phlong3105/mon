#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for padding layers.

This module provides various padding layers commonly used in convolutional neural
networks (CNNs) to adjust the spatial dimensions of input feature maps.
"""

__all__ = [
    "CircularPad1d",
    "CircularPad2d",
    "CircularPad3d",
    "ConstantPad1d",
    "ConstantPad2d",
    "ConstantPad3d",
    "ReflectionPad1d",
    "ReflectionPad2d",
    "ReflectionPad3d",
    "ReplicationPad1d",
    "ReplicationPad2d",
    "ReplicationPad3d",
    "ZeroPad1d",
    "ZeroPad2d",
    "ZeroPad3d",
]

from torch.nn.modules.padding import *
