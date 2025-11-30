#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A package for thermal map data type.

This package provides a data structure for handling thermal (infrared) maps,
which represent temperature distributions in images.
"""

__all__ = [
    "InfraredMap",
]

from .core import InfraredMap
