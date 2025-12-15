#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A package for datasets and data pools mixins.

This package contains mixin classes that provide additional functionality
to data handling classes in the MON framework. Mixins are designed to be
combined with other classes to extend their capabilities without using
traditional inheritance.
"""

__all__ = [
    "DatasetMixin",
    "SAMInstanceMixin",
]

from .base import DatasetMixin
from .label import SAMInstanceMixin
