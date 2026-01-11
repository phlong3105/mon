#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for data pools.

This module implements data pools (i.e., data reservoirs that support appending
and sampling datapoint).

This is primarily used in various training pipelines (e.g., active learning,
copy-paste, etc.) to manage labelled and unlabeled data. In a way, data pools
are suitable for tasks that works with object/instance level annotations.
"""

# __all__ = []  

from .image import *
