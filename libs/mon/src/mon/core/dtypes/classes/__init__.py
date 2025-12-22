#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Class data type.

This package contains a "full-stack" toolkit for class data, including data
structure, ingestion, analysis, atomic transformations, complex workflows, and
rendering utilities.
"""

__all__ = [
    "Class",
    "ClassList",
    "Probabilities",
    "class_id_to_one_hot",
]

from .core import *
from .io import *
from .meta import *
from .ops import *
from .proc import *
from .vis import *
