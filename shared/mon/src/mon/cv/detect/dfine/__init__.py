#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements D-FINE model for object detection.

References:
    - Paper: "D-FINE: Redefine Regression Task of DETRs as Fine-grained
      Distribution Refinement," ICLR 2025.
    - Code: https://github.com/Peterande/D-FINE
"""

__all__ = [
    "DFINE",
]

from .dfine import DFINE
