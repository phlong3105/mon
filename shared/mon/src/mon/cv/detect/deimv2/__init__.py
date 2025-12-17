#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements DEIMv2 model for object detection.

References:
    - Paper: "Real-Time Object Detection Meets DINOv3," arXiv 2025.
    - Code: https://github.com/Intellindust-AI-Lab/DEIMv2
"""

__all__ = [
    "DEIMv2",
]

from .engine import DEIMv2
