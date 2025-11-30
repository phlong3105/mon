#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package implements data structure and processing functions for instance
annotation.

One instance can have these kinds of annotations.
    - Bounding box: A rectangle around the object.
    - Oriented bounding box: A rotated rectangle around the object.
    - Polygon: Points to outline the object's shape.
    - Polyline: Lines to mark edges or paths.
    - Keypoints: Points on key parts, like eyes or joints.
    - Instance mask: Pixels that belong to the object.
    - 3D cuboid: Box with depth for 3D view.
    - Attributes: Extra labels, like color or size.
    - Class label: The type of object, like "car".
"""

__all__ = [
    "Instance",
]

from .core import Instance
