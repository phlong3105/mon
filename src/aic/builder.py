#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Factory classes.
"""

from __future__ import annotations

from one import Factory

__all__ = [
	"CAMERAS",
	"DETECTORS",
	"MOTIONS",
	"OBJECTS",
	"TRACKERS",
]


CAMERAS   = Factory(name="cameras")
DETECTORS = Factory(name="object_detectors")
MOTIONS   = Factory(name="motions")
OBJECTS   = Factory(name="objects")
TRACKERS  = Factory(name="trackers")
