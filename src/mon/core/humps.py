#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Humps Module.

This module extends :obj:`humps`.
"""

from __future__ import annotations

__all__ = [
	"camelize",
	"decamelize",
	"dekebabize",
	"depascalize",
	"is_camelcase",
	"is_kebabcase",
	"is_pascalcase",
	"is_snakecase",
	"kebabize",
	"pascalize",
	"snakecase",
]

from humps import *


def snakecase(x: str) -> str:
	"""Convert a string to snakecase."""
	x = x.replace(" ", "_")
	x = x.replace("-", "_")
	return x
