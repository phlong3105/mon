#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import inspect
import sys

import torch
from packaging import version


# MARK: - Functional

def torch_version() -> str:
	"""Parse the `torch.__version__` variable and removes +cu*/cpu."""
	return torch.__version__.split('+')[0]


def torch_version_geq(major, minor) -> bool:
	_version = version.parse(torch_version())
	return _version >= version.parse(f"{major}.{minor}")


# MARK: - Main

__all__ = [
    name for name, value in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__
    )
]
