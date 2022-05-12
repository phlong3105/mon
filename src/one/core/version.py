#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import torch
from packaging import version

__all__ = [
	"torch_version",
	"torch_version_geq",
]


# MARK: - Functional

def torch_version() -> str:
	"""Parse the `torch.__version__` variable and removes +cu*/cpu."""
	return torch.__version__.split('+')[0]


def torch_version_geq(major, minor) -> bool:
	_version = version.parse(torch_version())
	return _version >= version.parse(f"{major}.{minor}")
