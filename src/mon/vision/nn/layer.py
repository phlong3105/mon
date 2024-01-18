#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements layer for building vision deep learning models.

This package extends :mod:`mon.nn`. The main advantage of doing so is that we
can easily extract :mod:`mon.nn` without the need for :mod:`mon.vision` package.
"""

from __future__ import annotations

__all__ = [

]

from mon import core

console      = core.console
_current_dir = core.Path(__file__).absolute().parent
