#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Datasets Templates.

This module implements base classes for all datasets.

For transformation operations, we use
`albumentations <https://albumentations.ai/docs/api_reference/full_reference>`__
"""

from __future__ import annotations

import mon.core.data.dataset.base
import mon.core.data.dataset.image
import mon.core.data.dataset.video
from mon.core.data.dataset.base import *
from mon.core.data.dataset.image import *
from mon.core.data.dataset.video import *
