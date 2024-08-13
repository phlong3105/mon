#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements base classes for all datasets.

For transformation operations, we use
`albumentations <https://albumentations.ai/docs/api_reference/full_reference>`__
"""

from __future__ import annotations

import mon.data.datastruct.dataset.base
# import mon.data.datastruct.dataset.classification
# import mon.data.datastruct.dataset.detection
import mon.data.datastruct.dataset.image
# import mon.data.datastruct.dataset.segmentation
import mon.data.datastruct.dataset.video
from mon.data.datastruct.dataset.base import *
# from mon.data.datastruct.dataset.classification import *
# from mon.data.datastruct.dataset.detection import *
from mon.data.datastruct.dataset.image import *
# from mon.data.datastruct.dataset.segmentation import *
from mon.data.datastruct.dataset.video import *
