#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Zero-Restore

This module implements the paper: Zero-shot Single Image Restoration through
Controlled Perturbation of Koschmieder's Model.

References:
	https://github.com/aupendu/zero-restore
"""

from __future__ import annotations

import mon.vision.enhance.multitask.zero_restore.zero_restore_dehaze
import mon.vision.enhance.multitask.zero_restore.zero_restore_llie
import mon.vision.enhance.multitask.zero_restore.zero_restore_uie
from mon.vision.enhance.multitask.zero_restore.zero_restore_dehaze import *
from mon.vision.enhance.multitask.zero_restore.zero_restore_llie import *
from mon.vision.enhance.multitask.zero_restore.zero_restore_uie import *
