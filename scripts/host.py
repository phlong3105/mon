#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import os

from munch import Munch
from pytorch_lightning.plugins import DDPPlugin

from one import datasets_dir
from one import hinet_dehaze_a2i2hazeextra
from one import ModelState
from one import zerodce_lol199

hosts = {
	"lp-desktop-windows": Munch(
		accelerator = "gpu",
		amp_backend = "apex",
		config      = zerodce_lol199,
		gpus        = [0],
		infer_data  = os.path.join(datasets_dir, ""),
		phase       = ModelState.TRAINING,
		strategy    = "dp",  # DDPPlugin(find_unused_parameters=True),
	),
	"lp-labdesktop01-windows": Munch(
		accelerator = "gpu",
		amp_backend = "apex",
		config      = hinet_dehaze_a2i2hazeextra,
		gpus        = [0],
		infer_data  = os.path.join(datasets_dir, "iec", "iec22", "train", "low"),
		phase       = ModelState.TRAINING,
		strategy    = DDPPlugin(find_unused_parameters=True),
	),
	"lp-labdesktop01-ubuntu": Munch(
		accelerator = "gpu",
		amp_backend = "native",
		config      = hinet_dehaze_a2i2hazeextra,
		gpus        = [0],
		infer_data  = os.path.join(datasets_dir, "iec", "iec22", "train", "low"),
		phase       = ModelState.TRAINING,
		strategy    = DDPPlugin(find_unused_parameters=False),
	),
	"lp-labdesktop02-ubuntu": Munch(
		accelerator = "gpu",
		amp_backend = "native",
		config      = "",
		gpus        = [0],
		infer_data  = os.path.join(datasets_dir, ""),
		phase       = ModelState.TRAINING,
		strategy    = DDPPlugin(find_unused_parameters=True),
	),
	"vsw-server02-ubuntu": Munch(
		accelerator = "gpu",
		amp_backend = "native",
		config      = "",
		gpus        = [0, 1],
		infer_data  = os.path.join(datasets_dir, ""),
		strategy    = DDPPlugin(find_unused_parameters=True),
		phase       = ModelState.TRAINING,
	),
	"vsw-server03-ubuntu": Munch(
		accelerator = "gpu",
		amp_backend = "native",
		config      = "",
		gpus        = [0, 1],
		infer_data  = os.path.join(datasets_dir, ""),
		strategy    = DDPPlugin(find_unused_parameters=True),
		phase       = ModelState.TRAINING,
	),
}
