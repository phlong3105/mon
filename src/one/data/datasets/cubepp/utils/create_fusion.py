#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import glob
import os
from shutil import copyfile

from one.core import progress_bar
from one.io import create_dirs
from one.utils import datasets_dir

simplecube_dir = os.path.join(datasets_dir, "cube++", "simplecube++")
cube_dir       = os.path.join(datasets_dir, "cube++", "cube++")
splits         = ["train", "test"]


with progress_bar() as pbar:
	for split in splits:
		pattern = os.path.join(simplecube_dir, split, "png", "*")
		
		for image_path in pbar.track(glob.glob(pattern)):
			# png_name = image_path.split(".")[0]
			jpg      = image_path.replace(f"{split}\\", "")
			jpg      = jpg.replace(f"simplecube", "cube")
			jpg      = jpg.replace(f"png", "jpg")
			new_jpg  = image_path.replace(f"png", "jpg")
			create_dirs(os.path.join(simplecube_dir, split, "jpg"))
			copyfile(jpg, new_jpg)
