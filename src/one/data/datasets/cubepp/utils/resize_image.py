#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import cv2

from one.core import progress_bar
from one.core import VisionBackend
from one.imgproc import resize
from one.io import create_dirs
from one.io import is_image_file
from one.io import read_image
from one.utils import data_dir

simplecube_dir = os.path.join(data_dir, "cube++", "simplecube++")
splits         = ["train", "test"]


with progress_bar() as pbar:
	for split in splits:
		pattern = os.path.join(simplecube_dir, split, "*", "*")
		for image_path in pbar.track(glob.glob(pattern)):
			if not is_image_file(image_path):
				continue
			image   = read_image(image_path, VisionBackend.CV)
			image   = resize(image, [512, 512, 3])
			image   = image[:, :, ::-1]  # BGR -> RGB
			path    = image_path.replace(f"simplecube++", "simplecube++_512")
			new_dir = Path(path).parent
			create_dirs([new_dir])
			cv2.imwrite(path, image)
