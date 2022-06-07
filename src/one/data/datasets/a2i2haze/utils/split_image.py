#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import glob
import os.path

import cv2

from one.core import progress_bar
from one.core import VisionBackend
from one.io import read_image
from one.utils import data_dir

image_pattern = os.path.join(
	data_dir, "a2i2haze", "train", "haze_clean_images", "*.jpg"
)
clean_dir = os.path.join(
	data_dir, "a2i2haze", "train", "clean_images"
)
haze_dir = os.path.join(
	data_dir, "a2i2haze", "train", "haze_images"
)

with progress_bar() as pbar:
	for p in pbar.track(
		glob.glob(image_pattern), description=f"[bright_yellow]Splitting images"
	):
		image   = read_image(p, backend=VisionBackend.CV)  # RGB
		image   = image[:, :, ::-1]  # BGR
		h, w, c = image.shape
		haze    = image[0 : int(h / 2), 0 : w, 0 : c]
		clean   = image[int(h / 2) : h, 0 : w, 0 : c]
		name    = os.path.basename(p)
		cv2.imwrite(os.path.join(clean_dir, name), clean)
		cv2.imwrite(os.path.join(haze_dir, name),  haze)
		
		# cv2.imshow("Haze", haze)
		# cv2.imshow("Clean", clean)
		# cv2.waitKey(0)
