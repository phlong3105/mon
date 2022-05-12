#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Shuffle Files
"""

from __future__ import annotations

import glob
import os
import random
from shutil import copy

from one import create_dirs
from one import datasets_dir
from one import is_txt_file
from one import progress_bar

if __name__ == "__main__":
	image_pattern = os.path.join(
		datasets_dir, "visdrone_uavdt", "test", "images", "*"
	)
	new_images_dir = os.path.join(
		datasets_dir, "visdrone_uavdt", "test", "new_images"
	)
	new_labels_dir = os.path.join(
		datasets_dir, "visdrone_uavdt", "test", "new_labels"
	)
	create_dirs([new_images_dir, new_labels_dir])
	files = {}
	
	with progress_bar() as pbar:
		for file in pbar.track(
			glob.glob(image_pattern), description=f"[bright_yellow]Processing"
		):
			label_file  = file.replace("images", "labels")
			label_file  = label_file.split(".")[0]
			label_file  = label_file + ".txt"
			if not is_txt_file(label_file):
				raise ValueError()
			files[file] = label_file
	
	l = list(files.items())
	random.shuffle(l)
	files = dict(l)
	
	with progress_bar() as pbar:
		for i, (f, l) in enumerate(files.items()):
			image_file = os.path.join(new_images_dir, f"{i:05}" + f".{f.split('.')[-1]}")
			label_file = os.path.join(new_labels_dir, f"{i:05}" + f".{l.split('.')[-1]}")
			copy(f, image_file)
			copy(l, label_file)
