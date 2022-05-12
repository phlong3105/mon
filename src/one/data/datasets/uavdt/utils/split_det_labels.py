#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import glob
import os.path
from pathlib import Path

import numpy as np

from one.core import progress_bar
from one.core import VisionBackend
from one.io import create_dirs
from one.io import read_image
from one.utils import datasets_dir

split = "test"
label_pattern = os.path.join(
	datasets_dir, "uavdt", "mot", split, "annotations", "*_gt_whole.txt"
)

with progress_bar() as pbar:
	for file in pbar.track(
		glob.glob(label_pattern), description=f"[bright_yellow]Converting labels"
	):
		line_dict = {}
		with open(file) as f:
			for line in f:
				values   = line.rstrip().split(",")
				image_id = values[0].zfill(6)
				if image_id not in line_dict:
					line_dict[image_id] = [line]
				else:
					line_dict[image_id].append(line)
			f.close()
		
		image_subdir = os.path.basename(file).split("_")[0]
		for k, v in line_dict.items():
			image_file = os.path.join(
				datasets_dir, "uavdt", "mot", split, "images", image_subdir, f"img{k}.jpg"
			)
			image   = read_image(image_file, backend=VisionBackend.CV)  # RGB
			image   = image[:, :, ::-1]  # BGR
			image   = np.ascontiguousarray(image, dtype=np.uint8)
			h, w, c = image.shape

			label_file = os.path.join(
				datasets_dir, "uavdt", "mot", split, "labels", image_subdir, f"img{k}.txt"
			)
			label_dir  = Path(label_file).parent
			create_dirs([label_dir])
			with open(label_file, "w") as w:
				for l in v:
					w.write(f"{l}")
