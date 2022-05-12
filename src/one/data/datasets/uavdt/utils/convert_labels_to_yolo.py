#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import glob
import os.path
from pathlib import Path
from shutil import copy

import numpy as np

from one.core import progress_bar
from one.core import VisionBackend
from one.imgproc import box_xywh_to_cxcywh_norm
from one.io import create_dirs
from one.io import read_image
from one.utils import datasets_dir

split = "train"
label_pattern = os.path.join(
	datasets_dir, "uavdt", "mot", split, "labels", "*", "*.txt"
)

with progress_bar() as pbar:
	for file in pbar.track(
		glob.glob(label_pattern), description=f"[bright_yellow]Converting labels"
	):
		image_file = file.replace("labels", "images")
		image_file = image_file.replace(".txt", ".jpg")
		image      = read_image(image_file, backend=VisionBackend.CV)  # RGB
		image      = image[:, :, ::-1]  # BGR
		image      = np.ascontiguousarray(image, dtype=np.uint8)
		h, w, c    = image.shape
		
		subdir         = os.path.basename(os.path.dirname(file).split("_")[0])
		new_image_file = image_file.replace(split, f"{split}_yolo")
		new_image_file = new_image_file.replace(f"{subdir}/", f"{subdir}_")
		yolo_file      = file.replace(split, f"{split}_yolo")
		yolo_file      = yolo_file.replace(f"{subdir}/", f"{subdir}_")
		print(new_image_file)
		new_image_dir  = Path(new_image_file).parent
		label_dir      = Path(yolo_file).parent
		create_dirs([new_image_dir, label_dir])
		copy(image_file, new_image_file)
		
		with open(file, "r") as f:
			lines = []
			for line in f:
				values       = line.rstrip().split(",")
				x, y, bw, bh = values[2:6]
				id           = values[-1]
				if id == "1":
					lines.append([x, y, bw, bh])
				# image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
			
			if len(lines) > 0:
				labels = np.array(lines, np.float32)
				labels = box_xywh_to_cxcywh_norm(labels, h, w)
		
		with open(yolo_file, mode="w") as f:
			for l in labels:
				f.write(f"0 {l[0]} {l[1]} {l[2]} {l[3]}\n")
