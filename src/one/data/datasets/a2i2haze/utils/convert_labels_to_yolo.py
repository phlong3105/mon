#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import glob
import os.path

import numpy as np

from one.core import progress_bar
from one.core import VisionBackend
from one.imgproc import box_xyxy_to_cxcywh_norm
from one.io import read_image
from one.utils import datasets_dir

label_pattern = os.path.join(
	datasets_dir, "a2i2haze", "dryrun", "labels", "*.txt"
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
		
		yolo_file = file.replace(
			"labels", "labels_yolo"
		)
		
		with open(file) as f:
			lines = []
			for line in f:
				values         = line.rstrip().split(" ")
				x1, y1, x2, y2 = values[1:5]
				lines.append([x1, y1, x2, y2])
				# image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
				
			labels = np.array(lines, np.float32)
			labels = box_xyxy_to_cxcywh_norm(labels, h, w)
		
		with open(yolo_file, mode="w") as f:
			for l in labels:
				f.write(f"0 {l[0]} {l[1]} {l[2]} {l[3]}\n")
