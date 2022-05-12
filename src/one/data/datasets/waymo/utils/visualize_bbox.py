#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import glob
import os
import random

import cv2
import numpy as np

from one.data.data_class import ClassLabels
from one.data.data_class import ImageInfo
from one.imgproc import box_cxcywh_norm_to_xyxy
from one.utils import datasets_dir

# splits = ["train", "val"]
batch_size      = 8
splits          = ["train"]
annotations_dir = "annotations_yolo"

for split in splits:
	annotation_pattern = os.path.join(datasets_dir, "waymo", "detection2d", split, "front_easy", annotations_dir, "*.txt")
	annotation_paths   = glob.glob(annotation_pattern)
	indices            = [random.randint(0, len(annotation_paths) - 1) for _ in range(batch_size)]
	annotation_paths   = [annotation_paths[i] for i in indices]
	class_labels_path    = os.path.join(datasets_dir, "waymo", "detection2d", f"class_labels.json")
	class_labels        = ClassLabels.create_from_file(label_path=class_labels_path)
	
	for path in annotation_paths:
		image_path = path.replace(annotations_dir, "images")
		image_path = image_path.replace(".txt", ".jpeg")
		image_info = ImageInfo.from_image_file(image_path=image_path)
		shape0     = image_info.shape0
		
		with open(path, "r") as fi:
			labels = np.array([line.split() for line in fi.read().splitlines()], dtype=np.float32)
		
		image          = cv2.imread(image_path)  # BGR
		labels[:, 1:5] = box_cxcywh_norm_to_xyxy(labels[:, 1:5], shape0[0], shape0[1])
		colors         = class_labels.colors()
		print(labels)
		
		for i, l in enumerate(labels):
			class_id    = int(l[0])
			start_point = l[1:3].astype(np.int)
			end_point   = l[3:5].astype(np.int)
			color       = colors[class_id] if isinstance(colors, (tuple, list)) and len(colors) >= class_id else (255, 255, 255)
			image       = cv2.rectangle(image, start_point, end_point, color, 5)
		
		cv2.imshow("AA", image)
		cv2.waitKey(0)
