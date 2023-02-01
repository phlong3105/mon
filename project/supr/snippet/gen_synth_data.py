#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import glob
import math
import os.path
import random

import cv2

from onevision import box_xyxy_to_cxcywh_norm
from onevision import create_dirs
from onevision import datasets_dir
from onevision import get_image_hw
from onevision import is_image_file
from onevision import progress_bar
from onevision import random_patch_numpy_image_box
from onevision import read_image
from onevision import VisionBackend

random.seed(0)


def draw_rect(im, cords, color=None):
	im    = im.copy()
	cords = cords[:, :4]
	cords = cords.reshape(-1, 4)
	
	if not color:
		color = [255, 255, 255]
	for cord in cords:
		
		pt1, pt2 = (cord[0], cord[1]), (cord[2], cord[3])
		
		pt1 = int(pt1[0]), int(pt1[1])
		pt2 = int(pt2[0]), int(pt2[1])
		
		im = cv2.rectangle(im.copy(), pt1, pt2, color, 3)
	return im


image_file_pattern = os.path.join(datasets_dir, "aicity", "aic22retail", "train", "images", "*")
generate_image_dir = os.path.join(datasets_dir, "aicity", "aic22retail", "train", "generate_images")
generate_label_dir = os.path.join(datasets_dir, "aicity", "aic22retail", "train", "generate_labels")
create_dirs([generate_image_dir, generate_label_dir])

# NOTE: Get list of image sizes
image_files        = []
segmentation_files = []
sizes              = {}

with progress_bar() as pbar:
	for image_file in pbar.track(
		glob.glob(image_file_pattern),
		description=f"[bright_yellow]Listing files"
	):
		file_name         = image_file.split(".")[0]
		segmentation_file = f"{file_name}_seg.jpg"
		segmentation_file = segmentation_file.replace("images", "segmentation_labels")
		
		if is_image_file(image_file) and is_image_file(segmentation_file):
			# image        = read_image(image_file,        VisionBackend.CV)
			# segmentation = read_image(segmentation_file, VisionBackend.CV)
			# cv2.imshow("image", image)
			# cv2.imshow("segmentation", segmentation)
			# cv2.waitKey(0)
			image_files.append(image_file)
			segmentation_files.append(segmentation_file)
			# sizes[image_file] = get_image_size(image)

# NOTE: Shuffle
d = {}
for image, segment in zip(image_files, segmentation_files):
	d[image] = segment

keys = list(d.keys())
random.shuffle(keys)
shuffled_d = {}
for key in keys:
	shuffled_d[key] = d[key]

image_files        = list(shuffled_d.keys())
segmentation_files = list(shuffled_d.values())

# NOTE: Generate train images and labels for training YOLO models.
background          = read_image("../data/background.png", VisionBackend.CV)
num_items           = 7
num_generate_images = math.floor(len(image_files) / num_items)

with progress_bar() as pbar:
	item_idx = 0
	for i in pbar.track(
		range(num_generate_images), description=f"[bright_yellow]Generating files"
	):
		images   = []
		segments = []
		ids      = []
		for j in range(item_idx, item_idx + num_items):
			ids.append(int(os.path.basename(image_files[j]).split("_")[0]) - 1)
			images.append(read_image(image_files[j], VisionBackend.CV))
			segments.append(read_image(segmentation_files[j], VisionBackend.CV))
		
		gen_image, boxes = random_patch_numpy_image_box(
			canvas  = background.copy(),
			patch   = images,
			mask    = segments,
			id      = ids,
			angle   = [0, 360],
			scale   = [1.0, 1.0],
			gamma   = [0.9, 1.0],
			overlap = 0.10,
		)
		gen_image      = gen_image[:, :, ::-1]
		gen_image_file = os.path.join(generate_image_dir, f"{i:06}.jpg")
		cv2.imwrite(gen_image_file, gen_image)
		
		h, w           = get_image_hw(gen_image)
		boxes[:, 1:5]  = box_xyxy_to_cxcywh_norm(boxes[:, 1:5], h, w)
		gen_label_file = os.path.join(generate_label_dir, f"{i:06}.txt")
		
		with open(gen_label_file, "w") as f:
			for b in boxes:
				f.write(f"{int(b[0])} {b[1]} {b[2]} {b[3]} {b[4]}\n")
		
		item_idx += num_items
		
		"""
		result = draw_rect(gen_image, boxes)
		cv2.imwrite("result.jpg", result)
		cv2.imshow("gen_image", result)
		cv2.waitKey(0)
		"""
