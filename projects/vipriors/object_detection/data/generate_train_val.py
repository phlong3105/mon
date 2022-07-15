#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""To generated val and new train set from original train set.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Union

import numpy as np

from one.imgproc import box_xyxy_to_cxcywh_norm
from one.io import create_dirs
from one.io import load_file
from projects.vipriors.utils import data_dir


def generate_train_val(
	root       : Union[str, Path],
	json_file  : Union[str, Path],
	val_size   : int,
	yolo_labels: bool = True
):
	src_path   = os.path.join(root, "trainval", "images")
	train_path = os.path.join(root, "train",    "images")
	val_path   = os.path.join(root, "val",      "images")
	img_list   = os.listdir(src_path)
	val_list   = img_list[: val_size]
	train_list = img_list[val_size :]
	json_data  = json.load(open(os.path.join(root, "trainval", json_file)))
	
	# NOTE: Val set generation
	create_dirs(paths=[val_path])
	for img in val_list:
		shutil.copy(os.path.join(src_path, img), val_path)
	
	valset_list = os.listdir(val_path)
	if len(valset_list) == 1000:
		print("Val images are successfully generated.")
	
	val_dict = {}
	for im_name in valset_list:
		val_dict[im_name] = json_data[im_name]
	
	with open(os.path.join(root, "val", "val_annotations.json"), "w") as outfile:
		json.dump(val_dict, outfile)
	
	if yolo_labels:
		generate_yolo_labels(
			root      = os.path.join(root, "val"),
			json_file = "val_annotations.json"
		)
	print("Val labels are successfully generated.")
	
	# NOTE: Train set generation
	create_dirs(paths=[train_path])
	for img in train_list:
		shutil.copy(os.path.join(src_path, img), train_path)
		
	trainset_list = os.listdir(train_path)
	if len(trainset_list) == 7000:
		print("Train images are successfully generated.")
		
	train_dict = {}
	for im_name in train_list:
		train_dict[im_name] = json_data[im_name]
	
	with open(os.path.join(root, "train", "train_annotations.json"), "w") as outfile:
		json.dump(train_dict, outfile)
	
	if yolo_labels:
		generate_yolo_labels(
			root      = os.path.join(root, "train"),
			json_file = "train_annotations.json"
		)
	print("Train labels are successfully generated.")


def generate_yolo_labels(root: Union[str, Path], json_file: Union[str, Path]):
	# images    = list(sorted(os.listdir(os.path.join(root, "images"))))
	json_data   = load_file(os.path.join(root, json_file))
	labels_path = os.path.join(root, "yolo_labels")
	
	create_dirs(paths=[labels_path])
	
	for k, v in json_data.items():
		image_path = os.path.join(root, "images", k)
		image      = v["image"]
		channels   = image["channels"]
		height     = image["height"]
		width      = image["width"]

		label_ids   = []
		boxes       = []
		part_names  = []
		conf_scores = []
		for idx, i in enumerate(v["parts"], 0):
			label = v["parts"][i]
			label_ids.append(idx)
			part_names.append(label["part_name"])
			conf_scores.append(label["trust"])
			if label["object_state"] != "absent":
				loc = label["absolute_bounding_box"]
				x1  = loc["left"]
				x2  = loc["left"] + loc["width"]
				y1  = loc["top"]
				y2  = loc["top"] + loc["height"]
				boxes.append([x1, y1, x2, y2])
		
		boxes = np.array(boxes, np.float32)
		boxes = box_xyxy_to_cxcywh_norm(boxes, height, width)
		
		yolo_file = os.path.join(labels_path, k.replace(".jpg", ".txt"))
		with open(yolo_file, mode="w") as f:
			for i, b in enumerate(boxes):
				f.write(f"{label_ids[i]} {b[0]} {b[1]} {b[2]} {b[3]} {conf_scores[i]}\n")
		

# MARK: - Main

if __name__ == "__main__":
	generate_train_val(
		root        = os.path.join(data_dir, "delftbikes"),
		json_file   = "trainval_annotations.json",
		val_size    = 1000,
		yolo_labels = True
	)
