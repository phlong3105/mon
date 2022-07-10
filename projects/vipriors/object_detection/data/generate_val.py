#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""To generated val and new train set from original train set.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Union

from projects.vipriors.utils import data_dir


def generate_val(
	root           : Union[str, Path],
	destination    : Union[str, Path],
	train_json_path: Union[str, Path],
	val_size       : int
):
	src_path    = os.path.join(root, "train", "images")
	destination = os.path.join(root, destination, "images")
	img_list    = os.listdir(src_path)
	val_list    = img_list[: val_size]
	
	# Validation set generation
	if not os.path.exists(destination):
		os.makedirs(destination)
	
	for img in val_list:
		shutil.move(os.path.join(src_path, img), destination)
	
	valset_list = os.listdir(destination)
	if len(valset_list) == 1000:
		print("Validation images are successfully generated.")
	
	json_data = json.load(open(os.path.join(root, train_json_path)))
	
	val_dict = {}
	for im_name in valset_list:
		val_dict[im_name] = json_data[im_name]
	
	with open(os.path.join(root, "val_annotations.json"), "w") as outfile:
		json.dump(val_dict, outfile)
	print("Validation labels are successfully generated.")
	
	# Training set after moving validation images
	new_train_list = os.listdir(src_path)
	new_train_dict = {}
	for im_name in new_train_list:
		new_train_dict[im_name] = json_data[im_name]
	
	with open(os.path.join(root, "new_train_annotations.json"), "w") as outfile:
		json.dump(val_dict, outfile)
	print("New training labels are successfully generated.")


# MARK: - Main

if __name__ == "__main__":
	generate_val(
		root            = os.path.join(data_dir, "delftbikes"),
		destination     = "val",
		train_json_path = os.path.join(data_dir, "delftbikes", "train", "train_annotations.json"),
		val_size        = 1000
	)
