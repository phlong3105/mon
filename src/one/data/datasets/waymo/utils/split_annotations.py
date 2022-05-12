#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from one.io import create_dirs
from one.utils import datasets_dir

"""
'r'       open for reading (default)
'w'       open for writing, truncating the file first
'x'       create a new file and open it for writing
'a'       open for writing, appending to the end of the file if it exists
'b'       binary mode
't'       text mode (default)
'+'       open a disk file for updating (reading and writing)
'U'       universal newline mode (deprecated)
"""
write_mode = "a"
splits     = ["train", "val"]

for split in splits:
	annotation_pattern = os.path.join(datasets_dir, "waymo", "detection2d", split, "annotations", "*.txt")
	annotation_paths   = glob.glob(annotation_pattern)
	
	new_annotation_dirs = [
		os.path.join(datasets_dir, "waymo", "detection2d", split, "front",      "annotations"),
		os.path.join(datasets_dir, "waymo", "detection2d", split, "front_left", "annotations"),
		os.path.join(datasets_dir, "waymo", "detection2d", split, "front_right", "annotations"),
		os.path.join(datasets_dir, "waymo", "detection2d", split, "side_left",  "annotations"),
		os.path.join(datasets_dir, "waymo", "detection2d", split, "side_right", "annotations")
	]
	create_dirs(paths=new_annotation_dirs, recreate=True)
	
	for path in tqdm(annotation_paths, desc=f"{split}"):
		name = Path(path).name
		
		for d in new_annotation_dirs:
			open(os.path.join(d, name), write_mode)
			
		with open(path, "r") as f:
			lines  = f.read().splitlines()
			labels = np.array([line.split() for line in lines], dtype=np.float32)

		for line in lines:
			label  = np.array(line.split(), dtype=np.float32)
			camera = int(label[0])
			new_annotation_dir = ""
			
			if camera == 1:
				new_annotation_dir = os.path.join(datasets_dir, "waymo", "detection2d", split, "front", "annotations")
			elif camera == 2:
				new_annotation_dir = os.path.join(datasets_dir, "waymo", "detection2d", split, "front_left", "annotations")
			elif camera == 3:
				new_annotation_dir = os.path.join(datasets_dir, "waymo", "detection2d", split, "front_right", "annotations")
			elif camera == 4:
				new_annotation_dir = os.path.join(datasets_dir, "waymo", "detection2d", split, "side_left", "annotations")
			elif camera == 5:
				new_annotation_dir = os.path.join(datasets_dir, "waymo", "detection2d", split, "side_right", "annotations")
			
			new_annotation_path = os.path.join(new_annotation_dir, name)
			with open(new_annotation_path, write_mode) as f:
				line += "\n"
				f.writelines(line)
