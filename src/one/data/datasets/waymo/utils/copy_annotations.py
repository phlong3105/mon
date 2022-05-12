#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import glob
import os
from shutil import copyfile

from tqdm import tqdm

from one.utils import datasets_dir

image_pattern = os.path.join(datasets_dir, "waymo", "detection2d", "train", "front_easy", "images", "*.jpeg")

for file in tqdm(glob.glob(image_pattern)):
	annotation_file     = file.replace("front_easy", "front")
	annotation_file     = annotation_file.replace("images", "annotations")
	annotation_file     = annotation_file.replace(".jpeg", ".txt")
	new_annotation_file = annotation_file.replace("front", "front_easy")
	copyfile(annotation_file, new_annotation_file)
