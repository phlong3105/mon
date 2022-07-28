from __future__ import annotations

import argparse
import glob
import multiprocessing
import os

import cv2
import numpy
from joblib import delayed
from joblib import Parallel
from tqdm import tqdm

from one.imgproc import box_xywh_to_cxcywh_norm
from one.io import create_dirs
from one.io import list_subdirs
from one.utils import data_dir

"""VisDrone measurement label format:

	<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>

Where:
	<bbox_left>      : Fx coordinate of the top-left corner of the predicted bounding box.
	<bbox_top>       : Fy coordinate of the top-left corner of the predicted object bounding box.
	<bbox_width>     : Width in pixels of the predicted object bounding box.
	<bbox_height>    : Height in pixels of the predicted object bounding box.
	<score>          : Fscore in the DETECTION result file indicates the confidence of the predicted bounding box
					   enclosing an object measurement.Fscore in GROUNDTRUTH file is set to 1 or 0. 1 indicates the bounding
					   box is considered in evaluation, while 0 indicates the bounding box will be ignored.
	<object_category>: Object category indicates the type of annotated object, (i.e., ignored regions (0),
					   pedestrian (1), people (2), bicycle (3), car (4), van (5), truck (6), tricycle (7),
					   awning-tricycle (8), bus (9), motor (10), others (11))
	<truncation>     : Fscore in the DETECTION result file should be set to the constant -1. Fscore in the
	                   GROUNDTRUTH file indicates the degree of object parts appears outside a frame
	                   (i.e., no truncation = 0 (truncation ratio 0%), and partial truncation = 1(truncation ratio 1% ∼ 50%)).
	<occlusion>      : Fscore in the DETECTION result file should be set to the constant -1. Fscore in the
	                   GROUNDTRUTH file indicates the fraction of objects being occluded
	                   (i.e., no occlusion = 0 (occlusion ratio 0%), partial occlusion = 1(occlusion ratio 1% ∼ 50%),
	                   and heavy occlusion = 2 (occlusion ratio 50% ~ 100%)).

Examples:
	- For example for `img1.jpg` you will be created `img1.txt` containing:
		684,8,273,116,0,0,0,0
		406,119,265,70,0,0,0,0
		255,22,119,128,0,0,0,0
"""

"""YOLO label format:

	<object-class> <x_center> <y_center> <width> <height>

Where:
	<object-class>: integer object number from 0 to (classes-1).
	<x_center> <y_center>: are center of rectangle (not top-left corner).
	<x_center> <y_center> <width> <height>: float values relative to width and height of image, it can be equal from (0.0 to 1.0]
		- <x>      = <absolute_x>      / <image_width>
		- <y>      = <absolute_y>      / <image_height>
		- <width>  = <absolute_width>  / <image_width>
		- <height> = <absolute_height> / <image_height>
	
Examples:
	- For example for `img1.jpg` you will be created `img1.txt` containing:
		1 0.716797 0.395833 0.216406 0.147222
		0 0.687109 0.379167 0.255469 0.158333
		1 0.420312 0.395833 0.140625 0.166667

References: https://github.com/AlexeyAB/darknet#datasets
"""


# MARK: - Functional Interface

def convert_detection_labels(root_dir: str, format: str = "yolo"):
	"""Convert all labels in `train/val/testdev/testchallenge` to YOLO format.
	
	Args:
		root_dir (str):
			Root directory that contains `train/val/testdev/testchallenge` directories.
		format (str):
			Format to convert the labels to.
	"""
	# TODO: Check for `train/val/testdev` directories
	subdirs = list_subdirs(root_dir)
	if not any(item in subdirs for item in ["train", "val", "testdev"]):
		raise ValueError()
	
	# TODO: Create output directories
	output_dirs = [
		os.path.join(root_dir, "train",   "annotations_yolo"),
		os.path.join(root_dir, "val",     "annotations_yolo"),
		os.path.join(root_dir, "testdev", "annotations_yolo"),
		# os.path.join(root_dir, "testchallenge", "annotations_yolo"),
	]
	create_dirs(output_dirs, recreate=True)
	
	# TODO: List all images
	image_patterns = [
		os.path.join(data_dir, "train",         "images", "*.jpg"),
		os.path.join(data_dir, "val",           "images", "*.jpg"),
		os.path.join(data_dir, "testdev",       "images", "*.jpg"),
		# os.path.join(data_dir, "testchallenge", "images", "*.jpg")
	]
	
	image_files = []
	for pattern in image_patterns:
		image_files += glob.glob(pattern)
	
	# TODO: List all labels files and convert
	num_jobs = multiprocessing.cpu_count()
	if format == "yolo":
		labels_files = [image_file.replace("images", "annotations") for image_file  in image_files ]
		labels_files = [labels_file.replace(".jpg", ".txt")         for labels_file in labels_files]

		Parallel(n_jobs=num_jobs)(
			delayed(_convert_labels_yolo)(image_file, labels_file)
			for (image_file, labels_file) in tqdm(zip(image_files, labels_files))
		)
		
	elif format == "coco":
		pass
	
		
def _convert_labels_yolo(image_file: str, labels_file: str):
	"""Convert all labels in the `labels_file`.
	
	Args:
		image_file (str):
			Image file corresponding to the labels.
		labels_file (str):
			`.txt` file that contains labels.
	"""
	image = cv2.imread(image_file)  # BGR
	h, w  = image.shape[:2]         # Image HW
	
	labels_yolo_file = labels_file.replace("annotations", "annotations_yolo")
	
	with open(labels_file, "r") as file_in, open(labels_yolo_file, "w") as file_out:
		labels = [x.replace(",", " ")   for x in file_in.read().splitlines()]
		labels = numpy.array([x.split() for x in labels], dtype=numpy.float32)
		boxes  = labels[:, 0:4].copy()
		boxes  = box_xywh_to_cxcywh_norm(boxes, h, w)
		
		for l, b in zip(labels, boxes):
			ss = f"{int(l[5])} {b[0]} {b[1]} {b[2]} {b[3]} {int(l[4])} {int(l[6])} {int(l[7])}\n"
			file_out.writelines(ss)
	

# MARK: - Main

if __name__ == "__main__":
	
	# TODO: Arguments parsers
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", type=str, default="",     help="Path to the dataset directory that contains train/val/testdev/testchallenge directories.")
	parser.add_argument("--format",   type=str, default="yolo", help="Format to convert the labels to.")
	args = parser.parse_args()
	
	if args.data_dir == "":
		data_dir = os.path.join(data_dir, "visdrone", "det2019")
	else:
		data_dir = args.data_dir
	
	convert_detection_labels(root_dir=data_dir, format=args.format)
