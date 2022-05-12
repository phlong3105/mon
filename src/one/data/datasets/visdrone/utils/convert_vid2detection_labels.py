from __future__ import annotations

import argparse
import glob
import multiprocessing
import os
import random
from shutil import copyfile

import cv2
import numpy
from joblib import delayed
from joblib import Parallel
from tqdm import tqdm

from one.imgproc import box_xywh_to_cxcywh_norm
from one.io import create_dirs
from one.io import list_subdirs
from one.utils import datasets_dir

"""VisDrone's VID label format:

	<frame_index>,<object_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>

Where:
	<frame_index>    : Fframe index in the sequence.
	<object_id>      : Object id.
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
		188,36,558,248,21,55,1,1,0,1
		189,36,558,249,21,55,1,1,0,1
		190,36,558,250,21,55,1,1,0,1
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


# MARK: - Convert

def convert_vid2detection_labels(root_dir: str, format: str = "yolo"):
	"""Convert all labels in `train/val/testdev/testchallenge` to YOLO format.

	Args:
		root_dir (str):
			Root directory that contains `train/val/testdev/testchallenge` directories.
		format (str):
			Format to convert the labels to.
			Default: `yolo`.
	"""
	# TODO: Check for `train/val/testdev` directories
	subdirs = list_subdirs(path=root_dir)
	assert any(item in subdirs for item in ["train", "val", "testdev"])
	
	# TODO: Create output directories
	output_dirs = [
		os.path.join(root_dir, "train",         "images"),
		os.path.join(root_dir, "val",           "images"),
		os.path.join(root_dir, "testdev",       "images"),
		# os.path.join(root_dir, "testchallenge", "images"),
		os.path.join(root_dir, "train",          f"annotations_{format}"),
		os.path.join(root_dir, "val",            f"annotations_{format}"),
		os.path.join(root_dir, "testdev",        f"annotations_{format}"),
		# os.path.join(root_dir, "testchallenge",  f"annotations_{format}"),
	]
	create_dirs(paths=output_dirs, recreate=True)
	
	# TODO: List all images
	sequence_dir_patterns = [
		os.path.join(root_dir, "train",   "sequences", "*"),
		os.path.join(root_dir, "val",     "sequences", "*"),
		os.path.join(root_dir, "testdev", "sequences", "*"),
		# os.path.join(root_dir, "testchallenge", "images", "*.jpg")
	]
	
	sequence_dirs = []
	for pattern in sequence_dir_patterns:
		sequence_dirs += glob.glob(pattern)
	
	# TODO: List all labels files and convert
	num_jobs = multiprocessing.cpu_count()
	if format == "yolo":
		labels_files = [sequence_dir.replace("sequences", "annotations") for sequence_dir in sequence_dirs]
		labels_files = [f"{labels_file}.txt" for labels_file in labels_files]
		
		Parallel(n_jobs=num_jobs)(
			delayed(_convert_labels_yolo)(image_file, labels_file)
			for (image_file, labels_file) in tqdm(zip(sequence_dirs, labels_files))
		)
	
	elif format == "coco":
		pass


def _convert_labels_yolo(sequence_dir: str, labels_file: str):
	"""Convert all labels in the `labels_file`.

	Args:
		sequence_dir (str):
			Fsequence directory that contains several images corresponding to the labels.
		labels_file (str):
			`.txt` file that contains labels.
	"""
	with open(labels_file, "r") as file_in:
		labels        = [x.replace(",", " ")   for x in file_in.read().splitlines()]
		labels        = numpy.array([x.split() for x in labels], dtype=numpy.float32)
		labels        = labels[labels[:, 0].argsort()]  # Sort labels by frame index
		frame_indexes = numpy.unique(labels[:, 0].copy().astype(int))
		
		for i, index in enumerate(frame_indexes):
			# if i % 4 != 0:
			#	continue
				
			image_file   = os.path.join(sequence_dir, f"{index:07}.jpg")
			image        = cv2.imread(image_file)  # BGR
			h, w         = image.shape[:2]         # Image HW
			frame_labels = labels[labels[:, 0] == index]
			boxes        = frame_labels[:, 2:6].copy()
			boxes        = box_xywh_to_cxcywh_norm(xywh=boxes, width=w, height=h)

			labels_yolo_file = labels_file.replace("annotations", "annotations_yolo").split(".")[0]
			labels_yolo_file = f"{labels_yolo_file}_{index:07}.txt"
			with open(labels_yolo_file, "w") as file_out:
				for l, b in zip(frame_labels, boxes):
					ss = f"{int(l[7])} {b[0]} {b[1]} {b[2]} {b[3]} {int(l[6])} {int(l[8])} {int(l[9])}\n"
					file_out.writelines(ss)
			
			image_output_file = labels_yolo_file.replace("annotations_yolo", "images")
			image_output_file = image_output_file.replace("txt", "jpg")
			copyfile(image_file, image_output_file)
			

def merge_labels(root_dir: str, format: str = "yolo", output_dir: str = "", shuffle: bool = False):
	"""Convert all labels in `train/val/testdev/testchallenge` to YOLO format.

	Args:
		root_dir (str):
			Root directory that contains `train/val/testdev/testchallenge` directories.
		format (str):
			Format to convert the labels to.
			Default: `yolo`.
		output_dir (str):
			Location to move the images and labels files to.
			Default: "", move to the `root_dir`
		shuffle (bool):
			Should shuffle files?
			Default: `False`.
	"""
	# TODO: Check for `train/val/testdev` directories
	subdirs = list_subdirs(path=root_dir)
	assert any(item in subdirs for item in ["train", "val", "testdev"])
	
	# TODO: List all images and labels files
	image_patterns = [
		os.path.join(root_dir, "train"  , "images", "*.jpg"),
		os.path.join(root_dir, "val"    , "images", "*.jpg"),
		os.path.join(root_dir, "testdev", "images", "*.jpg")
	]
	
	image_files = []
	for pattern in image_patterns:
		image_files += glob.glob(pattern)
	
	labels_files = [image_file.replace("images", f"annotations_{format}") for image_file in image_files]
	labels_files = [labels_file.replace("jpg",  "txt") for labels_file in labels_files]

	# TODO: Shuffle
	image_labels = list(zip(image_files, labels_files))
	if shuffle:
		random.shuffle(image_labels)
	
	# TODO: Move images and labels files to new location
	output_dir        = root_dir if (output_dir == "" or output_dir is None) else output_dir
	image_output_dir  = os.path.join(output_dir, "images")
	labels_output_dir = os.path.join(output_dir, f"annotations_{format}")
	create_dirs(paths=[image_output_dir, labels_output_dir], recreate=True)
	
	for i, (image_file, labels_file) in enumerate(tqdm(image_labels)):
		image_output_file  = os.path.join(image_output_dir,  f"{i:07}.jpg")
		labels_output_file = os.path.join(labels_output_dir, f"{i:07}.txt")
		copyfile(image_file,  image_output_file)
		copyfile(labels_file, labels_output_file)
		
	
# MARK: - Main

if __name__ == "__main__":
	
	# TODO: Arguments parsers
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir",   type=str,  default="",     help="Fpath to the dataset directory that contains train/val/testdev/testchallenge directories.")
	parser.add_argument("--format",     type=str,  default="yolo", help="Format to convert the labels to.")
	parser.add_argument("--merge",      type=bool, default=True,   help="Should move all images and labels to a single location.")
	parser.add_argument("--output_dir", type=str,  default="",     help="Flocation to move the images and labels files to.")
	parser.add_argument("--shuffle",    type=bool, default=True,   help="Should shuffle files after merge?")
	args = parser.parse_args()
	
	if args.data_dir == "":
		data_dir = os.path.join(datasets_dir, "visdrone", "vid2019")
	else:
		data_dir = args.data_dir
	
	convert_vid2detection_labels(root_dir=data_dir, format=args.format)
	
	if args.merge:
		merge_labels(root_dir=data_dir, format=args.format, output_dir=args.output_dir, shuffle=args.shuffle)
