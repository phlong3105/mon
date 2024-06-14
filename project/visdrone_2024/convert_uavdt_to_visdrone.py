#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script visualizes bounding boxes on images."""

from __future__ import annotations

import click
import cv2
import numpy as np

import mon

_bbox_formats = ["voc", "coco", "yolo"]
_extensions   = ["jpg", "png"]


# region Function

@click.command(name="main", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--image-dir",  type=click.Path(exists=True),  default=mon.DATA_DIR/"visdrone/visdrone_2024_det/uavdt/train/images",      help="Image directory.")
@click.option("--label-dir",  type=click.Path(exists=True),  default=mon.DATA_DIR/"visdrone/visdrone_2024_det/uavdt/train/labels_ignore", help="Bounding bbox directory.")
@click.option("--output-dir", type=click.Path(exists=False), default=None, help="Output directory.")
@click.option("--verbose",    is_flag=True)
def convert_bbox(
	image_dir : mon.Path,
	label_dir : mon.Path,
	output_dir: mon.Path,
	verbose   : bool,
):
	assert image_dir is not None and mon.Path(image_dir).is_dir()
	assert label_dir is not None and mon.Path(label_dir).is_dir()
	
	image_dir  = mon.Path(image_dir)
	label_dir  = mon.Path(label_dir)
	data_name  = image_dir.name
	output_dir = output_dir or label_dir.parent / "labels"
	output_dir = mon.Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	
	# code = mon.ShapeCode.from_value(value=f"xywh_to_cxcyn")
	
	image_files = list(image_dir.rglob("*"))
	image_files = [f for f in image_files if f.is_image_file()]
	image_files = sorted(image_files)
	with mon.get_progress_bar() as pbar:
		for i in pbar.track(
			sequence    = range(len(image_files)),
			total       = len(image_files),
			description = f"[bright_yellow] Processing {data_name}"
		):
			image   = cv2.imread(str(image_files[i]))
			h, w, c = image.shape
			
			label_file = label_dir / f"{image_files[i].stem}.txt"
			if label_file.is_txt_file():
				with open(label_file, "r") as in_file:
					l = in_file.read().splitlines()
				l = [x.strip().split(" ") for x in l]
				b = np.array([list(map(float, x[1:5])) for x in l])  # bbox
				# b = mon.convert_bbox(bbox=b, code=code, height=h, width=w)
			
			output_file = output_dir / f"{image_files[i].stem}.txt"
			with open(output_file, "w") as out_file:
				for j, x in enumerate(l):
					c = int(x[0])  # class
					if c == 6:
						c = 5
					elif c == 4:
						c = 3
					elif c == 9:
						c = 8
					out_file.write(f"{c} {b[j][0]:.6f} {b[j][1]:.6f} {b[j][2]:.6f} {b[j][3]:.6f}\n")
					
# endregion


# region Main

if __name__ == "__main__":
	convert_bbox()

# endregion
