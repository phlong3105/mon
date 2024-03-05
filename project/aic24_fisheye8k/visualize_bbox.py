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
@click.option("--image-dir",  type=click.Path(exists=True),  default=mon.DATA_DIR/"aic/aic24-fisheye8k/train", help="Image directory.")
@click.option("--label-dir",  type=click.Path(exists=True),  default=mon.DATA_DIR/"aic/aic24-fisheye8k/train", help="Bounding bbox directory.")
@click.option("--output-dir", type=click.Path(exists=False), default=None,                                     help="Output directory.")
@click.option("--format",     type=click.Choice(_bbox_formats, case_sensitive=False), default="voc",           help="Bounding bbox format.")
@click.option("--ext",        type=click.Choice(_extensions,   case_sensitive=False), default="jpg",           help="Image extension.")
@click.option("--thickness",  type=int,                      default=1,                                        help="The thickness of the bounding box border line in px.")
@click.option("--fill",       is_flag=True,                                                                    help="Fill the region inside the bounding box with transparent color.")
@click.option("--show-label", is_flag=True)
@click.option("--save",       is_flag=True, default=True)
@click.option("--verbose",    is_flag=True)
def visualize_bbox(
	image_dir : mon.Path,
	label_dir : mon.Path,
	output_dir: mon.Path,
	format    : str,
	ext       : str,
	thickness : int,
	fill      : bool,
	show_label: bool,
	save      : bool,
	verbose   : bool,
):
	assert image_dir is not None and mon.Path(image_dir).is_dir()
	assert label_dir is not None and mon.Path(label_dir).is_dir()
	
	image_dir  = mon.Path(image_dir)
	label_dir  = mon.Path(label_dir)
	data_name  = image_dir.name
	output_dir = output_dir or label_dir.parent / "visualize"
	output_dir = mon.Path(output_dir)
	# if save:
		# output_dir.mkdir(parents=True, exist_ok=True)
	
	code = mon.ShapeCode.from_value(value=f"{format}_to_voc")
	
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
				l = [x for x in l if len(x) >= 5]
				b = np.array([list(map(float, x[1:])) for x in l])
				b = mon.convert_bbox(bbox=b, code=code, height=h, width=w)
				
				colors = mon.RGB.values()
				n      = len(colors)
				for j, x in enumerate(b):
					image = mon.draw_bbox(
						image     = image,
						bbox      = x,
						label     = l[j] if show_label else None,
						color     = colors[abs(hash(l[j][0])) % n],
						thickness = thickness,
						fill      = fill,
					)
			
			image = cv2.putText(
				img       = image,
				text      = f"{image_files[i].stem}",
				org       = [50, 50],
				fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
				fontScale = 1,
				color     = [255, 255, 255],
				thickness = 3,
				lineType  = cv2.LINE_AA,
			)
			if save:
				output_file = str(image_files[i]).replace("images", "visualize")
				output_file = mon.Path(output_file)
				output_file = output_file.parent / f"{image_files[i].stem}.jpg"
				output_file.parent.mkdir(parents=True, exist_ok=True)
				# output_file = output_dir / f"{image_files[i].stem}.{ext}"
				cv2.imwrite(str(output_file), image)
			if verbose:
				cv2.imshow("Image", image)
				cv2.waitKey(0)

# endregion


# region Main

if __name__ == "__main__":
	visualize_bbox()

# endregion
