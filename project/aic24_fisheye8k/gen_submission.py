#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json

import click
import cv2
import numpy as np

import mon

_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]
_bbox_formats = ["voc", "coco", "yolo"]


# region Function

def get_image_id(image_name: str):
	image_name   = mon.Path(image_name).stem
	# image_name   = image_name.split(".png")[0]
	scene_list   = ["M", "A", "E", "N"]
	camera_index = int(image_name.split("_")[0].split("camera")[1])
	scene_index  = scene_list.index(image_name.split("_")[1])
	frame_index  = int(image_name.split("_")[2])
	image_id     = int(str(camera_index) + str(scene_index) + str(frame_index))
	return image_id


@click.command(name="main", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--predict-dir", type=click.Path(exists=True),  default=_current_dir / "run/predict",    help="Prediction results directory.")
@click.option("--output-file", type=click.Path(exists=False), default=None,                            help="Output .json file.")
@click.option("--format",      type=click.Choice(_bbox_formats, case_sensitive=False), default="yolo", help="Bounding bbox format.")
@click.option("--verbose",     is_flag=True)
def gen_submission(
	predict_dir: str | mon.Path,
	output_file: str | mon.Path,
	format     : str,
	verbose    : bool,
):
	assert predict_dir is not None and mon.Path(predict_dir).is_dir()
	
	predict_dir = mon.Path(predict_dir)
	data_name   = predict_dir.name
	output_file = output_file or predict_dir / "results"
	output_file = mon.Path(output_file)
	output_file = output_file if mon.Path(output_file).is_json_file() else mon.Path(str(output_file) + ".json")
	output_file.parent.mkdir(parents=True, exist_ok=True)
	
	code = mon.ShapeCode.from_value(value=f"{format}_to_coco")
	
	image_files = list(predict_dir.rglob("*"))
	image_files = [f for f in image_files if f.is_image_file()]
	image_files = sorted(image_files)
	detections: list[dict] = []
	with mon.get_progress_bar() as pbar:
		for i in pbar.track(
			sequence    = range(len(image_files)),
			total       = len(image_files),
			description = f"[bright_yellow] Processing {data_name}"
		):
			image_file = image_files[i]
			image      = cv2.imread(str(image_file))
			h, w, c    = image.shape

			label_file = None
			for j in range(0, 4):
				file = image_file.parents[j] / "labels" / f"{image_file.stem}.txt"
				if file.is_txt_file():
					label_file = file
					break
			if label_file is None or not label_file.is_txt_file():
				continue

			if label_file.is_txt_file():
				with open(label_file, "r") as in_file:
					l = in_file.read().splitlines()
				l = [x.strip().split(" ") for x in l]
				l = [x for x in l if len(x) >= 5]
				b = np.array([list(map(float, x[1:5])) for x in l])
				b = mon.convert_bbox(bbox=b, code=code, height=h, width=w)
				assert len(b) == len(l)
				
				image_id = get_image_id(image_file.name)
				for _l, _b in zip(l, b):
					detection = {
						"image_id"   : int(image_id),
						"category_id": int(_l[0]),
						"bbox"       : [int(_b[0]), int(_b[1]), int(_b[2]), int(_b[3])],
						"score"      : float(_l[5]),
					}
					detections.append(detection)
	
	with open(str(output_file), "w") as json_file:
		json.dump(detections, json_file, indent=None)
		
# endregion


# region Main

if __name__ == "__main__":
	gen_submission()

# endregion
