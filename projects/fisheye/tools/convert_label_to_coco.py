#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convert bbox from YOLO to COCO format for training."""
"""COCO format:
{
    "info": {
        "year": "2020",
        "version": "1",
        "description": "Exported from roboflow.ai",
        "contributor": "Roboflow",
        "url": "https://app.roboflow.ai/datasets/hard-hat-sample/1",
        "date_created": "2000-01-01T00:00:00+00:00"
    },
    "licenses": [
        {
            "id": 1,
            "url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "name": "Public Domain"
        }
    ],
    "categories": [
        {
            "id": 0,
            "name": "Workers",
            "supercategory": "none"
        },
        {
            "id": 1,
            "name": "head",
            "supercategory": "Workers"
        },
        {
            "id": 2,
            "name": "helmet",
            "supercategory": "Workers"
        },
        {
            "id": 3,
            "name": "person",
            "supercategory": "Workers"
        }
    ],
    "images": [
        {
            "id": 0,
            "license": 1,
            "file_name": "0001.jpg",
            "height": 275,
            "width": 490,
            "date_captured": "2020-07-20T19:39:26+00:00"
        }
    ],
    "annotations": [
        {
            "id": 0,
            "image_id": 0,
            "category_id": 2,
            "bbox": [
                45,
                2,
                85,
                85
            ],
            "area": 7225,
            "segmentation": [],
            "iscrowd": 0
        },
        {
            "id": 1,
            "image_id": 0,
            "category_id": 2,
            "bbox": [
                324,
                29,
                72,
                81
            ],
            "area": 5832,
            "segmentation": [],
            "iscrowd": 0
        }
    ]
}
"""

import argparse
from datetime import datetime

import fjson

import mon

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]
root_dir     = current_file.parents[1]
if root_dir.has_subdir("data"):
    data_dir = root_dir / "data"
else:
    data_dir = root_dir


# ----- Convert -----
def convert_label_to_coco(data: str, split: str):
    image_dir    = data_dir / data / split / "image"
    label_dir    = data_dir / data / split / "label"
    classes_file = data_dir / data / "classes.yaml"
    json_file    = data_dir / data / split / f"{split}.json"

    if not image_dir.exists():
        raise FileNotFoundError(f"``image_dir`` does not exist: {image_dir}.")
    if not label_dir.exists():
        raise FileNotFoundError(f"``label_dir`` does not exist: {label_dir}.")
    if not classes_file.exists():
        raise FileNotFoundError(f"``classes_file`` does not exist: {classes_file}.")
    
    # Read classes from YAML file
    classes = mon.load_config(classes_file, verbose=False)
    classes = classes.get("classes", [])
    
    # COCO JSON Format
    info        = {
        "year"        : f"{datetime.now().year}",
        "version"     : "1",
        "description" : f"{data}",
        "contributor" : "Long H. Pham",
        "url"         : "",
        "date_created": f"{datetime.now()}"
    }
    licenses    = []
    categories  = classes
    images      = []
    annotations = []
    ann_id      = 0

    # Process each image
    image_files = sorted([f for f in list(image_dir.rglob("*")) if f.is_image_file()])
    with mon.create_progress_bar() as pbar:
        for i, image_file in pbar.track(
            sequence    = enumerate(image_files),
            total       = len(image_files),
            description = f"[bright_yellow]Processing"
        ):
            # Append image
            h, w, _  = mon.image.read_shape(image_file)
            image_id = i
            images.append({"id": image_id, "file_name": image_file.name, "height": h, "width": w})
            
            # Append annotations
            label_file = label_dir / f"{image_file.stem}.txt"
            if label_file is None or not label_file.is_txt_file(exist=True):
                continue
            
            # Read the YOLO label file and convert bbox format
            bs = mon.hbb.load(path=label_file, fmt=mon.BBoxFormat.YOLO2COCO, imgsz=(h, w))
            if len(bs) == 0:
                continue

            # Append annotations
            for b in bs:
                annotations.append({
                    "id"         : ann_id,
                    "image_id"   : image_id,
                    "category_id": int(b[4]),
                    "bbox"       : b[0:4].tolist(),
                    "area"       : float(b[2] * b[3]),
                    "iscrowd"    : 0,
                })
                ann_id += 1
            
    # Write to JSON file
    json_data = {
        "info"       : info,
        "licenses"   : licenses,
        "categories" : categories,
        "images"     : images,
        "annotations": annotations
    }
    with open(str(json_file), "w") as f:
        fjson.dump(json_data, f, float_format=".32f", indent=None)


# ----- Main -----
if __name__ == "__main__":
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val", required=True)
    args = parser.parse_args()
    
    convert_label_to_coco("fisheye8k", args.split)
    '''
    
    convert_label_to_coco("fisheye8k", "train")
