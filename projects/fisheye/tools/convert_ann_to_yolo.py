#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convert prediction results from COCO to YOLO format."""

import argparse
import json

import mon

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]
root_dir     = current_file.parents[1]
if root_dir.has_subdir("data"):
    data_dir = root_dir / "data"
else:
    data_dir = root_dir


# ----- Utils -----
def get_image_id(filename: str) -> int:
    filename     = mon.Path(filename).stem
    scene_list   = ["M", "A", "E", "N"]
    camera_index = int(filename.split("_")[0].split("camera")[1])
    scene_index  = scene_list.index(filename.split("_")[1])
    frame_index  = int(filename.split("_")[2])
    image_id     = int(str(camera_index) + str(scene_index) + str(frame_index))
    return int(image_id)


# ----- Convert -----
def convert_ann_to_yolo(data: str, split: str, label: str):
    image_dir = data_dir / data / split / "image"
    json_file = data_dir / data / split / f"{label}.json"
    label_dir = data_dir / data / split / f"{label}"

    if not image_dir.exists():
        raise FileNotFoundError(f"``image_dir`` does not exist: {image_dir}.")
    if not json_file.exists():
        raise FileNotFoundError(f"``json_file`` does not exist: {json_file}.")

    # Read all labels in JSON file
    annotations = []
    with open(str(json_file), "r") as f:
        annotations = json.load(f)

    # Map image_id with image_file
    image_files = sorted([f for f in list(image_dir.rglob("*")) if f.is_image_file()])
    with mon.create_progress_bar() as pbar:
        for i, image_file in pbar.track(
            sequence    = enumerate(image_files),
            total       = len(image_files),
            description = f"[bright_yellow]Processing"
        ):
            h, w, _    = mon.image.read_shape(image_file)
            image_id   = get_image_id(image_file)
            anns       = [a for a in annotations if a["image_id"] == image_id]

            label_file = label_dir / f"{image_file.stem}.txt"
            label_file.parent.mkdir(parents=True, exist_ok=True)
            with open(label_file, "w") as f:
                for ann in anns:
                    # Convert bbox to YOLO format
                    c  = ann["category_id"]
                    b  = ann["bbox"]
                    s  = ann["score"]
                    b0 = (b[0] + b[2] / 2) / w
                    b1 = (b[1] + b[3] / 2) / h
                    b2 = b[2] / w
                    b3 = b[3] / h
                    # Write to label file in YOLO format
                    f.write(f"{c} {b0:.32f} {b1:.32f} {b2:.32f} {b3:.32f} {s:.32f}\n")


# ----- Main -----
if __name__ == "__main__":
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val", required=True)
    parser.add_argument("--label", type=str, default="038", required=True)
    args = parser.parse_args()
    
    convert_ann_to_yolo("fisheye8k", args.split, args.label)
    '''
    
    convert_ann_to_yolo("fisheye8k", "test", "038")
