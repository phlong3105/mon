#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Visualization scripts."""

import argparse
import cv2

import mon

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]
root_dir     = current_file.parents[1]
if root_dir.has_subdir("data"):
    data_dir = root_dir / "data"
else:
    data_dir = root_dir


# ----- Visualize -----
def visualize_bbox(data: str, split: str, label: str):
    image_dir    = data_dir / data / split / "image"
    label_dir    = data_dir / data / split / f"{label}"
    classes_file = data_dir / data / "classes.yaml"
    viz_dir 	 = data_dir / data / split / f"{label}_viz"

    if not image_dir.exists():
        raise FileNotFoundError(f"``image_dir`` does not exist: {image_dir}.")
    if not label_dir.exists():
        raise FileNotFoundError(f"``label_dir`` does not exist: {label_dir}.")

    # Read classes from YAML file
    classes = mon.load_config(classes_file, verbose=False)
    classes = classes.get("classes", [])

    # Process each image
    image_files = sorted([f for f in list(image_dir.rglob("*")) if f.is_image_file()])
    with mon.create_progress_bar() as pbar:
        for image_file in pbar.track(
            sequence    = image_files,
            total       = len(image_files),
            description = f"[bright_yellow]Processing"
        ):
            # Read image
            image   = cv2.imread(str(image_file))
            image   = image[:, :, ::-1]  # Convert BGR to RGB
            h, w, _ = image.shape

            # Read YOLO label file
            label_file = label_dir / f"{image_file.stem}.txt"
            if not label_file.is_txt_file(exist=True):
                continue
            bs = mon.bbox.load(path=label_file, fmt=mon.BBoxFormat.YOLO2VOC, imgsz=(h, w))
            
            # Draw bounding boxes on the image
            for j, b in enumerate(bs):
                if len(b) >= 6:
                    l = f"{j} {int(b[5])}: {b[6]:.4f}"
                else:
                    l = f"{j} {int(b[5])}"
                l = ""
                image = mon.dtypes.draw_bbox(
                    image     = image,
                    bbox      = b,
                    label     = l,
                    color     = classes[int(b[5])]["color"],
                    thickness = 2,
                    fill      = False,
                )
            image = cv2.putText(
                img       = image,
                text      = f"{image_file.stem}",
                org       = [50, 50],
                fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 1,
                color     = [255, 255, 255],
                thickness = 3,
                lineType  = cv2.LINE_AA,
            )
            
            # Save
            image	    = image[:, :, ::-1]  # Convert RGB back to BGR for saving
            output_file = viz_dir / f"{image_file.stem}.jpg"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_file), image)


# ----- Main -----
if __name__ == "__main__":
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val", required=True)
    parser.add_argument("--label", type=str, default="038", required=True)
    args = parser.parse_args()
    
    visualize_bbox("fisheye8k", args.split, args.label)
    '''
    
    visualize_bbox("fisheye8k", split="test", label="label")
