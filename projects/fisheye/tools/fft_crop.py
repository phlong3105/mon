#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Apply fisheye tomography transformation (FFT) to images and bounding boxes
in a dataset.
"""

import cv2
import numpy as np

import mon

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]
root_dir     = current_file.parents[1]
if root_dir.has_subdir("data"):
    data_dir = root_dir / "data"
else:
    data_dir = root_dir


# ----- Utils -----
def pad_image_to_square(image: np.ndarray) -> np.ndarray:
    # Get image dimensions
    h, w     = image.shape[:2]
    max_side = max(h, w)
    # Calculate padding
    top      = (max_side - h) // 2
    bottom   = max_side - h - top
    left     = (max_side - w) // 2
    right    = max_side - w - left
    # Add borders using copyMakeBorder
    padded_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded_img


def transform_bbox(bbox: np.ndarray, image: np.ndarray, focal_len: int = 150, imgsz: int = 1280) -> np.ndarray:
    h, w = mon.image_size(image)
    bbox = mon.convert_hbb(bbox, fmt=mon.BBoxFormat.CXCYWHN2XYXY, imgsz=(h, w))

    transform = mon.FisheyeTransform(focal_len, imgsz, p=1)

    new_bbox = []
    for b in bbox:
        x1, y1, x2, y2, c = int(b[0]), int(b[1]), int(b[2]), int(b[3]), int(b[4])
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[y1:y2, x1:x2, :] = [255, 255, 255]  # White in BGR format # image[y1:y2, x1:x2]

        # Apply fisheye transformation
        transform.set_ext_params([0, 0, 0, 0, 0, 0])
        transformed = transform(image=canvas)
        t_canvas    = transformed["image"]

        # Find contours
        ret, thresh = cv2.threshold(t_canvas, 200, 255, cv2.THRESH_BINARY)
        # cv2.imshow("thresh", thresh)
        # cv2.waitKey(10)
        thresh      = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        if thresh.dtype != np.uint8:
            thresh = thresh.astype(np.uint8)
        contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the bounding box of the contours
        min_x = float("inf")
        min_y = float("inf")
        max_x = float("-inf")
        max_y = float("-inf")
        for contour in contours:
            x_, y_, w_, h_ = cv2.boundingRect(contour)
            min_x = min(min_x, x_)
            min_y = min(min_y, y_)
            max_x = max(max_x, x_ + w_)
            max_y = max(max_y, y_ + h_)

        # If no contours found, skip this bbox
        if any(x == float("inf") or x == float("-inf") for x in [min_x, min_y, max_x, max_y]):
            continue
        new_bbox.append([min_x, min_y, max_x, max_y, c])

    # Convert back to YOLO format
    new_bbox = np.array(new_bbox, dtype=np.float32)
    new_bbox = mon.convert_hbb(new_bbox, fmt=mon.BBoxFormat.XYXY2CXCYWHN, imgsz=(imgsz, imgsz))
    return new_bbox


# ----- Convert -----
def convert_fisheye(
    data      : str,
    split     : str,
    extra_data: str,
    focal_len : int = 150,
    imgsz     : int = 1280
):
    image_dir         = data_dir / data / split / extra_data / "image"
    label_dir         = data_dir / data / split / extra_data / "label"
    image_fisheye_dir = data_dir / data / split / extra_data / f"image_fisheye_f{focal_len}"
    label_fisheye_dir = data_dir / data / split / extra_data / f"label_fisheye_f{focal_len}"

    if not image_dir.exists():
        raise FileNotFoundError(f"[image_dir] does not exist: {image_dir}.")
    if not label_dir.exists():
        raise FileNotFoundError(f"[label_dir] does not exist: {label_dir}.")

    transform = mon.FisheyeTransform(focal_len, imgsz, p=1)

    # Process each image
    image_files = sorted([f for f in list(image_dir.rglob("*")) if f.is_image_file()])
    with mon.create_progress_bar() as pbar:
        for i, image_file in pbar.track(
            sequence    = enumerate(image_files),
            total       = len(image_files),
            description = f"[bright_yellow]Processing"
        ):
            # Read image
            image   = cv2.imread(str(image_file))
            # image   = pad_image_to_square(image)
            # image = mon.resize(image, focal_len, side="long", interpolation="bicubic")
            h, w, _ = image.shape

            # Read YOLO label file
            label_file = label_dir / f"{image_file.stem}.txt"
            if not label_file.is_txt_file(exist=True):
                continue

            bs = mon.load_hbb(path=label_file, fmt=mon.BBoxFormat.YOLO, imgsz=(h, w))

            # Pad image to square
            crop_sz   = min(h, w)
            image, bs = mon.center_crop_image_and_hbbs(image, bs, (crop_sz, crop_sz))

            # Apply fisheye transformation
            transform.set_ext_params([0, 0, 0, 0, 0, 0])
            transformed   = transform(image=image, bboxes=bs)
            transformed_i = transformed["image"]
            transformed_b = transformed["bboxes"]

            image_fisheye_file = image_fisheye_dir / f"{image_file.stem}_fisheye.jpg"
            image_fisheye_file.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_fisheye_file), transformed_i)

            label_fisheye_file = label_fisheye_dir / f"{image_file.stem}_fisheye.txt"
            label_fisheye_file.parent.mkdir(parents=True, exist_ok=True)
            with open(label_fisheye_file, "w") as f:
                for b in transformed_b:
                    f.write(f"{int(b[4])} {b[0]:.32f} {b[1]:.32f} {b[2]:.32f} {b[3]:.32f}\n")


# ----- Main -----
if __name__ == "__main__":
    # convert_fisheye("fisheye8k", "extra", "visdrone", focal_len=150, imgsz=1280)
    # convert_fisheye("fisheye8k", "extra", "pet", focal_len=150, imgsz=1280)
    convert_fisheye("fisheye8k", "extra", "taiwancctv", focal_len=300, imgsz=1280)
