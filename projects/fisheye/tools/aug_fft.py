#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Apply fisheye tomography transformation (FFT) to images and bounding boxes
in a dataset.
"""

import cv2

import mon
from fisheye import FisheyeTomographyTransform

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]
root_dir     = current_file.parents[1]
if root_dir.has_subdir("data"):
    data_dir = root_dir / "data"
else:
    data_dir = root_dir


# ----- Convert -----
def apply_fft(data: str, focal_len: int = 150):
    image_dir     = data_dir / data / "image"
    label_dir     = data_dir / data / "label"
    fft_image_dir = data_dir / data / f"image_ftt_{focal_len}"
    fft_label_dir = data_dir / data / f"label_ftt_{focal_len}"
    
    if not image_dir.exists():
        raise FileNotFoundError(f"``image_dir`` does not exist: {image_dir}.")
    if not label_dir.exists():
        raise FileNotFoundError(f"``label_dir`` does not exist: {label_dir}.")
    
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
            h, w, _ = image.shape
            cropsz  = min(h, w)

            # Read YOLO label file
            label_file = label_dir / f"{image_file.stem}.txt"
            if not label_file.is_txt_file(exist=True):
                continue
            bs = mon.hbb.load(path=label_file, fmt=mon.BBoxFormat.YOLO, imgsz=(h, w))

            # Transform
            FFT = FisheyeTomographyTransform(f=focal_len, imgsz=cropsz)
            FFT.set_ext_params([0, 0, 0, 0, 0, 0])

            transformed = FFT(image=image, bboxes=bs)
            fft_image   = transformed["image"]
            fft_bboxes  = transformed["bboxes"]

            # Postprocessing
            fft_image, fft_bboxes = mon.hbb.crop_fit_square(fft_image, fft_bboxes)
            
            # Save
            fft_image_file = fft_image_dir / f"{image_file.stem}_ftt_{f}.jpg"
            fft_image_file.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(fft_image_file), fft_image)
            
            label_fisheye_file = fft_label_dir / f"{image_file.stem}_ftt_{f}.txt"
            label_fisheye_file.parent.mkdir(parents=True, exist_ok=True)
            with open(label_fisheye_file, "w") as f:
                for b in fft_bboxes:
                    f.write(f"{int(b[4])} {b[0]:.32f} {b[1]:.32f} {b[2]:.32f} {b[3]:.32f}\n")


# ----- Main -----
if __name__ == "__main__":
    for f in [150, 300]:
        apply_fft(data="fisheye8k/extra/visdrone", focal_len=f)
