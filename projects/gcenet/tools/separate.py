#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

import mon
from mon import albumentations as A

# mon.dev()

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]
data_dir     = current_file.parents[1]
if data_dir.has_subdir("data"):
    data_dir = data_dir / "data"


# ----- Inputs -----
filename   = "Test20"
# filename   = "image00266"
image_file = data_dir / f"{filename}_image.jpg"
depth_file = data_dir / f"{filename}_depth.jpg"


# ----- Load data -----
image = mon.image.load(image_file, cv2.IMREAD_COLOR)
depth = mon.image.load(depth_file, cv2.IMREAD_GRAYSCALE)

transform = A.Compose([
    A.Normalize(normalization="min_max"),
    A.ToTensorV2(transpose_mask=True),
], additional_targets={"depth": "image"})

augmented    = transform(image=image, depth=depth)
image_tensor = augmented["image"].unsqueeze(0)
depth_tensor = augmented["depth"].unsqueeze(0)
mon.log(f"image: {image_tensor.shape}\n"
        f"depth: {depth_tensor.shape}")


# ----- Process -----
bam_tensor  = mon.image.brightness_attention_map(image_tensor, 1.5)
# bam_tensor  = (1.0 - bam_tensor)
dark_tensor = image_tensor * bam_tensor


# ----- Visualize -----
def show_images(outputs: dict):
    for k, v in outputs.items():
        cv2.imshow(k, v)
    cv2.waitKey(0)


def save_images(saves: dict):
    for k, v in saves.items():
        cv2.imwrite(str(data_dir / f"{filename}_{k}.jpg"), v)


image   = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
bam     = (bam_tensor.squeeze().numpy() * 255).astype(np.uint8)
dark    = cv2.cvtColor((dark_tensor.squeeze().numpy().transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
outputs = {
    "image": image,
    "depth": depth,
    "bam"  : bam,
    "dark" : dark,
}
saves   = {
    "bam"  : bam,
    "dark" : dark,
}
show_images(outputs)
save_images(saves)
