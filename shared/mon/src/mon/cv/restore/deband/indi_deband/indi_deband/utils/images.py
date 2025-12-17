#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import torch


def read_image(img_name: str, to_tensor: bool = True):
    image = cv2.imread(img_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255
    image = image.astype("float32")
    # move channel dimension to 0
    if to_tensor:
        image = torch.from_numpy(image).permute(2, 0, 1)
    return image


def cv_to_tensor(img: np.ndarray) -> torch.Tensor:
    img    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img    = img / 255
    img    = img.astype("float32")
    tensor = torch.from_numpy(img).permute(2, 0, 1)
    return tensor


def to_opencv(image: torch.Tensor) -> np.ndarray:
    cv = np.uint8(image.permute(1, 2, 0).cpu().numpy() * 255)
    cv = cv2.cvtColor(cv, cv2.COLOR_RGB2BGR)
    return cv


def display(image: np.ndarray, text: str = "image"):
    cv2.imshow(text, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# expects a single image with batch size
def write_images(images_map: dict, epoch: int, writer):
    for field, image in images_map.items():
        image = image.squeeze(0)
        writer.add_image("Image " + field, image.squeeze(0), epoch)


# returns real and imag components stacked across the batch
def fft_2d(x, dims=(-2,-1)):
    fft    = torch.fft.fftn(x, dim=dims)
    real   = fft.real
    imag   = fft.imag
    output = torch.cat([real, imag],dim = 1)
    return output
