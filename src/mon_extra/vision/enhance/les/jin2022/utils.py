#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import os

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import misc

import mon

console = mon.console


def load_test_data(image_path, size=512):
    img = misc.imread(image_path, mode="RGB")
    img = misc.imresize(img, [size, size])
    img = np.expand_dims(img, axis=0)
    img = preprocessing(img)
    return img


def preprocessing(x):
    x = x / 127.5 - 1  # -1 ~ 1
    return x


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def inverse_transform(images):
    return (images+1.) / 2


def imsave(images, size, path):
    return misc.imsave(path, merge(images, size))


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img  = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h * j:h * (j + 1), w * i:w * (i + 1), :] = image
    return img


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def str2bool(x):
    return x.lower() in ("true")


def imagenet_norm(x):
    mean = [0.485, 0.456, 0.406]
    std  = [0.299, 0.224, 0.225]
    mean = torch.FloatTensor(mean).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)
    std  = torch.FloatTensor(std).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)
    return (x - mean) / std


def denorm(x):
    return x * 0.5 + 0.5


def tensor2numpy(x):
    return x.detach().cpu().numpy().transpose(1, 2, 0)


def RGB2BGR(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG", ".jpeg", ".bmp"])


def rgb_loader(img_path):
    assert (is_image_file(img_path))
    return Image.open(img_path).convert("RGB")


def gray_loader(img_path):
    assert (is_image_file(img_path))
    return Image.open(img_path).convert("L")
