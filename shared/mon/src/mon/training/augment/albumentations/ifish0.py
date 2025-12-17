#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements fisheye transformation for an image and horizontal bounding boxes.

References:
    - Code: https://github.com/Gil-Mor/iFish
"""

import cv2
import numpy as np

from mon.core import BBoxFormat, bbox as B


# ----- Utils -----
def fish_xn_yn(
    source_x  : np.ndarray,
    source_y  : np.ndarray,
    radius    : float,
    distortion: float
) -> tuple[float, float]:
    """Converts a pixel's coordinates in the original image to its corresponding
    coordinates in fisheye image.
    
    Args:
        source_x (np.ndarray): Image pixel coordinates.
        source_y (np.ndarray): Image pixel coordinates.
        radius (float): Pixel's distance from the image center.
        distortion (float): Distortion coefficient.

    Returns:
        float: Pixel's new x-coordinate.
        float: Pixel's new y-coordinate.
    """
    if 1 - distortion * (radius ** 2) == 0:
        fish_x = source_x
        fish_y = source_y
    else:
        fish_x = source_x / (1 - distortion * (radius ** 2))
        fish_y = source_y / (1 - distortion * (radius ** 2))
    return fish_x, fish_y


def reverse_fish_xn_yn(
    source_x  : np.ndarray,
    source_y  : np.ndarray,
    radius    : float,
    distortion: float
) -> tuple[float, float]:
    """Converts a pixel's coordinates in fisheye image back to its corresponding
    coordinates in the original image.
    
    Args:
        source_x (np.ndarray): Image pixel coordinates.
        source_y (np.ndarray): Image pixel coordinates.
        radius (float): Pixel's distance from the image center.
        distortion (float): Distortion coefficient.

    Returns:
        float: Pixel's new x-coordinate.
        float: Pixel's new y-coordinate.
    """
    if radius == 0:
        return source_x, source_y
    coefficient = (np.sqrt(1 + 4 * distortion * (radius ** 2)) - 1) / (2 * distortion * (radius ** 2))
    return source_x * coefficient, source_y * coefficient


def fish(image: np.ndarray, distortion: float) -> np.ndarray:
    """Converts an ordinary image to fisheye image.
    
    Args:
        image (numpy.ndarray): The original image.
        distortion (float): Distortion coefficient.
        
    Returns:
        numpy.ndarray: Newly generated fisheye image.
    """
    w, h, c = image.shape

    # RGB to RGBA
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = np.dstack((image, np.full((w, h), 255)))
    
    # Prepare array for dst image
    fisheye = np.zeros_like(image)

    # Floats and calculations
    w, h = float(w), float(h)
    
    # Easier calculation if we traverse x, y in dst image
    for x in range(len(fisheye)):
        for y in range(len(fisheye[x])):
            # normalize x and y to be in interval of [-1, 1]
            xnd, ynd = float((2 * x - w) / w), float((2 * y - h) / h)
            # get xn and yn distance from normalized center
            rd = np.sqrt(xnd ** 2 + ynd ** 2)
            # new normalized pixel coordinates
            xdu, ydu = fish_xn_yn(xnd, ynd, rd, distortion)
            # convert the normalized distorted xdn and ydn back to image pixels
            xu, yu = int(((xdu + 1) * w) / 2), int(((ydu + 1) * h) / 2)
            # if new pixel is in bounds copy from source pixel to destination pixel
            if (0 <= xu) and (xu < image.shape[0]) and (0 <= yu) and (yu < image.shape[1]):
                fisheye[x][y] = image[xu][yu]
    
    return fisheye.astype(np.uint8)


def pad_square(image: np.ndarray, pad_value: int = 0) -> np.ndarray:
    """Adds padding to make the image square.
    
    Args:
        image (numpy.ndarray): The input image.
        pad_value (int): The padding value. Defaults to 0.
        
    Returns:
        numpy.ndarray: The padded square image.
    """
    h, w, c = image.shape
    if w >= h:
        border_w = (w - h) // 2
        image    = cv2.copyMakeBorder(image, border_w, border_w, 0, 0, cv2.BORDER_CONSTANT, value=pad_value)
    else:
        border_w = (h - w) // 2
        image    = cv2.copyMakeBorder(image, 0, 0, border_w, border_w, cv2.BORDER_CONSTANT, value=pad_value)
    return image


# ----- Transformation -----
def convert_image(image: np.ndarray, distortion: float, crop: bool = True) -> np.ndarray:
    """Converts an ordinary image to fisheye image.
    
    Args:
        image (numpy.ndarray): The original image.
        distortion (float): Distortion coefficient.
        crop (bool): Whether to crop the dark area around images. Defaults to True.
        
    Returns:
        numpy.ndarray: Newly generated fisheye image.
    """
    new_img = fish(image, distortion)
    if not crop:
        return new_img

    h, w, c = image.shape

    # Calculate the coordinates of the furthest point to the left and up
    left = (0.0, float(h / 2))
    top  = (float(w / 2), 0.0)

     # Normalize the coordinates
    left = ((2 * left[0] - w) / w, (2 * left[1] - h) / h)
    top  = ((2 * top[0]  - w) / w, (2 * top[1]  - h) / h)

    # Calculate the new coordinates
    new_left_x, new_left_y = reverse_fish_xn_yn(left[0], left[1], np.sqrt(left[0] ** 2 + left[1] ** 2), distortion)
    new_top_x, new_top_y   = reverse_fish_xn_yn(top[0],  top[1],  np.sqrt(top[0]  ** 2 + top[1]  ** 2), distortion)
    
    # Un-normalize the new coordinates
    left = (int((new_left_x + 1) * w / 2), int((new_left_y + 1) * h / 2))
    top  = (int((new_top_x  + 1) * w / 2), int((new_top_y  + 1) * h / 2))
    
    new_img = new_img[top[1]:(h - top[1]), left[0]:(w - left[0]), :]
    return new_img


def convert_bboxes(
    bboxes    : np.ndarray,
    old_size  : tuple,
    new_size  : tuple,
    distortion: float,
    crop      : bool = True
) -> np.ndarray:
    """Converts bounding boxes from an ordinary image to fisheye image.
    
    Args:
        bboxes (np.ndarray): The bounding boxes.
        old_size (tuple[int, int]): Original image size as (W, H).
        new_size (tuple[int, int]): New image size as (W, H).
        distortion (float): Distortion coefficient.
        crop (bool): Whether to crop the dark area around images. Defaults to True.

    Returns:
        numpy.ndarray: Newly generated bounding boxes.
    """
    old_w, old_h = old_size
    new_w, new_h = new_size
    left_margin  = int((old_w - new_w) // 2)
    top_margin   = int((old_h - new_h) // 2)
    
    bboxes     = B.convert(bboxes, fmt=BBoxFormat.YOLO2VOC, imgsz=old_size)
    new_bboxes = []
    for bbox in bboxes:
        # top_left, top_right, bottom_left, bottom_right
        bbox_x = np.array([bbox[0], bbox[2], bbox[0], bbox[2]]).astype(float)
        bbox_y = np.array([bbox[1], bbox[1], bbox[3], bbox[3]]).astype(float)
        
        rd = np.zeros_like(bbox_x)
        bbox_x_fish = np.zeros_like(bbox_x)
        bbox_y_fish = np.zeros_like(bbox_y)

        # Calculate the new coordinates individually
        for i in range(4):
            bbox_x[i] = (2 * bbox_x[i] - old_w) / old_w
            bbox_y[i] = (2 * bbox_y[i] - old_h) / old_h
            rd[i]     = np.sqrt(bbox_x[i] ** 2 + bbox_y[i] ** 2)
            bbox_x_fish[i], bbox_y_fish[i] = reverse_fish_xn_yn(bbox_x[i], bbox_y[i], rd[i], distortion)
            bbox_x_fish[i] = int(((bbox_x_fish[i] + 1) * old_w) / 2)
            bbox_y_fish[i] = int(((bbox_y_fish[i] + 1) * old_h) / 2)
        if crop:
            left_fish  = int(min(bbox_x_fish)) - left_margin
            top_fish   = int(min(bbox_y_fish)) - top_margin
            right_fish = int(max(bbox_x_fish)) - left_margin
            bot_fish   = int(max(bbox_y_fish)) - top_margin
            new_bboxes.append([left_fish, top_fish, right_fish, bot_fish, bbox[4]])
        else:
            left_fish  = int(min(bbox_x_fish))
            top_fish   = int(min(bbox_y_fish))
            right_fish = int(max(bbox_x_fish))
            bot_fish   = int(max(bbox_y_fish))
            new_bboxes.append([left_fish, top_fish, right_fish, bot_fish, bbox[4]])
    
    new_bboxes = np.array(new_bboxes)
    new_bboxes = B.convert(new_bboxes, fmt=BBoxFormat.VOC2YOLO, imgsz=new_size)
    return new_bboxes
