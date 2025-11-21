#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements fisheye transformation for an image and horizontal bounding boxes.

References:
    - Code: https://github.com/Gil-Mor/iFish
"""

__all__ = [
    "iFishTransform",
]

from typing import Any

import numpy as np
from albumentations.core.transforms_interface import (
    BaseTransformInitSchema,
    DualTransform,
)
from albumentations.core.type_definitions import Targets
from pydantic import Field

from mon.core import ALBUMENTATIONS, image as I


# ----- Utils -----
def get_fisheye_factor(r: float, d: float) -> float:
    """Calculate the fisheye transformation factor.

    Args:
        r: Radius in normalized coordinates.
        d: Distortion factor.
    """
    factor = 1 - d * (r ** 2)
    return factor


def get_reverse_fisheye_factor(r: float, d: float) -> float:
    """Calculate the reverse fisheye transformation factor.

    Args:
        r: Radius in normalized coordinates.
        d: Distortion factor.
    """
    factor = (np.sqrt(1 + 4 * d * (r ** 2)) - 1) / (2 * d * (r ** 2))
    return factor


def fisheye_xy_n(x_n: float, y_n: float, r: float, d: float) -> tuple[float, float]:
    """Calculate the fisheye transformation for normalized coordinates.

    Args:
        x_n: Normalized x-coordinate.
        y_n: Normalized y-coordinate.
        r: Radius in normalized coordinates.
        d: Distortion factor.

    Returns:
        tuple: Transformed x and y coordinates.
    """
    factor = get_fisheye_factor(r, d)
    if factor == 0:
        return x_n, y_n
    else:
        return x_n / factor, y_n / factor


def reverse_fisheye_xy_n(x_n: float, y_n: float, r: float, d: float) -> tuple[float, float]:
    """Calculate the reverse fisheye transformation for normalized coordinates.

    Args:
        x_n: Normalized x-coordinate.
        y_n: Normalized y-coordinate.
        r: Radius in normalized coordinates.
        d: Distortion factor.

    Returns:
        tuple: Transformed x and y coordinates.
    """
    if isinstance(r, float) and r == 0:
        return x_n, y_n
    elif isinstance(r, np.ndarray) and np.all(r == 1):
        return x_n, y_n
    factor = get_reverse_fisheye_factor(r, d)
    return x_n * factor, y_n * factor


# ----- Transformation Functions -----
def transform_image(image: np.ndarray, distortion: float) -> np.ndarray:
    """Apply fisheye transformation to an image.

    Args:
        image: Input image as ``numpy.ndarray`` in [H, W, C] format.
        distortion: Distortion factor.

    Returns:
        np.ndarray: Transformed image.
    """
    def transform(img: np.ndarray) -> np.ndarray:
        w, h = I.imgsz(img)  # Note: we must reverse the order of w and h
        if len(img.shape) == 2:
            img = np.dstack((img, img, img))
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = np.dstack((img, np.full((w, h), 255)))
        # Floats for calculations
        w, h = float(w), float(h)
        distort_img = np.zeros_like(img)
        for x in range(len(distort_img)):
            for y in range(len(distort_img[x])):
                # Normalize x and y to be in interval of [-1, 1]
                x_n = float((2 * x - w) / w)
                y_n = float((2 * y - h) / h)
                # Get xn and yn distance from normalized center
                r_n = np.sqrt(x_n ** 2 + y_n ** 2)
                # New normalized pixel coordinates
                x_n_u, y_n_u = fisheye_xy_n(x_n, y_n, r_n, distortion)
                # Convert the normalized distorted xdn and ydn back to image pixels
                x_u = int(((x_n_u + 1) * w) / 2)
                y_u = int(((y_n_u + 1) * h) / 2)
                # If new pixel is in bounds copy from source pixel to destination pixel
                if 0 <= x_u < img.shape[0] and 0 <= y_u < img.shape[1]:
                    distort_img[x][y] = img[x_u][y_u]
        return distort_img.astype(np.uint8)

    def crop(img: np.ndarray) -> np.ndarray:
        h, w = I.imgsz(img)
        # Calculate the coordinates of the furthest point to the left and up
        left = (0.0, float(h / 2))
        top  = (float(w / 2), 0.0)
        # Normalize the coordinates
        left = ((2 * left[0] - w) / w, (2 * left[1] - h) / h)
        top  = ((2 *  top[0] - w) / w, (2 *  top[1] - h) / h)
        # Calculate the new coordinates
        left_x_new, left_y_new = reverse_fisheye_xy_n(left[0], left[1], np.sqrt(left[0] ** 2 + left[1] ** 2), distortion)
        top_x_new,  top_y_new  = reverse_fisheye_xy_n(top[0],  top[1],   np.sqrt(top[0] ** 2 +  top[1] ** 2), distortion)
        # un-normalize the new coordinates
        left = (int((left_x_new + 1) * w / 2), int((left_y_new + 1) * h / 2))
        top  = (int((top_x_new  + 1) * w / 2), int((top_y_new  + 1) * h / 2))
        return img[top[1]:(h - top[1]), left[0]:(w - left[0]), :]

    return crop(transform(image))


def transform_bbox0(
    bbox        : np.ndarray,
    old_size    : tuple[int, int],
    new_size    : tuple[int, int],
    distortion  : float,
    area_thres  : int   = 0,
    aspect_thres: float = 0.0,
) -> np.ndarray:
    """Transform HBBs from original image to the corresponding fisheye images.

    Args:
        bbox: HBBs as ``numpy.ndarray`` in [N, 4+], CXCYWHN format (YOLO), normalized.
        old_size: Original image size as [W, H] format.
        new_size: Transformed image size as [W, H] format.
        distortion: Distortion factor.
        area_thres: Minimum area threshold for HBBs. Default: ``0``.
        aspect_thres: Minimum height-to-width ratio threshold for HBBs. Default: ``0.0``.
    """
    w0, h0      = old_size
    w1, h1      = new_size
    left_margin = int((w0 - w1) // 2)
    top_margin  = int((h0 - h1) // 2)

    bbox_new = []
    for b in bbox:
        # Convert normalized YOLO coords to pixel coords
        cx_n, cy_n, w_n, h_n = b[:4]
        cx = cx_n * w0
        cy = cy_n * h0
        w  = w_n  * w0
        h  = h_n  * h0
        #
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # top_left, top_right, bottom_left, bottom_right
        bbox_x = np.array([x1, x2, x1, x2]).astype(float)
        bbox_y = np.array([y1, y1, y2, y2]).astype(float)

        # Calculate the new coordinates individually
        dr    = np.zeros_like(bbox_x)
        b_x_u = np.zeros_like(bbox_x)
        b_y_u = np.zeros_like(bbox_y)
        for i in range(4):
            bbox_x[i] = (2 * bbox_x[i] - w0) / w0
            bbox_y[i] = (2 * bbox_y[i] - h0) / h0
            dr[i]     = np.sqrt(bbox_x[i] ** 2 + bbox_y[i] ** 2)
            b_x_u[i], b_y_u[i] = reverse_fisheye_xy_n(bbox_x[i], bbox_y[i], dr[i], distortion)
            b_x_u[i]  = int(((b_x_u[i] + 1) * w0) / 2)
            b_y_u[i]  = int(((b_y_u[i] + 1) * h0) / 2)

        x1_new = int(min(b_x_u)) - left_margin
        y1_new = int(min(b_y_u)) - top_margin
        x2_new = int(max(b_x_u)) - left_margin
        y2_new = int(max(b_y_u)) - top_margin
        #        
        cx_new     = (x1_new + x2_new) / 2
        cy_new     = (y1_new + y2_new) / 2
        w_new      = x2_new - x1_new
        h_new      = y2_new - y1_new
        area_new   = w_new * h_new
        aspect_new = h_new / w_new if h_new < w_new else w_new / h_new
        # Convert back to normalized YOLO format
        cx_n_new   = cx_new / w1
        cy_n_new   = cy_new / h1
        w_n_new    = w_new  / w1
        h_n_new    = h_new  / h1

        # Ensure coordinates are within [0, 1]
        if (    0 <= cx_n_new <= 1
            and 0 <= cy_n_new <= 1
            and w_n_new > 0
            and h_n_new > 0
            and area_new   >= area_thres
            and aspect_new >= aspect_thres
        ):
            # Preserve additional fields (e.g., class ID, confidence) if M>4
            bbox_new.append(np.concatenate(([cx_n_new, cy_n_new, w_n_new, h_n_new], b[4:])))

    return np.array(bbox_new, dtype=np.float32)


def transform_bbox1(
    bbox        : np.ndarray,
    old_size    : tuple[int, int],
    new_size    : tuple[int, int],
    distortion  : float,
    area_thres  : int   = 0,
    aspect_thres: float = 0.0,
) -> np.ndarray:
    """Transform HBBs from original image to the corresponding fisheye images.

    Args:
        bbox: HBBs as ``numpy.ndarray`` in [N, 4+], CXCYWHN format (YOLO), normalized.
        old_size: Original image size as [W, H] format.
        new_size: Transformed image size as [W, H] format.
        distortion: Distortion factor.
        area_thres: Minimum area threshold for HBBs. Default: ``0``.
        aspect_thres: Minimum height-to-width ratio threshold for HBBs. Default: ``0.0``.
    """
    w0, h0      = old_size
    w1, h1      = new_size
    left_margin = int((w0 - w1) // 2)
    top_margin  = int((h0 - h1) // 2)

    # Convert normalized CXCYWHN coords to CXCYWH format
    cx_n, cy_n, w_n, h_n, *rest = bbox.T
    n  = bbox.shape[0]  # Number of boxes
    cx = cx_n * w0
    cy = cy_n * h0
    w  = w_n  * w0
    h  = h_n  * h0
    # Convert CXCYWH to XYXY format
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    # Initialize output arrays
    x1_new = np.zeros(n)
    y1_new = np.zeros(n)
    x2_new = np.zeros(n)
    y2_new = np.zeros(n)

    # Generate transformed points for each box
    for i in range(n):
        # top_left, top_right, bottom_left, bottom_right
        bx = np.array([x1[i], x2[i], x1[i], x2[i]]).astype(float)
        by = np.array([y1[i], y1[i], y2[i], y2[i]]).astype(float)
        # Calculate the new coordinates individually
        dr   = np.zeros_like(bx)
        bx_u = np.zeros_like(bx)
        by_u = np.zeros_like(by)
        for j in range(4):
            bx[j] = (2 * bx[j] - w0) / w0
            by[j] = (2 * by[j] - h0) / h0
            dr[j] = np.sqrt(bx[j] ** 2 + by[j] ** 2)
            bx_u[j], by_u[j] = reverse_fisheye_xy_n(bx[j], by[j], dr[j], distortion)
            bx_u[j] = int(((bx_u[j] + 1) * w0) / 2)
            by_u[j] = int(((by_u[j] + 1) * h0) / 2)
        # Store new box coordinates
        x1_new[i] = int(min(bx_u)) - left_margin
        y1_new[i] = int(min(by_u)) - top_margin
        x2_new[i] = int(max(bx_u)) - left_margin
        y2_new[i] = int(max(by_u)) - top_margin

    # Convert back to CXCYWH format
    cx_new     = (x1_new + x2_new) / 2
    cy_new     = (y1_new + y2_new) / 2
    w_new      = x2_new - x1_new
    h_new      = y2_new - y1_new
    area_new   = w_new * h_new
    aspect_new = np.where(h_new < w_new, h_new / w_new, w_new / h_new)
    # Convert back to CXCYWHN format
    cx_n_new   = cx_new / w1
    cy_n_new   = cy_new / h1
    w_n_new    = w_new  / w1
    h_n_new    = h_new  / h1
    # Stack
    bbox_new = np.stack([cx_n_new, cy_n_new, w_n_new, h_n_new] + rest, axis=-1)
    bbox_new = [
        bbox_new[i] for i in range(n)
        if (    0 <= cx_n_new[i] <= 1
            and 0 <= cy_n_new[i] <= 1
            and    w_n_new[i] > 0
            and    h_n_new[i] > 0
            and   area_new[i] >= area_thres
            and aspect_new[i] >= aspect_thres
        )
    ]
    return bbox_new


def transform_bbox(
    bbox        : np.ndarray,
    old_size    : tuple[int, int],
    new_size    : tuple[int, int],
    distortion  : float,
    area_thres  : int   = 0,
    aspect_thres: float = 0.0,
    grid_points : int   = 5,
) -> np.ndarray:
    """Transform HBBs from original image to the corresponding fisheye images.

    Args:
        bbox: HBBs as ``numpy.ndarray`` in [N, 4+], CXCYWHN format (YOLO), normalized.
        old_size: Original image size as [W, H] format.
        new_size: Transformed image size as [W, H] format.
        distortion: Distortion factor.
        area_thres: Minimum area threshold for HBBs. Default: ``0``.
        aspect_thres: Minimum height-to-width ratio threshold for HBBs. Default: ``0.0``.
        grid_points: Number of points per axis for sampling within each box. Default: ``5``.
    """
    w0, h0      = old_size
    w1, h1      = new_size
    left_margin = int((w0 - w1) // 2)
    top_margin  = int((h0 - h1) // 2)

    # Convert normalized CXCYWHN coords to CXCYWH format
    cx_n, cy_n, w_n, h_n, *rest = bbox.T
    n  = bbox.shape[0]  # Number of boxes
    cx = cx_n * w0
    cy = cy_n * h0
    w  = w_n  * w0
    h  = h_n  * h0
    # Convert CXCYWH to XYXY format
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    # Initialize output arrays
    x1_new = np.zeros(n)
    y1_new = np.zeros(n)
    x2_new = np.zeros(n)
    y2_new = np.zeros(n)

    # Generate grid points for each box
    for i in range(n):
        # Create a grid of points within the box
        x_range = np.linspace(x1[i], x1[i] + w[i], grid_points)
        y_range = np.linspace(y1[i], y1[i] + h[i], grid_points)
        xx, yy  = np.meshgrid(x_range, y_range)
        bx      = xx.ravel()
        by      = yy.ravel()
        # Normalize to [-1, 1]
        bx_n = (2 * bx - w0) / w0
        by_n = (2 * by - h0) / h0
        dr   = np.sqrt(bx_n ** 2 + by_n ** 2)
        # Apply fisheye transformation
        bx_u, by_u = reverse_fisheye_xy_n(bx_n, by_n, dr, distortion)
        # Convert back to pixel coordinates
        bx_u = ((bx_u + 1) * w0) / 2 - left_margin
        by_u = ((by_u + 1) * h0) / 2 - top_margin
        # Store new box coordinates
        x1_new[i] = np.min(bx_u)
        y1_new[i] = np.min(by_u)
        x2_new[i] = np.max(bx_u)
        y2_new[i] = np.max(by_u)

    # Convert back to CXCYWH format
    cx_new     = (x1_new + x2_new) / 2
    cy_new     = (y1_new + y2_new) / 2
    w_new      = x2_new - x1_new
    h_new      = y2_new - y1_new
    area_new   = w_new * h_new
    aspect_new = np.where(h_new < w_new, h_new / w_new, w_new / h_new)
    # Convert back to CXCYWHN format
    cx_n_new   = cx_new / w1
    cy_n_new   = cy_new / h1
    w_n_new    = w_new  / w1
    h_n_new    = h_new  / h1
    # Stack
    bbox_new = np.stack([cx_n_new, cy_n_new, w_n_new, h_n_new] + rest, axis=-1)
    bbox_new = [
        bbox_new[i] for i in range(n)
        if (    0 <= cx_n_new[i] <= 1
            and 0 <= cy_n_new[i] <= 1
            and    w_n_new[i] > 0
            and    h_n_new[i] > 0
            and   area_new[i] >= area_thres
            and aspect_new[i] >= aspect_thres
        )
    ]
    return bbox_new


# ----- Augmentation -----
@ALBUMENTATIONS.register()
class iFishTransform(DualTransform):
    """Apply fisheye transformation to inputs.

    Args:
        distortion: Distortion factor.
        area_thres: Minimum area threshold for bounding boxes. Default: ``0``.
        aspect_thres: Minimum height-to-width ratio threshold for bounding boxes.
            Default: ``0.0``.
        p: Probability of applying the transformation. Default: ``1.0``.
    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES)

    class InitSchema(BaseTransformInitSchema):
        distortion  : float = Field(ge=0.0)
        area_thres  : int   = Field(ge=0)
        aspect_thres: float = Field(ge=0.0)
        p           : float = 1.0

    def __init__(
        self,
        distortion  : float = 1.0,
        area_thres  : int   = 0,
        aspect_thres: float = 0.0,
        p           : float = 1
    ):
        super().__init__(p=p)
        self.distortion   = distortion
        self.area_thres   = area_thres
        self.aspect_thres = aspect_thres

    def apply(
        self,
        img          : np.ndarray,
        fisheye_image: np.ndarray,
        old_size     : tuple[int, int],
        new_size     : tuple[int, int],
        *args: Any, **params: Any
    ) -> np.ndarray:
        return fisheye_image

    def apply_to_mask(
        self,
        img          : np.ndarray,
        fisheye_image: np.ndarray,
        old_size     : tuple[int, int],
        new_size     : tuple[int, int],
        *args: Any, **params: Any
    ) -> np.ndarray:
        return transform_image(img, self.distortion)

    def apply_to_bboxes(
        self,
        bboxes       : np.ndarray,
        fisheye_image: np.ndarray,
        old_size     : tuple[int, int],
        new_size     : tuple[int, int],
        *args: Any, **params: Any
    ) -> np.ndarray:
        return transform_bbox0(bboxes, old_size, new_size, self.distortion, self.area_thres, self.aspect_thres)

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Returns parameters dependent on input."""
        image         = data["image"]
        fisheye_image = transform_image(image, self.distortion)
        h0, w0        = I.imgsz(image)
        h1, w1        = I.imgsz(fisheye_image)
        return params | {
            "fisheye_image": fisheye_image,
            "old_size"     : (w0, h0),
            "new_size"     : (w1, h1),
        }
