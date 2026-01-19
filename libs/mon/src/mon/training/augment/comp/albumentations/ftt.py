#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for fisheye tomography transformation.

This module defines a fisheye tomography transformation class that can be
used as an augmentation technique in image processing pipelines. It applies
fisheye distortion to images and masks, with options for customizing focal length,
image size, background color/label, and transformation parameters.
"""

from __future__ import annotations

__all__ = [
    "FisheyeTomographyTransform"
]

import math
import random
from typing import Any

import cv2
import numpy as np
from albumentations.core.transforms_interface import (
    BaseTransformInitSchema,
    DualTransform,
)
from albumentations.core.type_definitions import Targets
from pydantic import Field

from mon.core import ALBUMENTATIONS, bbox as B, BBoxFormat, image as I, log


@ALBUMENTATIONS.register()
class FisheyeTomographyTransform(DualTransform):
    """A transformation that applies fisheye tomography distortion to images
    and masks.

    References:
        - Code: https://github.com/Zane-Gu/AirEyeSeg

    Attributes:
        _targets (tuple): The target types this transform can be applied to.
            Supports images and masks.
        _f (int): Focal length for fisheye transformation.
        _imgsz (tuple): Size of the output image (height, width).
        _ratio (float): Ratio for scaling coordinates.
        _bg_color (tuple): Background color for areas outside the fisheye
            transformation.
        _bg_label (int): Background label for mask areas outside the fisheye
            transformation.
        _reuse (bool): Whether to reuse the calculated coordinate map.
        _p (float): Probability of applying the transformation.
        _bad_index (np.ndarray): Mask for bad pixels outside the fisheye area.
        _param (int): Parameter for pinhole camera model.
        _alpha_range (list): Range for alpha rotation parameter.
        _beta_range (list): Range for beta rotation parameter.
        _theta_range (list): Range for theta rotation parameter
    """

    _targets = (Targets.IMAGE, Targets.MASK)

    class InitSchema(BaseTransformInitSchema):
        f        : int   = Field(ge=0)
        imgsz    : int   = Field(ge=0)
        bg_color : tuple = (0, 0, 0)
        bg_label : int   = 20
        reuse    : bool  = False
        p        : float = 1.0

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        f        : int   = 150,
        imgsz    : int   = 720,
        bg_color : tuple[int, int, int] = (0, 0, 0),
        bg_label : int   = 20,
        reuse    : bool  = False,
        p        : float = 1.0,
    ):
        """Initializes the FisheyeTomographyTransform instance.

        Args:
            f (int): Focal length for fisheye transformation. Defaults to 150.
            imgsz (int): Size of the output image (height, width). Defaults to 720.
            bg_color (tuple): Background color for areas outside the fisheye
                transformation. Defaults to (0, 0, 0).
            bg_label (int): Background label for mask areas outside the fisheye
                transformation. Defaults to 20.
            reuse (bool): Whether to reuse the calculated coordinate map.
                Defaults to False.
            p (float): Probability of applying the transformation. Defaults to 1.0.
        """
        super().__init__(p=p)
        self._f        = f
        self._imgsz    = I.imgsz(imgsz)
        self._ratio    = min(self._imgsz[0], self._imgsz[1]) / (self._f * math.pi)
        self._bg_color = bg_color
        self._bg_label = bg_label
        self._reuse    = reuse

        # Mask for bad pixels
        mask     = np.ones([self._imgsz[0], self._imgsz[1]], dtype=np.uint8)
        square_r = (min(self._imgsz[0], self._imgsz[1]) / 2) ** 2
        for i in range(self._imgsz[0]):
            for j in range(self._imgsz[1]):
                if ((i - self._imgsz[0] / 2) ** 2 + (j - self._imgsz[1] / 2) ** 2) >= square_r:
                    mask[i, j] = 0
        mask = np.array(mask)
        mask = mask.reshape(-1)
        self._bad_index = (mask == 0)

        # Parameters
        self._param         = 500
        self._alpha_range   = [0, 0]
        self._beta_range    = [0, 0]
        self._theta_range   = [0, 0]
        self._x_trans_range = [-self._imgsz[1] / 2, self._imgsz[1] / 2]
        self._y_trans_range = [-self._imgsz[0] / 2, self._imgsz[0] / 2]
        self._z_trans_range = [-0.6 * self._param , 0.6 * self._param ]
        self._alpha         = 0
        self._beta          = 0
        self._theta         = 0
        self._x_trans       = 0
        self._y_trans       = 0
        self._z_trans       = 0

    # --- Initialize ---
    def set_ext_params_range(self, ext_params_range: list[int]):
        """Sets the range for external parameters.

        Args:
            ext_params_range (list[int]): List of ranges for external parameters
                [alpha_range, beta_range, theta_range,
                x_trans_range, y_trans_range, z_trans_range].
        """
        self._alpha_range   = [-ext_params_range[0] * math.pi / 180, ext_params_range[0] * math.pi / 180]
        self._beta_range    = [-ext_params_range[1] * math.pi / 180, ext_params_range[1] * math.pi / 180]
        self._theta_range   = [-ext_params_range[2] * math.pi / 180, ext_params_range[2] * math.pi / 180]
        self._x_trans_range = [-self._imgsz[1] * ext_params_range[3], self._imgsz[1] * ext_params_range[3]]
        self._y_trans_range = [-self._imgsz[0] * ext_params_range[4], self._imgsz[0] * ext_params_range[4]]
        self._z_trans_range = [-ext_params_range[5] * self._param   , ext_params_range[5] * self._param]

    def set_ext_params(self, ext_params: list[int]):
        """Sets the external parameters.

        Args:
            ext_params (list[int]): List of external parameters
                [alpha, beta, theta, x_trans, y_trans, z_trans].
        """
        self._alpha   = ext_params[0] * math.pi / 180
        self._beta    = ext_params[1] * math.pi / 180
        self._theta   = ext_params[2] * math.pi / 180
        self._x_trans = ext_params[3] * self._imgsz[1]
        self._y_trans = ext_params[4] * self._imgsz[0]
        self._z_trans = ext_params[5] * self._param

    def random_focal_len(self, focal_len_range: tuple[int, int] = (200, 400)):
        """Randomly set the focal length.

        Args:
            focal_len_range (tuple[int, int]): Range for focal length. Defaults
                to (200, 400).
        """
        tmp    = random.random()
        self._f = focal_len_range[0] * (1 - tmp) + focal_len_range[1] * tmp

    def random_ext_params(self):
        """Randomly set the external parameters."""
        tmp1         = random.random()
        self._alpha   = self._alpha_range[0] * (1 - tmp1) + self._alpha_range[1] * tmp1
        tmp2         = random.random()
        self._beta    = self._beta_range[0] * (1 - tmp2) + self._beta_range[1] * tmp2
        tmp3         = random.random()
        self._theta   = self._theta_range[0] * (1 - tmp3) + self._theta_range[1] * tmp3
        tmp4         = random.random()
        self._x_trans = self._x_trans_range[0] * (1 - tmp4) + self._x_trans_range[1] * tmp4
        tmp5         = random.random()
        self._y_trans = self._y_trans_range[0] * (1 - tmp5) + self._y_trans_range[1] * tmp5
        tmp6         = random.random()
        self._z_trans = self._z_trans_range[0] * (1 - tmp6) + self._z_trans_range[1] * tmp6

    # --- Internal Calculation ---
    def _calculate_coord_map(self, image: np.ndarray):
        """Calculates the coordinate map for fisheye transformation.

        Args:
            image (numpy.ndarray): The input image as a numpy.ndarray of shape
                (H, W, C) with pixel values in the range [0, 255].
        """
        self._init_ext_matrix()
        self._init_pin_matrix(image.shape)

        src_rows = image.shape[0]
        src_cols = image.shape[1]
        dst_rows = self._imgsz[0]
        dst_cols = self._imgsz[1]

        #
        cord_x, cord_y = np.meshgrid(np.arange(dst_cols), np.arange(dst_rows))
        cord = np.dstack((cord_x, cord_y)).astype(np.float32) - np.array([dst_cols / 2, dst_rows / 2])
        cord = cord.reshape(-1, 2)

        # shape=(dst_rows * dst_cols, 2)
        cord = np.array(cord) / self._ratio

        radius_array = np.sqrt(np.square(cord[:, 0]) + np.square(cord[:, 1]))
        theta_array  = radius_array / self._f

        new_x_array  = np.tan(theta_array) * cord[:, 0] / radius_array * self._f
        new_y_array  = np.tan(theta_array) * cord[:, 1] / radius_array * self._f

        temp_index1  = radius_array == 0
        temp_index2  = cord[:, 0] == 0
        temp_index3  = cord[:, 1] == 0
        bad_x_index  = temp_index1 | (temp_index2 & temp_index1)
        bad_y_index  = temp_index1 | (temp_index3 & temp_index1)

        new_x_array[bad_x_index] = 0
        new_y_array[bad_y_index] = 0

        new_x_array = new_x_array.reshape((-1, 1))
        new_y_array = new_y_array.reshape((-1, 1))

        new_cord = np.hstack((new_x_array, new_y_array))
        new_cord = np.hstack((new_cord, np.ones((dst_rows * dst_cols, 1)) * self._param))
        new_cord = np.hstack((new_cord, np.ones((dst_rows * dst_cols, 1))))

        # shape=(pix_num, 4)
        pin_camera_array = np.matmul(self.rotate_trans_matrix, new_cord.T).T

        # shape=(pix_num, 3)
        pin_image_cords  = np.matmul(self.pin_matrix, pin_camera_array.T).T

        self.map_cols = pin_image_cords[:, 0] / pin_image_cords[:, 2]
        self.map_rows = pin_image_cords[:, 1] / pin_image_cords[:, 2]

        self.map_cols = self.map_cols.round().astype(int)
        self.map_rows = self.map_rows.round().astype(int)

        index1 = self.map_rows < 0
        index2 = self.map_rows >= src_rows
        index3 = self.map_cols < 0
        index4 = self.map_cols >= src_cols
        index5 = pin_image_cords[:, 2] <= 0

        bad_index = index1 | index2 | index3 | index4 | index5
        bad_index = bad_index | self._bad_index
        self.map_cols[bad_index] = image.shape[1]
        self.map_rows[bad_index] = 0

    def _init_ext_matrix(self):
        """Initializes the external rotation and translation matrix."""
        self.rotate_trans_matrix = \
            np.array([
                [
                    math.cos(self._beta) * math.cos(self._theta),
                    math.cos(self._beta) * math.sin(self._theta),
                    -math.sin(self._beta),
                     self._x_trans
                ],
                [
                    -math.cos(self._alpha) * math.sin(self._theta) + math.sin(self._alpha) * math.sin(self._beta) * math.cos(self._theta),
                    math.cos(self._alpha) * math.cos(self._theta) + math.sin(self._alpha) * math.sin(self._beta) * math.sin(self._theta),
                    math.sin(self._alpha) * math.cos(self._beta),
                     self._y_trans
                ],
                [
                    math.sin(self._alpha) * math.sin(self._theta) + math.cos(self._alpha) * math.sin(self._beta) * math.cos(self._theta),
                    -math.sin(self._alpha) * math.cos(self._theta) + math.cos(self._alpha) * math.sin(self._beta) * math.sin(self._theta),
                    math.cos(self._alpha) * math.cos(self._beta),
                     self._z_trans
                ],
                [0, 0, 0, 1]
            ])

    def _init_pin_matrix(self, shape: tuple[int, int, int]):
        """Initializes the pinhole camera matrix.

        Args:
            shape (tuple[int, int, int]): Shape of the input image as (H, W, C).
        """
        rows = shape[0]
        cols = shape[1]
        self.pin_matrix = \
            np.array([
                [self._param, 0,           cols / 2, 0],
                [0,           self._param, rows / 2, 0],
                [0,           0,           1,        0]
            ])

    # --- Distortion ---
    def _transform_image(self, image: np.ndarray) -> np.ndarray:
        """Applies fisheye transformation to the input image.

        Args:
            image (numpy.ndarray): The input image as a numpy.ndarray of shape
                (H, W, C) with pixel values in the range [0, 255].

        Returns:
            numpy.ndarray: The fisheye transformed image as a numpy.ndarray of
                shape (H, W, C) with pixel values in the range [0, 255].
        """
        if not self._reuse:
            self._calculate_coord_map(image)

        fisheye = np.hstack((image, np.zeros((image.shape[0], 1, 3), dtype=np.uint8)))
        fisheye[0, fisheye.shape[1] - 1] = self._bg_color
        fisheye = np.array(fisheye[(self.map_rows, self.map_cols)])
        fisheye = fisheye.reshape(self._imgsz[0], self._imgsz[1], 3)
        return fisheye

    def _transform_mask(self, image: np.ndarray) -> np.ndarray:
        """Applies fisheye transformation to the input mask.

        Args:
            image (numpy.ndarray): The input mask as a numpy.ndarray of shape
                (H, W) with integer label values.

        Returns:
            numpy.ndarray: The fisheye transformed mask as a numpy.ndarray of
                shape (H, W) with integer label values.
        """
        if not self._reuse:
            self._calculate_coord_map(image)

        fisheye = np.hstack((image, np.zeros((image.shape[0], 1), dtype=np.uint8)))
        fisheye[0, fisheye.shape[1] - 1] = self._bg_label
        fisheye = np.array(fisheye[(self.map_rows, self.map_cols)])
        fisheye = fisheye.reshape(self._imgsz[0], self._imgsz[1], 3)
        return fisheye

    def _transform_bbox(self, bbox: np.ndarray, old_size: tuple[int, int]) -> np.ndarray:
        """Applies fisheye transformation to the input bounding boxes.

        Args:
            bbox (numpy.ndarray): The input bounding boxes as a numpy.ndarray of
                shape (N, 7+) in CXCYWHN format.
            old_size (tuple[int, int]): The original size of the image as (H, W).

        Returns:
            numpy.ndarray: The fisheye transformed bounding boxes as a
                numpy.ndarray of shape (M, 7+) in CXCYWHN format, where M <= N.
        """
        imgsz  = self._imgsz
        h0, w0 = I.imgsz(old_size)
        bbox   = B.convert(bbox, fmt=BBoxFormat.CXCYWHN2XYXY, imgsz=(h0, w0))

        t_bbox = []
        for b in bbox:
            # Convert bbox to XYXY format
            x1, y1, x2, y2, _, c = int(b[0]), int(b[1]), int(b[2]), int(b[3]), int(b[4]), int(b[5])

            # Create a canvas for the bbox
            canvas = np.zeros((h0, w0, 3), dtype=np.uint8)
            canvas[y1:y2, x1:x2, :] = [255, 255, 255]  # White in BGR format # image[y1:y2, x1:x2]

            # Apply fisheye transformation on the canvas
            t_canvas = self._transform_image(image=canvas)

            # Find contours
            ret, thresh = cv2.threshold(t_canvas, 200, 255, cv2.THRESH_BINARY)
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

            # Update bbox
            t_b = b
            t_b[0] = min_x
            t_b[1] = min_y
            t_b[2] = max_x
            t_b[3] = max_y
            t_bbox.append(t_b)

        # Convert back to YOLO format
        t_bbox = np.array(t_bbox, dtype=np.float32)
        t_bbox = B.convert(t_bbox, fmt=BBoxFormat.XYXY2CXCYWHN, imgsz=imgsz)
        return t_bbox

    # --- Apply ---
    def apply(self, img: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        """Applies fisheye transformation to the input image.

        Args:
            img (numpy.ndarray): The input image as a numpy.ndarray of shape
                (H, W, C) with pixel values in the range [0, 255].

        Returns:
            numpy.ndarray: The fisheye transformed image as a numpy.ndarray of
                shape (H, W, C) with pixel values in the range [0, 255].
        """
        return self._transform_image(img)

    def apply_to_mask(self, mask: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        """Applies fisheye transformation to the input mask.

        Args:
            mask (numpy.ndarray): The input mask as a numpy.ndarray of shape
                (H, W) with integer label values.

        Returns:
            numpy.ndarray: The fisheye transformed mask as a numpy.ndarray of
                shape (H, W) with integer label values.
        """
        return self._transform_mask(mask)

    def apply_to_bboxes(self, bboxes: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        """Applies fisheye transformation to the input bounding boxes.

        Args:
            bboxes (numpy.ndarray): The input bounding boxes as a numpy.ndarray
                of shape (N, 7+) in CXCYWHN format.

        Returns:
            numpy.ndarray: The fisheye transformed bounding boxes as a
                numpy.ndarray of shape (M, 7+) in CXCYWHN format, where M <= N.
        """
        return self._transform_bbox(bboxes, old_size=params["old_size"])

    # --- Utils ---
    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Gets parameters dependent on the input data.

        Args:
            params (dict[str, Any]): The current parameters.
            data (dict[str, Any]): The input data containing the image.

        Returns:
            dict[str, Any]: The updated parameters including the original image
                size.
        """
        image  = data["image"]
        h0, w0 = I.imgsz(image)
        return params | {
            "old_size": (h0, w0),
        }

    def print_ext_param(self):
        """Prints the external parameters."""
        log(f"alpha:         {self._alpha * 180 / math.pi}.")
        log(f"beta:          {self._beta  * 180 / math.pi}.")
        log(f"theta:         {self._theta * 180 / math.pi}.")
        log(f"X translation: {self._x_trans}.")
        log(f"Y translation: {self._y_trans}.")
        log(f"Z translation: {self._z_trans}.")
