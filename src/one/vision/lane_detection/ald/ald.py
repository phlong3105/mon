#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Advanced Lane Detection: Computer Vision algorithm to compute road curvature
and lane vehicle offset using OpenCV Image Processing, Camera Calibration,
Perspective Transform, Color Masks, Sobel and Polynomial Fit.

References:
    https://github.com/OanaGaskey/Advanced-Lane-Detection
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Optional
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from munch import Munch

from one.core import error_console
from one.core import MODELS
from one.io import create_dirs
from one.io import is_image_file
from one.io import is_video_file
from one.io import is_yaml_file
from one.io import load
from one.vision.lane_detection.lane_detector import LaneDetector

__all__ = [
    "AdvancedLaneDetector",
    "run",
]


# MARK: - Module

# noinspection PyMethodMayBeStatic
@MODELS.register(name="advanced_lane_detector")
class AdvancedLaneDetector(LaneDetector):
    """Advanced Lane Detector: Computer Vision algorithm to compute road
    curvature and lane vehicle offset using OpenCV Image Processing, Camera
    Calibration, Perspective Transform, Color Masks, Sobel and Polynomial Fit.
    
    Attributes:
    
    """

    # MARK: Magic Functions
    
    def __init__(
        self,
        config               : Optional[Union[str, dict, Munch]] = None,
        offset               : Optional[int]        = 300,
        src_points           : Optional[np.ndarray] = None,
        camera_matrix        : Optional[np.ndarray] = None,
        distortion_coeffs    : Optional[np.ndarray] = None,
        camera_calib_dir     : Optional[str]        = os.path.join(os.getcwd(), "data", "camera_calib"),
        camera_undistored_dir: Optional[str]        = os.path.join(os.getcwd(), "data", "camera_undistored"),
        *args, **kwargs
    ):
        super().__init__()
        self.config                = config
        self.offset                = offset
        self.src_points            = src_points
        self.camera_matrix         = camera_matrix
        self.distortion_coeffs     = distortion_coeffs
        self.camera_calib_dir      = camera_calib_dir
        self.camera_undistored_dir = camera_undistored_dir
        self.left_fit_hist         = np.array([])
        self.right_fit_hist        = np.array([])
        self.prev_left_fit         = np.array([])
        self.prev_right_fit        = np.array([])
        self.init_config()
        self.init_intrinsic()
    
    # MARK: Properties
    
    @property
    def config(self) -> Optional[Munch]:
        return self._config
    
    @config.setter
    def config(self, config: Optional[Union[str, dict, Munch]]):
        if isinstance(config, str) and is_yaml_file(config):
            config = load(config)
        if isinstance(config, dict):
            self._config = Munch(dict)
        else:
            self._config = config
    
    @property
    def offset(self) -> int:
        return self._offset

    @offset.setter
    def offset(self, offset: Optional[int]):
        if offset is None and self.config:
            offset = self.config.get("offset", None)
        if offset:
            self._offset = offset
        else:
            raise RuntimeError(f"`offset` cannot be initialized with: {offset}.")

    @property
    def src_points(self) -> np.ndarray:
        return self._src_points

    @src_points.setter
    def src_points(self, src_points: Optional[np.ndarray]):
        if src_points is None and self.config:
            src_points = self.config.get("src_points", None)
        if src_points:
            self._src_points = src_points
        else:
            raise RuntimeError(f"`src_points` cannot be initialized with: {src_points}.")

    @property
    def camera_matrix(self) -> np.ndarray:
        return self._camera_matrix

    @camera_matrix.setter
    def camera_matrix(self, camera_matrix: Optional[np.ndarray]):
        if camera_matrix is None and self.config:
            camera_matrix = self.config.get("camera_matrix", None)
        if camera_matrix:
            self._camera_matrix = camera_matrix
        else:
            raise RuntimeError(f"`camera_matrix` cannot be initialized with: "
                               f"{camera_matrix}.")

    @property
    def distortion_coeffs(self) -> np.ndarray:
        return self._distortion_coeffs

    @distortion_coeffs.setter
    def distortion_coeffs(self, distortion_coeffs: Optional[np.ndarray]):
        if distortion_coeffs is None and self.config:
            distortion_coeffs = self.config.get("distortion_coeffs", None)
        if distortion_coeffs:
            self._distortion_coeffs = distortion_coeffs
        else:
            raise RuntimeError(
                f"`distortion_coeffs` cannot be initialized with: "
                f"{distortion_coeffs}."
            )

    @property
    def camera_calib_dir(self) -> Optional[str]:
        return self._camera_calib_dir

    @camera_calib_dir.setter
    def camera_calib_dir(self, camera_calib_dir: Optional[str]):
        if camera_calib_dir is None and self.config:
            camera_calib_dir = self.config.get("camera_calib_dir", None)
        self._camera_calib_dir = camera_calib_dir

    @property
    def camera_undistored_dir(self) -> Optional[str]:
        return self._camera_undistored_dir

    @camera_undistored_dir.setter
    def camera_undistored_dir(self, camera_undistored_dir: Optional[str]):
        if camera_undistored_dir is None and self.config:
            camera_undistored_dir = self.config.get("camera_undistored_dir", None)
        self._camera_undistored_dir = camera_undistored_dir
        create_dirs(paths=[self._camera_undistored_dir])
    
    @property
    def has_intrinsics(self) -> bool:
        return self.camera_matrix is not None and self.distortion_coeffs is not None
    
    # MARK: Configure
    
    def init_intrinsic(self):
        """Initialize camera intrinsics matrix and distortion coefficients."""
        if not self.has_intrinsics and self.camera_calib_dir is None:
            raise RuntimeError(
                f"Intrinsic parameters cannot be initialized. "
                f"`camera_matrix`, `distortion_coeffs`, and `camera_calib_dir` "
                f"must be defined. "
                f"But got: {self.camera_matrix}, {self.distortion_coeffs}, "
                f"and {self.camera_calib_dir}."
            )
        if self.has_intrinsics:
            pass
        elif self.camera_calib_dir:
            self.calibrate_camera()
    
    # STEP 1: Camera Calibration
    def calibrate_camera(self):
        """Calibrate the camera from chessboard images."""
        image_files = []
        if self.camera_calib_dir is not None:
            image_files = glob.glob(os.path.join(self.camera_calib_dir, "*"))
        if len(image_files) == 0:
            raise RuntimeError(f"No calibration images can be found at: "
                               f"{self.camera_calib_dir}.")
        
        # Prepare object points. From the provided calibration images, 9*6
        # corners are identified
        nx         = 9
        ny         = 6
        obj_points = []
        img_points = []
        # Object points are real world points, here a 3D coordinates matrix is
        # generated z coordinates are 0 and x, y are equidistant as it is known
        # that the chessboard is made of identical squares
        objp        = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Imagepoints are the coresspondant object points with their coordinates
        # in the distorted image. They are found in the image using the Open CV
        # 'findChessboardCorners' function
        for image_file in image_files:
            if not is_image_file(image_file):
                continue
            image = cv2.imread(image_file)
            # Convert to grayscale
            gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            # If found, draw corners
            if ret:
                # Draw and display the corners
                # cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                img_points.append(corners)
                obj_points.append(objp)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, gray.shape[::-1], None, None
        )
        self.camera_matrix     = mtx
        self.distortion_coeffs = dist

        # Checking the undistored image
        if self.camera_undistored_dir:
            for image_file in image_files:
                image_name  = os.path.basename(image_file)
                img         = cv2.imread(image_name)
                undistorted = cv2.undistort(img, mtx, dist, None, mtx)
                export_to   = os.path.join(self.camera_undistored_dir, image_name)
                cv2.imwrite(export_to, undistorted)
    
    # MARK: Forward Pass
    
    def forward_once(self, x: np.ndarray, *args, **kwargs):
        binary_thresh        = self.binary_thresholded(x)
        binary_warped, m_inv = self.warp_image(binary_thresh)
    
        # Checking
        # binary_thresh_s = np.dstack((binary_thresh, binary_thresh, binary_thresh)) * 255
        # binary_warped_s = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    
        if len(self.left_fit_hist) == 0:
            left_x, left_y, right_x, right_y = \
                self.find_lane_pixels_using_histogram(binary_warped)
            left_fit, right_fit, left_fit_x, right_fit_x, plot_y = \
                self.fit_poly(binary_warped, left_x, left_y, right_x, right_y)
            # Store fit in history
            self.left_fit_hist  = np.array(left_fit)
            new_left_fit        = np.array(left_fit)
            self.left_fit_hist  = np.vstack([self.left_fit_hist, new_left_fit])
            self.right_fit_hist = np.array(right_fit)
            new_right_fit       = np.array(right_fit)
            self.right_fit_hist = np.vstack([self.right_fit_hist, new_right_fit])
    
        else:
            self.prev_left_fit  = [
                np.mean(self.left_fit_hist[:, 0]),
                np.mean(self.left_fit_hist[:, 1]),
                np.mean(self.left_fit_hist[:, 2])
            ]
            self.prev_right_fit = [
                np.mean(self.right_fit_hist[:, 0]),
                np.mean(self.right_fit_hist[:, 1]),
                np.mean(self.right_fit_hist[:, 2])
            ]
            left_x, left_y, right_x, right_y \
                = self.find_lane_pixels_using_prev_poly(binary_warped)
            if len(left_y) == 0 or len(right_y) == 0:
                left_x, left_y, right_x, right_y = \
                    self.find_lane_pixels_using_histogram(binary_warped)
            left_fit, right_fit, left_fit_x, right_fit_x, plot_y = \
                self.fit_poly(binary_warped, left_x, left_y, right_x, right_y)
        
            # Add new values to history
            new_left_fit        = np.array(left_fit)
            self.left_fit_hist  = np.vstack([self.left_fit_hist, new_left_fit])
            new_right_fit       = np.array(right_fit)
            self.right_fit_hist = np.vstack([self.right_fit_hist, new_right_fit])
        
            # Remove old values from history
            if len(self.left_fit_hist) > 5:  # 10
                self.left_fit_hist  = np.delete(self.left_fit_hist, 0, 0)
                self.right_fit_hist = np.delete(self.right_fit_hist, 0, 0)
    
        # Checking
        draw_poly_img = self.draw_poly_lines(binary_warped, left_fit_x, right_fit_x, plot_y)
    
        left_curve_rad, right_curve_rad = \
            self.measure_curvature_meters(binary_warped, left_fit_x, right_fit_x, plot_y)
        veh_pos = self.measure_position_meters(binary_warped, left_fit, right_fit)
        out_image, color_warp_img, new_warp = self.project_lane_info(
            image           = x,
            binary_warped   = binary_warped,
            plot_y          = plot_y,
            left_fit_x      = left_fit_x,
            right_fit_x     = right_fit_x,
            m_inv           = m_inv,
            left_curve_rad  = left_curve_rad,
            right_curve_rad = right_curve_rad,
            veh_pos         = veh_pos
        )
        return out_image, veh_pos, color_warp_img, draw_poly_img

    # STEP 2: Perspective Transform from Car Camera to Bird's Eye View
    def warp_image(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        undistorted = cv2.undistort(image, self.matrix, self.distortion, None, self.matrix)
        img_size    = (image.shape[1], image.shape[0])
        offset      = self.offset
        # Source points taken from images with straight lane lines, these are
        # to become parallel after the warp transform
        src = self.src_points
        # Destination points are to be parallel, taken into account the image
        # size
        dst = np.float32(
            [
                [offset, img_size[1]],               # bottom-left corner
                [offset, 0],                         # top-left corner
                [img_size[0] - offset, 0],           # top-right corner
                [img_size[0] - offset, img_size[1]]  # bottom-right corner
            ]
        )
        # Calculate the transformation matrix and it's inverse transformation
        matrix     = cv2.getPerspectiveTransform(src, dst)
        matrix_inv = cv2.getPerspectiveTransform(dst, src)
        warped     = cv2.warpPerspective(undistorted, matrix, img_size)
        return warped, matrix_inv

    # STEP 3: Process Binary Thresholded Images
    def binary_threshold(self, image: np.ndarray) -> np.ndarray:
        # Transform image to gray scale
        gray_img     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply sobel (derivative) in x direction, this is useful to detect
        # lines that tend to be vertical
        sobel_x      = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0)
        abs_sobel_x  = np.absolute(sobel_x)
        # Scale result to 0-255
        scaled_sobel = np.uint8(255 * abs_sobel_x / np.max(abs_sobel_x))
        sx_binary    = np.zeros_like(scaled_sobel)
        # Keep only derivative values that are in the margin of interest
        sx_binary[(scaled_sobel >= 30) & (scaled_sobel <= 255)] = 1
        
        # Detect pixels that are white in the grayscale image
        white_binary = np.zeros_like(gray_img)
        white_binary[(gray_img > 200) & (gray_img <= 255)] = 1
        
        # Convert image to HLS
        hls        = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        h          = hls[:, :, 0]
        s          = hls[:, :, 2]
        sat_binary = np.zeros_like(s)
        # Detect pixels that have a high saturation value
        sat_binary[(s > 90) & (s <= 255)] = 1
        
        hue_binary = np.zeros_like(h)
        # Detect pixels that are yellow using the hue component
        hue_binary[(h > 10) & (h <= 25)] = 1
        
        # Combine all pixels detected above
        binary_1 = cv2.bitwise_or(sx_binary,  white_binary)
        binary_2 = cv2.bitwise_or(hue_binary, sat_binary)
        binary   = cv2.bitwise_or(binary_1,   binary_2)

        return binary

    # STEP 4: Detection of Lane Lines Using Histogram
    def find_lane_pixels_using_histogram(self, binary_warped: np.ndarray):
        # Take a histogram of the bottom half of the image
        histogram    = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint     = np.int(histogram.shape[0] // 2)
        left_x_base  = np.argmax(histogram[:midpoint])
        right_x_base = np.argmax(histogram[midpoint:]) + midpoint
        
        n_windows    = 9    # Choose the number of sliding windows
        margin       = 100  # Set the width of the windows +/- margin
        min_pix      = 50   # Set minimum number of pixels found to recenter window
        
        # Set height of windows - based on n_windows above and image shape
        window_height   = np.int(binary_warped.shape[0] // n_windows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero         = binary_warped.nonzero()
        nonzero_y       = np.array(nonzero[0])
        nonzero_x       = np.array(nonzero[1])
        # Current positions to be updated later for each window in n_windows
        left_x_current  = left_x_base
        right_x_current = right_x_base
        
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds  = []
        right_lane_inds = []
        
        # Step through the windows one by one
        for window in range(n_windows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low        = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high       = binary_warped.shape[0] - window * window_height
            win_x_left_low   = left_x_current  - margin
            win_x_left_high  = left_x_current  + margin
            win_x_right_low  = right_x_current - margin
            win_x_right_high = right_x_current + margin
            
            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = (
                (nonzero_y >= win_y_low)      & (nonzero_y < win_y_high) &
                (nonzero_x >= win_x_left_low) & (nonzero_x < win_x_left_high)
            ).nonzero()[0]
            good_right_inds = (
                (nonzero_y >= win_y_low)       & (nonzero_y < win_y_high) &
                (nonzero_x >= win_x_right_low) & (nonzero_x < win_x_right_high)
            ).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # If you found > min_pix pixels, recenter next window on their mean position
            if len(good_left_inds) > min_pix:
                left_x_current = np.int(np.mean(nonzero_x[good_left_inds]))
            if len(good_right_inds) > min_pix:
                right_x_current = np.int(np.mean(nonzero_x[good_right_inds]))
        
        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds  = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass
        
        # Extract left and right line pixel positions
        left_x  = nonzero_x[left_lane_inds]
        left_y  = nonzero_y[left_lane_inds]
        right_x = nonzero_x[right_lane_inds]
        right_y = nonzero_y[right_lane_inds]
        return left_x, left_y, right_x, right_y
    
    def fit_poly(
        self,
        binary_warped: np.ndarray,
        left_x       : np.ndarray,
        left_y       : np.ndarray,
        right_x      : np.ndarray,
        right_y      : np.ndarray
    ):
        # Fit a second order polynomial to each with np.polyfit()
        left_fit  = np.polyfit(left_y,  left_x,  2)
        right_fit = np.polyfit(right_y, right_x, 2)
        
        # Generate x and y values for plotting
        plot_y = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        try:
            left_fit_x  = left_fit[0]  * plot_y ** 2 + left_fit[1]  * plot_y + left_fit[2]
            right_fit_x = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            error_console("The function failed to fit a line!")
            left_fit_x  = 1 * plot_y ** 2 + 1 * plot_y
            right_fit_x = 1 * plot_y ** 2 + 1 * plot_y
    
        return left_fit, right_fit, left_fit_x, right_fit_x, plot_y

    # STEP 5: Detection of Lane Lines Based on Previous Step
    def find_lane_pixels_using_prev_poly(self, binary_warped: np.ndarray):
        # Width of the margin around the previous polynomial to search
        margin    = int(binary_warped.shape[1] * (100 / 1920))  # int(1280 * (100 / 1920))
        # Grab activated pixels
        nonzero   = binary_warped.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])
        # Set the area of search based on activated x-values within the +/-
        # margin of our polynomial function
        left_lane_inds = (
            (nonzero_x > (
                self.prev_left_fit[0] * (nonzero_y ** 2) +
                self.prev_left_fit[1] * nonzero_y +
                self.prev_left_fit[2] - margin)
             ) &
            (nonzero_x < (
                self.prev_left_fit[0] * (nonzero_y ** 2) +
                self.prev_left_fit[1] * nonzero_y +
                self.prev_left_fit[2] + margin)
             )
        ).nonzero()[0]
        right_lane_inds = (
            (nonzero_x > (
                self.prev_right_fit[0] * (nonzero_y ** 2) +
                self.prev_right_fit[1] * nonzero_y +
                self.prev_right_fit[2] - margin)
             ) &
            (nonzero_x < (
                self.prev_right_fit[0] * (nonzero_y ** 2) +
                self.prev_right_fit[1] * nonzero_y +
                self.prev_right_fit[2] + margin)
             )
        ).nonzero()[0]
        
        # Again, extract left and right line pixel positions
        left_x  = nonzero_x[left_lane_inds]
        left_y  = nonzero_y[left_lane_inds]
        right_x = nonzero_x[right_lane_inds]
        right_y = nonzero_y[right_lane_inds]
    
        return left_x, left_y, right_x, right_y

    # STEP 6: Calculate Vehicle Position and Curve Radius
    def measure_curvature_meters(
        self,
        binary_warped: np.ndarray,
        left_fit_x   : np.ndarray,
        right_fit_x  : np.ndarray,
        plot_y       : np.ndarray
    ):
        # Define conversions in x and y from pixels space to meters
        # ym_per_pix =  30 / 1080 *  (720 / 1080)  # meters per pixel in y dimension
        # xm_per_pix = 3.7 / 1920 * (1280 / 1920)  # meters per pixel in x dimension
        ym_per_pix =  30 / 1080 * (binary_warped.shape[0] / 1080)  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 1920 * (binary_warped.shape[1] / 1920)  # meters per pixel in x dimension
        
        left_fit_cr  = np.polyfit(plot_y * ym_per_pix, left_fit_x  * xm_per_pix, 2)
        right_fit_cr = np.polyfit(plot_y * ym_per_pix, right_fit_x * xm_per_pix, 2)
        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(plot_y)
    
        # Calculation of R_curve (radius of curvature)
        left_curve_rad  = (
            (1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5
        ) / np.absolute(2 * left_fit_cr[0])
        right_curve_rad = (
            (1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5
        ) / np.absolute(2 * right_fit_cr[0])
    
        return left_curve_rad, right_curve_rad

    def measure_position_meters(self, binary_warped: np.ndarray, left_fit, right_fit):
        # Define conversion in x from pixels space to meters
        xm_per_pix  = 3.7 / 1920 * (binary_warped.shape[1] / 1920)  # meters per pixel in x dimension
        # Choose the y value corresponding to the bottom of the image
        y_max       = binary_warped.shape[0]
        # Calculate left and right line positions at the bottom of the image
        left_x_pos  =  left_fit[0] * y_max ** 2 +  left_fit[1] * y_max + left_fit[2]
        right_x_pos = right_fit[0] * y_max ** 2 + right_fit[1] * y_max + right_fit[2]
        # Calculate the x position of the center of the lane
        center_lanes_x_pos = (left_x_pos + right_x_pos) // 2
        # Calculate the deviation between the center of the lane and the center
        # of the picture. The car is assumed to be placed in the center of the
        # picture. If the deviation is negative, the car is on the felt hand
        # side of the center of the lane
        veh_pos = ((binary_warped.shape[1] // 2) - center_lanes_x_pos) * xm_per_pix
        return veh_pos

    # STEP 7: Project Lane Delimitations Back on Image Plane and Add Text for
    # Lane Info
    def project_lane_info(
        self,
        image,
        binary_warped,
        plot_y,
        left_fit_x,
        right_fit_x,
        m_inv,
        left_curve_rad,
        right_curve_rad,
        veh_pos
    ):
        # Create an image to draw the lines on
        warp_zero   = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp  = np.dstack((warp_zero, warp_zero, warp_zero))
    
        # Center Line modified
        margin      = 400 * (binary_warped.shape[1] / 1920)
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left    = np.array([np.transpose(np.vstack([left_fit_x, plot_y]))])
        pts_right   = np.array([
            np.flipud(np.transpose(np.vstack([right_fit_x, plot_y])))
        ])
    
        pts_left_c  = np.array([
            np.transpose(np.vstack([left_fit_x + margin, plot_y]))
        ])
        pts_right_c = np.array([
            np.flipud(np.transpose(np.vstack([right_fit_x - margin, plot_y])))
        ])
        pts         = np.hstack((pts_left_c, pts_right_c))
    
        pts_left_i  = np.array([
            np.transpose(np.vstack([left_fit_x + margin + 150, plot_y]))
        ])
        pts_right_i = np.array([
            np.flipud(np.transpose(np.vstack([right_fit_x - margin - 150, plot_y])))
        ])
        pts_i       = np.hstack((pts_left_i, pts_right_i))
    
        # Draw the lane onto the warped blank image
        color_warp_img = cv2.polylines(color_warp, np.int_([pts_left]),  False, (0, 0, 255), 50)
        color_warp_img = cv2.polylines(color_warp, np.int_([pts_right]), False, (0, 0, 255), 50)
        color_warp_img = cv2.fillPoly( color_warp, np.int_([pts]), (0, 255, 0))
        # color_warp_img=cv2.fillPoly(color_warp, np.int_([pts_i]), (0,0, 255))
    
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        new_warp = cv2.warpPerspective(color_warp, m_inv, (image.shape[1], image.shape[0]))
    
        # Combine the result with the original image
        out_image = cv2.addWeighted(image, 0.7, new_warp, 0.3, 0)
    
        cv2.putText(
            img       = out_image,
            text      = "Curve Radius [m]: " + str((left_curve_rad + right_curve_rad) / 2)[:7],
            org       = (40, 70),
            fontFace  = cv2.FONT_HERSHEY_COMPLEX_SMALL,
            fontScale = 1.6,
            color     = (255, 255, 255),
            thickness = 2,
            lineType  = cv2.LINE_AA
        )
        cv2.putText(
            img       = out_image,
            text      = "Center Offset [m]: " + str(veh_pos)[:7],
            org       = (40, 150),
            fontFace  = cv2.FONT_HERSHEY_COMPLEX_SMALL,
            fontScale = 1.6,
            color     = (255, 255, 255),
            thickness = 2,
            lineType  = cv2.LINE_AA
        )
    
        return out_image, color_warp_img, new_warp
    
    # MARK: Visualize
    
    def draw_poly_lines(
	    self,
	    binary_warped: np.ndarray,
	    left_fit_x   : np.ndarray,
	    right_fit_x  : np.ndarray,
	    plot_y       : np.ndarray
    ):
        # Create an image to draw on and an image to show the selection window
        out_img    = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        window_img = np.zeros_like(out_img)
        margin     = 100
        
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1  = np.array([np.transpose(np.vstack([left_fit_x - margin, plot_y]))])
        left_line_window2  = np.array([np.flipud(np.transpose(np.vstack([left_fit_x + margin, plot_y])))])
        left_line_pts      = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fit_x - margin, plot_y]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fit_x + margin, plot_y])))])
        right_line_pts     = np.hstack((right_line_window1, right_line_window2))
    
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]),  (100, 100, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (100, 100, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        
        # Plot the polynomial lines onto the image
        plt.plot(left_fit_x,  plot_y, color="green")
        plt.plot(right_fit_x, plot_y, color="blue")
        
        # End visualization steps
        return result


# MARK: - Main

def run(opt):
    if not is_yaml_file(opt.config):
        opt.config = os.path.join(os.getcwd(), "data", opt.config)
    if is_yaml_file(opt.config):
        config = Munch(load(opt.config))
    else:
        raise RuntimeError(f"`config` cannot be initialized with: {opt.config}.")
    
    # NOTE: Video capture
    path = config.path
    if path == "" or path is None or \
        (isinstance(path, str) and not os.path.isdir(path)):
        path = os.getcwd()
    video_file = os.path.join(path, config.video_file)
    capture    = cv2.VideoCapture(video_file)
    if not capture.isOpened():
        error_console("File open failed!")
        capture.release()
        sys.exit()
    
    # NOTE: Video writer
    if opt.save_video:
        output = opt.output
        if output == "" or output is None or \
            (isinstance(output, str) and not is_video_file(output)):
            output = f"{video_file.split('.')[0]}_results.mp4"
        w      = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = capture.get(cv2.CAP_PROP_FPS)
        delay  = int(1000 / fps)
        angle  = 0
        writer = cv2.VideoWriter(
            output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )
    
    # NOTE: Lane detector
    lane_detector = AdvancedLaneDetector(config=config)
    
    # NOTE: Main loop
    if opt.verbose:
        cv2.namedWindow("frame",      cv2.WINDOW_NORMAL)
        cv2.namedWindow("color_warp", cv2.WINDOW_NORMAL)
        cv2.namedWindow("draw_poly",  cv2.WINDOW_NORMAL)
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        img_out, angle, color_warp, draw_poly_img = lane_detector.forward(x=frame)
        
        if opt.save_video:
            writer.write(img_out)
        if opt.verbose:
            cv2.imshow("frame",      img_out)
            cv2.imshow("color_warp", color_warp)
            cv2.imshow("draw_poly",  draw_poly_img)
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    capture.release()
    if opt.save_video:
        writer.release()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="./data/challenge.yml", type=str)
    parser.add_argument("--output",     default="",                     type=str)
    parser.add_argument("--save_video", default=True,                   type=bool)
    parser.add_argument("--verbose",    default=True,                   type=bool)
    opt = parser.parse_args()
    run(opt)
