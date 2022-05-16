#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Advanced Lane Detection: Computer Vision algorithm to compute road curvature
and lane vehicle offset using OpenCV Image Processing, Camera Calibration,
Perspective Transform, Color Masks, Sobel and Polynomial Fit.

References:
    https://github.com/OanaGaskey/Advanced-Lane-Detection
"""

from __future__ import annotations

from typing import Any
from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import os
import sys

from numpy import ndarray

from one import error_console
from one import is_image_file

__all__ = [
    "AdvancedLaneDetector",
]


# MARK: - Module

class AdvancedLaneDetector:
    """Advanced Lane Detection: Computer Vision algorithm to compute road
    curvature and lane vehicle offset using OpenCV Image Processing, Camera
    Calibration, Perspective Transform, Color Masks, Sobel and Polynomial Fit.
    
    Attributes:
    
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        camera_calib_dir: str = os.path.join(os.getcwd(), "data", "camera_calib"),
    ):
        super().__init__()
        self.camera_calib_dir = camera_calib_dir
        self.matrix           = None
        self.distortion       = None
        self.init_intrinsic_params()

    # MARK: Setup
    
    def init_intrinsic_params(self):
        # Prepare object points. From the provided calibration images, 9*6
        # corners are identified
        nx        = 9
        ny        = 6
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
        for image_file in glob.glob(os.path.join(self.camera_calib_dir, "*")):
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
        self.matrix     = mtx
        self.distortion = dist
    
    # MARK: Forward Pass
    
    def process(self, image: np.ndarray):
        
        pass
    
    def warp_image(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        undistorted = cv2.undistort(image, self.matrix, self.distortion, None, self.matrix)
        img_size    = (image.shape[1], image.shape[0])
        offset      = 300
        # Source points taken from images with straight lane lines, these are
        # to become parallel after the warp transform
        src = np.float32(
            [
                (190, 720),  # bottom-left corner
                (596, 447),  # top-left corner
                (685, 447),  # top-right corner
                (1125, 720)  # bottom-right corner
            ]
        )
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
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        H   = hls[:, :, 0]
        S   = hls[:, :, 2]
        sat_binary = np.zeros_like(S)
        # Detect pixels that have a high saturation value
        sat_binary[(S > 90) & (S <= 255)] = 1
        
        hue_binary = np.zeros_like(H)
        # Detect pixels that are yellow using the hue component
        hue_binary[(H > 10) & (H <= 25)] = 1
        
        # Combine all pixels detected above
        binary_1 = cv2.bitwise_or(sx_binary, white_binary)
        binary_2 = cv2.bitwise_or(hue_binary, sat_binary)
        binary   = cv2.bitwise_or(binary_1, binary_2)

        return binary
    
    def find_lane_pixels_using_histogram(
        self, binary_warped: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
                (nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                (nonzero_x >= win_x_left_low) & (nonzero_x < win_x_left_high)
            ).nonzero()[0]
            good_right_inds = (
                (nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
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
    ) -> tuple[
        tuple[ndarray, Any, ndarray],
        tuple[ndarray, Any, ndarray],
        int | Any,
        int | Any,
        ndarray | tuple[ndarray, float | None]
    ]:
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
