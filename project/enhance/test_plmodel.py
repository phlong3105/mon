#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script crops images."""

from __future__ import annotations

import cv2
import numpy as np


def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table    = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)


# Estimate surface normals from gradients
def estimate_normals(image):
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
	sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
	normal_map = np.dstack((sobel_x, sobel_y, np.ones_like(sobel_x)))
	norm = np.linalg.norm(normal_map, axis=-1, keepdims=True)
	normal_map = normal_map / (norm + 1e-6)
	return normal_map


# Compute ambient, diffuse, and specular reflections
def compute_phong_shading(image, normal_map):
	# Ambient reflection
	k_a = 0.1  # Ambient reflection coefficient
	I_a = 0.2  # Ambient light intensity
	I_ambient = k_a * I_a * np.ones_like(image)
	
	# Light direction (assume from above)
	light_dir = np.array([0, 1, 0])  # Light direction: top
	light_dir = light_dir / np.linalg.norm(light_dir)
	
	# Diffuse reflection
	dot_ln = np.einsum('ijk,k->ij', normal_map, light_dir)
	dot_ln = np.clip(dot_ln, 0, 1)
	k_d = 0.7  # Diffuse reflection coefficient
	I_L = 1.0  # Light intensity
	I_diffuse = k_d * dot_ln[..., np.newaxis] * image
	
	# Specular reflection
	view_dir = np.array([0, 0, 1])  # View direction (assumed along z-axis)
	
	# Compute reflection vector
	reflection_vector = 2 * np.einsum('ijk,k->ij', normal_map, light_dir)[..., np.newaxis] * normal_map - light_dir
	reflection_vector = reflection_vector / np.linalg.norm(reflection_vector, axis=-1, keepdims=True)
	
	dot_rv = np.einsum('ijk,k->ij', reflection_vector, view_dir)
	dot_rv = np.clip(dot_rv, 0, 1)
	n   = 10  # Shininess coefficient
	k_s = 0.5  # Specular reflection coefficient
	I_specular = k_s * np.power(dot_rv, n)[..., np.newaxis] * image
	
	# Combine reflections
	final_image = I_ambient + I_diffuse + I_specular
	final_image = np.clip(final_image, 0, 1) * 255.0
	final_image = final_image.astype(np.uint8)
	
	return final_image


image = cv2.imread("data/01.jpg")
# image = cv2.GaussianBlur(image, (3, 3), 0)

#
yuv            = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
y, u, v        = cv2.split(yuv)
y_enhanced     = adjust_gamma(y, gamma=2.0)
yuv_enhanced   = cv2.merge([y_enhanced, u, v])
blur           = cv2.GaussianBlur(y_enhanced, (0, 0), sigmaX=3, sigmaY=3)
detail         = cv2.addWeighted(y_enhanced,  1.5, blur, -0.5, 0)
specular       = cv2.threshold(detail, 220, 255, cv2.THRESH_BINARY)[1]
specular       = cv2.dilate(specular, None, iterations=1)
enhanced       = cv2.add(detail, specular)
enhanced_image = cv2.cvtColor(yuv_enhanced, cv2.COLOR_YUV2BGR)

# Show
cv2.imshow("Y"             , y)
cv2.imshow("Y Enhanced"    , y_enhanced)
cv2.imshow("U"             , u)
cv2.imshow("V"             , v)
cv2.imshow("Image"         , image)
cv2.imshow("Enhanced Image", enhanced_image)
cv2.waitKey(0)
