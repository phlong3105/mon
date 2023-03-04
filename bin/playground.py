#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A playground."""

from __future__ import annotations

import timeit

import cv2
import numpy as np

import mon

t = timeit.Timer("import mon")
print(t.timeit(number=1000000))

image       = cv2.imread("lenna.png")
gray        = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Define the HOG descriptor parameters
winSize     = (64, 128)
blockSize   = (16, 16)
blockStride = (8, 8)
cellSize    = (8, 8)
nbins       = 9
# Create a HOG descriptor object
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
# Compute the HOG features of the image
features = hog.compute(gray)

# Print the size of the feature vector
print("Feature vector size:", features.shape, type(features))
print(features)

cv2.imshow("Image", image)
cv2.waitKey(0)
