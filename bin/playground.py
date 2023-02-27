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

image = cv2.imread("lenna.png")
orb   = cv2.ORB_create(100, 1.2)
keypoints, descriptors = orb.detectAndCompute(image, None)
for kp, des in zip(keypoints, descriptors):
    print(kp.pt, des)

image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)
cv2.imshow("Image", image)
cv2.waitKey(0)
