#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""""

from __future__ import annotations

import timeit

import cv2
import numpy as np

import mon

t = timeit.Timer("import mon")
print(t.timeit(number=1000000))
print(mon.ZOO_DIR)

image = cv2.imread("lenna.png")
bbox  = np.array([20, 20, 100, 100])
trajectory = [[20, 20], [30, 30], [140, 40], [50, 150]]
drawing = mon.draw_bbox(image, bbox, label="Text", color=[0, 255, 0], fill=True)
drawing = mon.draw_contour(drawing, trajectory, color=[0, 0, 255])
drawing = mon.draw_trajectory(drawing, trajectory, color=[0, 0, 255])
cv2.imshow("Image", drawing)
cv2.waitKey(0)
