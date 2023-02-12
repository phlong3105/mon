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

image = cv2.imread("lenna.png")
bbox  = np.array([20, 20, 100, 100])
trajectory = [[20, 20], [30, 30], [140, 40], [50, 150], [360, 360]]
mon.draw_bbox(image, bbox, label="Text", color=[0, 255, 0], fill=True)
mon.draw_trajectory(image, trajectory, color=[0, 0, 255], radius=3, thickness=1, point=True)
cv2.imshow("Image", image)
cv2.waitKey(0)
