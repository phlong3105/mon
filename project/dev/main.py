#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script crops images."""

from __future__ import annotations

import cv2
import numpy as np

import mon

image         = cv2.imread("data/lenna.png")
image         = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_tensor  = mon.to_image_tensor(image, False, True)

# Box Filter
box_filter    = mon.box_filter(image_tensor, 1)
box_filter    = mon.to_image_nparray(box_filter, False, True)
box_filter    = cv2.cvtColor(box_filter, cv2.COLOR_BGR2RGB)
box_filter_cv = mon.box_filter(image, 1)
box_filter_cv = cv2.cvtColor(box_filter_cv, cv2.COLOR_BGR2RGB)
cv2.imshow("Box Filter",       box_filter)
cv2.imshow("Box Filter (cv2)", box_filter_cv)

# Guided Filter
guided_filter      = mon.guided_filter(image_tensor, image_tensor, 3, 1e-8)
guided_filter      = mon.to_image_nparray(guided_filter, False, True)
guided_filter      = cv2.cvtColor(guided_filter, cv2.COLOR_BGR2RGB)
fast_guided_filter = mon.FastGuidedFilter(3, 1e-8, 8)(image_tensor, image_tensor)
fast_guided_filter = mon.to_image_nparray(fast_guided_filter, False, True)
fast_guided_filter = cv2.cvtColor(fast_guided_filter, cv2.COLOR_BGR2RGB)
guided_filter_cv   = mon.guided_filter(image, image, 3, 1e-8)
guided_filter_cv   = np.uint8(guided_filter_cv)
guided_filter_cv   = cv2.cvtColor(guided_filter_cv, cv2.COLOR_BGR2RGB)
cv2.imshow("Guided Filter",       guided_filter)
cv2.imshow("Fast Guided Filter",  fast_guided_filter)
cv2.imshow("Guided Filter (cv2)", guided_filter_cv)

cv2.waitKey(0)
