#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import cv2
import mon
from mon.vision.enhance.llie.d2ce import DepthBoundaryAware

image  = mon.read_image("data/01.jpg", to_rgb=True, to_tensor=True, normalize=True)
dba    = DepthBoundaryAware()
db     = dba(image)
output = mon.to_image_nparray(db, keepdim=False, denormalize=True)
cv2.imshow("output", output)
cv2.waitKey(0)
