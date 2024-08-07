#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import cv2
import torch

import mon
from mon import albumentation as A
from mon.vision.enhance.llie.d2ce import D2CE

transform = A.Compose([
	A.ResizeMultipleOf(
		height            = 504,
		width             = 504,
		keep_aspect_ratio = False,
		multiple_of       = 14,
		resize_method     = "lower_bound",
		interpolation     = cv2.INTER_AREA,
	),
	# A.NormalizeImageMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image = cv2.imread("data/01.jpg")
h, w  = image.shape[:2]
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # / 255.0
image = transform(image=image)["image"]
image = mon.to_image_tensor(image, keepdim=False, normalize=True)
image = image.to(torch.get_default_dtype())

c_1, c_2, d, e, gf, o = D2CE()(image)
d = d.squeeze(1)
d = mon.to_image_nparray(d, False, True)
print(d.shape, d.dtype)
cv2.imshow("d", d)

e = mon.to_image_nparray(e, False, True)
cv2.imshow("e", e)
cv2.waitKey(0)
