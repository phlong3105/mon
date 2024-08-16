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

datamodule = {
	"name"      : "lol_v1",
	"root"      : mon.DATA_DIR / "llie",  # A root directory where the data is stored.
	"transform" : A.Compose(transforms=[
		A.ResizeMultipleOf(
			height            = 504,
			width             = 504,
			keep_aspect_ratio = False,
			multiple_of       = 14,
			resize_method     = "lower_bound",
			interpolation     = cv2.INTER_AREA,
		),
		# A.Flip(),
		# A.Rotate(),
	]),  # Transformations performing on both the input and target.
	"to_tensor" : True,          # If ``True``, convert input and target to :class:`torch.Tensor`.
	"cache_data": False,         # If ``True``, cache data to disk for faster loading next time.
	"batch_size": 4,             # The number of samples in one forward pass.
	"devices"   : 0,             # A list of devices to use. Default: ``0``.
	"shuffle"   : True,          # If ``True``, reshuffle the datapoints at the beginning of every epoch.
	"verbose"   : True,          # Verbosity.
}
datamodule: mon.DataModule = mon.DATAMODULES.build(config=datamodule)
datamodule.prepare_data()
datamodule.setup(stage="train")
input, target, extra = next(iter(datamodule.val_dataloader))

# image = cv2.imread("data/01.jpg")
# image = mon.read_image(path="data/01.jpg", to_rgb=True, to_tensor=False, normalize=False)
# h, w  = image.shape[:2]
# image = transform(image=image)["image"]
# image = mon.to_image_tensor(image, keepdim=False, normalize=True)
# image = image.to(torch.get_default_dtype())

model = D2CE()
# c_1, c_2, d, e, gf, o = model(input)
results = model.forward_loss(input, None)

d = results["depth"]
d = d[1]
d = mon.to_image_nparray(d, False, True)
cv2.imshow("d", d)

e = results["edge"]
e = e[1]
e = mon.to_image_nparray(e, False, True)
cv2.imshow("e", e)
cv2.waitKey(0)
