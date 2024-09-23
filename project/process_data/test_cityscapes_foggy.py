#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2

import mon

dataset     = mon.CityscapesFoggy(split=mon.Split.VAL, verbose=True)
iterator    = iter(dataset)
datapoint   = next(iterator)
classlabels = dataset.classlabels
image       = datapoint["image"]
hq_image    = datapoint["hq_image"]
image       = cv2.cvtColor(image,    cv2.COLOR_RGB2BGR)
hq_image    = cv2.cvtColor(hq_image, cv2.COLOR_RGB2BGR)
cv2.imshow("Image", image)
cv2.imshow("HQ Image", hq_image)
cv2.waitKey(0)
