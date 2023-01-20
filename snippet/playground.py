#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""""

from __future__ import annotations

import cv2

from mon import coreimage as ci, foundation

console = foundation.console

# t = timeit.Timer("import mon.foundation")
# print(t.timeit(number = 1000000))

console.log("Hi")
gb = foundation.MemoryUnit.GB
print(gb.value)

image = ci.read_image("lenna.png", to_tensor=True, normalize=True)
image = ci.adjust_brightness(image=image, brightness_factor=0.5)
image = ci.to_image(image, denormalize=True)
cv2.imshow("Lenna", image)
cv2.waitKey(0)
