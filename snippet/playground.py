#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""""

from __future__ import annotations

from mon import core

console = core.console


class A:
    zoo = {
        "B": 2,
    }
    
    def __init__(self):
        self.z = "A"
        
    @property
    def z(self):
        return self._z
    
    @z.setter
    def z(self, z):
        self._z = self.zoo[z]


class B(A):
    zoo = {
        "A": 10,
        "B": 20,
    }


b = B()
print(b.__class__.__name__)

# t = timeit.Timer("import mon.foundation")
# print(t.timeit(number = 1000000))
"""
console.log("Hi")
gb = foundation.MemoryUnit.GB
print(gb.value)

image = ci.read_image("lenna.png", to_tensor=True, normalize=True)
image = ci.adjust_brightness(image=image, brightness_factor=0.5)
image = ci.to_image(image, denormalize=True)
cv2.imshow("Lenna", image)
cv2.waitKey(0)
"""
