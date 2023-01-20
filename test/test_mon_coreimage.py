#!/usr/bin/env python
# -*- coding: utf-8 -*-

r""":mod:`test.test_mon_coreimage` perform unit testings for
:mod:`mon.coreimage` package.
"""

from __future__ import annotations

import unittest

import cv2

from mon import coreimage as ci

image = ci.read_image("lenna.png", to_tensor=True, normalize=True)


class TestIntensity(unittest.TestCase):
    
    def test_adjust_brightness(self):
        output = ci.adjust_brightness(image=image, brightness_factor=0.5)
        output = ci.to_image(output, denormalize=True)
        cv2.imshow("Adjust Brightness", output)
        cv2.waitKey(0)
        self.assertIsNotNone(output)
    

if __name__ == '__main__':
    unittest.main()
