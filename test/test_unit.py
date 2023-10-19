#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unit testing for in-develop components."""

from __future__ import annotations

import time
import timeit

import cv2
import numpy as np

import mon

console = mon.console

print(timeit.repeat(stmt="import mon"))


# region TestCase

def unit_test():
    x = np.array([[1, 2, 3],
                  [4, 5, 6]])
    print(x, x.shape)
    print(x[:, -1])
    
    # image      = cv2.imread("./data/01.jpg")
    # guide      = cv2.imread("./data/01-guide.jpg")  # [..., ::-1]
    start_time = time.time()
    # dcp        = mon.get_dark_channel_prior(image, 15)
    # bcp        = mon.get_bright_channel_prior(image, 15)
    # bcp        = cv2.merge((bcp, bcp, bcp))
    # filter     = mon.guided_filter(image, guide, 2)
    run_time   = (time.time() - start_time)
    
    console.log(f"Average time: {run_time}")
    # cv2.imshow("Image", image)
    # cv2.imshow("Guide", guide)
    # cv2.imshow("DCP", dcp)
    # cv2.imshow("BCP", bcp)
    # cv2.imshow("Guided Filter", filter)
    # cv2.waitKey(0)
    
# endregion


# region Main

if __name__ == "__main__":
    unit_test()

# endregion
