#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unit testing for in-develop components."""

from __future__ import annotations

import cv2
import numpy as np

import mon

console = mon.console


def main():
    path       = mon.Path("./data/10.jpg")
    image      = cv2.imread(str(path))

    dcp1       = mon.get_dark_channel_prior(image, 15)
    dcp2       = mon.get_dark_channel_prior_02(image, 15)
    cv2.imshow("DCP 01", dcp1)
    cv2.imshow("DCP 02", dcp2)

    prior0     = mon.get_guided_brightness_enhancement_map_prior(image, 2, None)
    prior      = np.where(prior0 > 0.2, 255, 0).astype(np.uint8)
    dark       = cv2.bitwise_and(image, image, mask=prior)
    bright     = cv2.bitwise_and(image, image, mask=(255 - prior))
    contrast   = 3.5  # Contrast control (0-127)
    brightness = 5.0  # Brightness control (0-100)
    e_dark     = cv2.addWeighted(dark, contrast, dark, 0, brightness)
    enhance    = cv2.addWeighted(e_dark, 1, bright, 1, 0)

    mon.detect_blur_spot(image, verbose=True)
    mon.detect_bright_spot(image, verbose=True)

    contours, hierarchy = cv2.findContours(255 - prior, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        c      = max(contours, key=cv2.contourArea)
        mask   = np.full(image.shape, 255, image.dtype)
        cv2.drawContours(mask, c, -1, 255, 3)
        image  = cv2.bitwise_and(image, mask, mask=prior)
        # image  = np.where(image > 0, image, 255).astype(np.uint8)
    
    cv2.imwrite(f"data/{path.stem}.png",         image)
    cv2.imwrite(f"data/{path.stem}-prior0.png",  255 * prior0[..., ::-1])
    cv2.imwrite(f"data/{path.stem}-prior.png",   prior[..., ::-1])
    cv2.imwrite(f"data/{path.stem}-bright.png",  bright)
    cv2.imwrite(f"data/{path.stem}-dark.png",    dark)
    cv2.imwrite(f"data/{path.stem}-edark.png",   e_dark)
    cv2.imwrite(f"data/{path.stem}-enhance.png", enhance)
    
    cv2.imshow("Image",   image)
    cv2.imshow("Prior 0", prior0)
    cv2.imshow("Prior",   prior)
    cv2.imshow("Bright",  bright)
    cv2.imshow("Dark",    dark)
    cv2.imshow("EDark",   e_dark)
    cv2.imshow("Enhance", enhance)
    cv2.waitKey(0)
    

# region Main

if __name__ == "__main__":
    main()

# endregion
