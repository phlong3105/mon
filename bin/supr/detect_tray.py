#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script implements tray detection algorithm."""

from __future__ import annotations

import click
import cv2
import numpy as np

import mon

_current_dir = mon.Path(__file__).absolute().parent


# region Functions

def process_image(image: np.ndarray):
    h, w, c     = image.shape
    gray        = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred     = cv2.GaussianBlur(gray, (5, 5), 0)
    scale       = 1
    delta       = 0
    grad_x      = cv2.Scharr(blurred, cv2.CV_16S, 1, 0, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y      = cv2.Scharr(blurred, cv2.CV_16S, 0, 1, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x  = cv2.convertScaleAbs(grad_x)
    abs_grad_y  = cv2.convertScaleAbs(grad_y)
    grad        = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    seed        = (int(w / 2), int(h / 2))
    ret, out_img, mask, rect = cv2.floodFill(grad, np.zeros((h + 2, w + 2), dtype=np.uint8), seed, 255, 5, 5)
    image       = cv2.cvtColor(out_img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 7)
    """
    edge        = cv2.Canny(blurred, 10, 250)
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 5:
            cv2.drawContours(image, [cnt], 0, 255, -1)
        elif len(approx) == 3:
            cv2.drawContours(image, [cnt], 0, (0, 255, 0), -1)
        elif len(approx) == 4:
            cv2.drawContours(image, [cnt], 0, (0, 0, 255), -1)
    """
    return image, rect
    

@click.command()
@click.option("--source",  default=mon.DATA_DIR/"aic23-checkout/testA/testA_3.mp4", type=click.Path(exists=True), help="Image/video directory.")
@click.option("--verbose", default=True, is_flag=True)
def detect_tray(
    source : mon.Path,
    verbose: bool
):
    """Convert bounding boxes"""
    assert source is not None and (mon.Path(source).is_dir() or mon.Path(source).is_video_file())
    
    source = mon.Path(source)
    
    if source.is_dir():
        image_files = list(source.rglob("*"))
        image_files = [f for f in image_files if f.is_image_file()]
        image_files = sorted(image_files)
        with mon.get_progress_bar() as pbar:
            for i in pbar.track(
                sequence    = range(len(image_files)),
                total       = len(image_files),
                description = f"[bright_yellow] Converting"
            ):
                image = cv2.imread(str(image_files[i]))
                image = process_image(image=image)
                if verbose:
                    cv2.imshow("Image", image)
                    cv2.waitKey(0)
                    
    elif source.is_video_file():
        cap = cv2.VideoCapture(str(source))
        while True:
            ret, image = cap.read()
            if not ret:
                break
            result, _  = process_image(image=image)
            if verbose:
                cv2.imshow("Image", result)
                if cv2.waitKey(1) == ord("q"):
                    break
        
# endregion


# region Main

if __name__ == "__main__":
    detect_tray()
    
# endregion
