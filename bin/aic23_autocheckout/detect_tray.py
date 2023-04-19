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
    blurred     = cv2.GaussianBlur(gray, (7, 7), 0)
    scale       = 1
    delta       = 0
    grad_x      = cv2.Scharr(blurred, cv2.CV_16S, 1, 0, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y      = cv2.Scharr(blurred, cv2.CV_16S, 0, 1, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x  = cv2.convertScaleAbs(grad_x)
    abs_grad_y  = cv2.convertScaleAbs(grad_y)
    grad        = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    cv2.imshow("Grad", grad)
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
@click.option("--source",      default=mon.DATA_DIR/"aic23-autocheckout/testA/background/", type=click.Path(exists=True), help="Video filepath or directory.")
@click.option("--destination", default=mon.DATA_DIR/"aic23-autocheckout/testA/tray/", type=click.Path(exists=False), help="Output video filepath or directory.")
@click.option("--extension",   default="png", type=click.Choice(["jpg", "png"], case_sensitive=False), help="Image extension.")
@click.option("--verbose",     default=True, is_flag=True)
def detect_tray(
    source     : mon.Path,
    destination: mon.Path,
    extension  : str,
    verbose    : bool
):
    assert source is not None and (mon.Path(source).is_dir() or mon.Path(source).is_video_file())
    
    source = mon.Path(source)
    source = [source] if mon.Path(source).is_image_file() else list(source.glob("*"))
    source = [s for s in source if s.is_image_file()]
    source = sorted(source)
    
    if destination is not None:
        destination = mon.Path(destination)
        destination = [destination / f"{s.stem}.{extension}" for s in source]
    else:
        destination = [s.parent / f"{s.stem}-tray.{extension}" for s in source]

    with mon.get_progress_bar() as pbar:
        for i in pbar.track(
            sequence    = range(len(source)),
            total       = len(source),
            description = f"[bright_yellow] Detecting tray"
        ):
            image = cv2.imread(str(source[i]))
            image, ret = process_image(image=image)
            destination[i].parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(destination[i]), image)
            if verbose:
                cv2.imshow("Tray", image)
                if cv2.waitKey(1) == ord("q"):
                    break

# endregion


# region Main

if __name__ == "__main__":
    detect_tray()
    
# endregion
