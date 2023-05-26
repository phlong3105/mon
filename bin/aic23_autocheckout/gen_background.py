#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script performs background subtraction."""

from __future__ import annotations

import click
import cv2

import mon


# region Function

@click.command()
@click.option("--source",      default=mon.DATA_DIR/"aic23-autocheckout/testA/inpainting/", type=click.Path(exists=True), help="Video path or directory.")
@click.option("--destination", default=mon.DATA_DIR/"aic23-autocheckout/testA/background/", type=click.Path(exists=False), help="Output video filepath or directory.")
@click.option("--extension",   default="png", type=click.Choice(["jpg", "png"], case_sensitive=False), help="Image extension.")
@click.option("--verbose",     is_flag=True)
def convert_video(
    source     : mon.Path,
    destination: mon.Path,
    extension: str,
    verbose    : bool
):
    assert source is not None and (mon.Path(source).is_video_file() or mon.Path(source).is_dir())

    source = mon.Path(source)
    source = [source] if mon.Path(source).is_image_file() else list(source.glob("*"))
    source = [s for s in source if s.is_image_file()]
    source = sorted(source)

    if destination is not None:
        destination = mon.Path(destination)
        destination = [destination / f"{s.stem}.{extension}" for s in source]
    else:
        destination = [s.parent / f"{s.stem}-background.{extension}" for s in source]
    
    bg_sub = cv2.createBackgroundSubtractorMOG2()

    with mon.get_progress_bar() as pbar:
        for i in pbar.track(
            sequence    = range(len(source)),
            total       = len(source),
            description = f"[bright_yellow] Generating background"
        ):
            image = cv2.imread(str(source[i]))
            fg    = bg_sub.apply(image)
            bg    = bg_sub.getBackgroundImage()
            destination[i].parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(destination[i]), bg)
            if verbose:
                # cv2.imshow("Foreground", fg)
                cv2.imshow("Background", bg)
                if cv2.waitKey(1) == ord("q"):
                    break

# endregion


# region Main

if __name__ == "__main__":
    convert_video()

# endregion
