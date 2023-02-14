#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script extract images from a video."""

from __future__ import annotations

import click
import cv2

import mon


# region Function

@click.command()
@click.option("--source",     default=mon.DATA_DIR/"aic23-checkout/run/", type=click.Path(exists=True), help="Video path or directory.")
@click.option("--output-dir", default=None, type=click.Path(exists=False), help="Output directory.")
@click.option("--size",       default=None, type=int, nargs="+", help="Output images' sizes.")
@click.option("--extension",  default="jpg", type=click.Choice(["jpg", "png"], case_sensitive=False), help="Image extension.")
@click.option("--verbose",    is_flag=True)
def extract_video_image(
    source    : mon.Path,
    output_dir: mon.Path,
    size      : int | list[int],
    extension : str,
    verbose   : bool
):
    """Visualize bounding boxes on images."""
    assert source is not None and (mon.Path(source).is_video_file() or mon.Path(source).is_dir())

    source = mon.Path(source)
    source = [source] if mon.Path(source).is_video_file() else list(source.rglob("*"))
    source = [s for s in source if s.is_video_file()]
    
    if output_dir is not None:
        output_dir = mon.Path(output_dir)
        output_dir = [output_dir / f"{s.stem}/images" for s in source]
    else:
        output_dir = [s.parent / f"{s.stem}/images" for s in source]
    
    if size is not None:
        size = mon.get_hw(size=size)
    
    for src, dst in zip(source, output_dir):
        dst.mkdir(parents=True, exist_ok=True)
        
        cap    = cv2.VideoCapture(str(src))
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with mon.get_progress_bar() as pbar:
            for i in pbar.track(
                sequence    = range(length),
                total       = length,
                description = f"[bright_yellow] Extracting {src.name}"
            ):
                ret, image = cap.read()
                if not ret:
                    break
                if size is not None:
                    image = cv2.resize(image, size)
                
                output_file = dst / f"{i:06}.{extension}"
                cv2.imwrite(str(output_file), image)
                if verbose:
                    cv2.imshow("Image", image)
                    if cv2.waitKey(1) == ord("q"):
                        break
            
# endregion


# region Main

if __name__ == "__main__":
    extract_video_image()

# endregion
