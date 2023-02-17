#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script extract images from a video."""

from __future__ import annotations

import click
import cv2

import mon


# region Function

@click.command()
@click.option("--source",      default=mon.DATA_DIR/"aic23-checkout/run/", type=click.Path(exists=True), help="Video filepath or directory.")
@click.option("--destination", default=None, type=click.Path(exists=False), help="Output directory.")
@click.option("--from-index",  default=None, type=int, help="From/to frame index.")
@click.option("--to-index",    default=None, type=int, help="From/to frame index.")
@click.option("--size",        default=None, type=int, nargs="+", help="Output images' sizes.")
@click.option("--extension",   default="jpg", type=click.Choice(["jpg", "png"], case_sensitive=False), help="Image extension.")
@click.option("--verbose",     is_flag=True)
def extract_video_image(
    source     : mon.Path,
    destination: mon.Path,
    from_index : int,
    to_index   : int,
    size       : int | list[int],
    extension  : str,
    verbose    : bool
):
    assert source is not None and (mon.Path(source).is_video_file() or mon.Path(source).is_dir())

    source = mon.Path(source)
    source = [source] if mon.Path(source).is_video_file() else list(source.glob("*"))
    source = [s for s in source if s.is_video_file()]
    
    if destination is not None:
        destination = mon.Path(destination)
        destination = [destination / f"{s.stem}/images" for s in source]
    else:
        destination = [s.parent / f"{s.stem}/images" for s in source]
    
    if size is not None:
        size = mon.get_hw(size=size)
    
    for src, dst in zip(source, destination):
        dst.mkdir(parents=True, exist_ok=True)
        
        cap         = cv2.VideoCapture(str(src))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        from_index  = from_index or 0
        to_index    = to_index or frame_count
        if not to_index >= from_index:
            raise ValueError(
                f"to_index must >= to from_index, but got {from_index} and "
                f"{to_index}."
            )
        
        with mon.get_progress_bar() as pbar:
            for i in pbar.track(
                sequence    = range(frame_count),
                total       = frame_count,
                description = f"[bright_yellow] Extracting {src.name}"
            ):
                ret, image = cap.read()
                if not ret:
                    continue
                if i < from_index or i > to_index:
                    continue
                if size is not None:
                    image = cv2.resize(image, size)
                output_file = dst / f"{i:06}.{extension}"
                cv2.imwrite(str(output_file), image)
                if verbose:
                    cv2.imshow("Image", image)
                    cv2.waitKey(0)
            
# endregion


# region Main

if __name__ == "__main__":
    extract_video_image()

# endregion
