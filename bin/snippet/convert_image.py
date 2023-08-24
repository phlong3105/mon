#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script visualizes images."""

from __future__ import annotations

import click
import cv2

import mon
import mon.vision.core.image


# region Function

@click.command()
@click.option("--source",      default=mon.DATA_DIR, type=click.Path(exists=True), help="Image directory.")
@click.option("--destination", default=mon.DATA_DIR, type=click.Path(exists=False), help="Output video filepath or directory.")
@click.option("--start-frame", default=None, type=int, help="Start/end frame.")
@click.option("--end-frame",   default=None, type=int, help="Start/end frame.")
@click.option("--size",        default=None, type=int, nargs="+", help="Output images/video size.")
@click.option("--skip",        default=1, type=int, help="Skip n frames.")
@click.option("--save-video",  is_flag=True, help="Save images.")
@click.option("--extension",   default="png", type=click.Choice(["jpg", "png"], case_sensitive=False), help="Image extension.")
@click.option("--verbose",     is_flag=True)
def visualize_image(
    source     : mon.Path,
    destination: mon.Path,
    start_frame: int,
    end_frame  : int,
    size       : int | list[int],
    skip       : int,
    save_video : bool,
    extension  : str,
    verbose    : bool
):
    assert source is not None and mon.Path(source).is_dir()
    
    source = mon.Path(source)
    source = list(source.glob("*"))
    source = [s for s in source if s.is_dir()]
    
    if destination is not None:
        destination = mon.Path(destination)
        destination = [destination / f"{s.stem}" for s in source]
    else:
        destination = [s.parent / f"{s.stem}-convert" for s in source]
    if save_video:
        destination = [d.parent / f"{d.stem}.mp4" for d in destination]
    
    if size is not None:
        size = mon.vision.core.image.get_hw(size=size)

    skip = skip or 1
    
    for src, dst in zip(source, destination):
        image_files = list(src.rglob("*"))
        image_files = [f for f in image_files if f.is_image_file()]
        image_files = sorted(image_files)
        start_frame = start_frame or 0
        end_frame   = end_frame or len(image_files)
        if not end_frame >= start_frame:
            raise ValueError(
                f"end_frame must >= start_frame, but got {start_frame} and "
                f"{end_frame}."
            )

        dst.mkdir(parents=True, exist_ok=True)
        
        with mon.get_progress_bar() as pbar:
            for i in pbar.track(
                sequence    = range(len(image_files)),
                total       = len(image_files),
                description = f"[bright_yellow] Converting {src.name}"
            ):
                if not start_frame <= i <= end_frame or i % skip != 0:
                    continue
                
                image = cv2.imread(str(image_files[i]))
                
                if size is not None:
                    image = cv2.resize(image, size)
                
                if save_video:
                    pass
                else:
                    cv2.imwrite(dst/f"{i:06}.{extension}", image)
                    
                if verbose:
                    cv2.imshow("Image", image)
                    if cv2.waitKey(1) == ord("q"):
                        break
                
# endregion


# region Main

if __name__ == "__main__":
    visualize_image()

# endregion
