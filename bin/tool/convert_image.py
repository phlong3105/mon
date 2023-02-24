#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script visualizes images."""

from __future__ import annotations

import click
import cv2

import mon


# region Function

@click.command()
@click.option("--source",      default=mon.DATA_DIR, type=click.Path(exists=True), help="Image directory.")
@click.option("--destination", default=mon.DATA_DIR, type=click.Path(exists=False), help="Output video filepath or directory.")
@click.option("--from-index",  default=None, type=int, help="From/to frame index.")
@click.option("--to-index",    default=None, type=int, help="From/to frame index.")
@click.option("--size",        default=None, type=int, nargs="+", help="Output images/video size.")
@click.option("--skip",        default=1, type=int, help="Skip n frames.")
@click.option("--save-video",  is_flag=True, help="Save images.")
@click.option("--extension",   default="png", type=click.Choice(["jpg", "png"], case_sensitive=False), help="Image extension.")
@click.option("--verbose",     is_flag=True)
def visualize_image(
    source     : mon.Path,
    destination: mon.Path,
    from_index : int,
    to_index   : int,
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
        size = mon.get_hw(size=size)

    skip = skip or 1
    
    for src, dst in zip(source, destination):
        image_files = list(src.rglob("*"))
        image_files = [f for f in image_files if f.is_image_file()]
        image_files = sorted(image_files)

        f_index     = from_index or 0
        t_index     = to_index or len(image_files)
        if not t_index >= f_index:
            raise ValueError(
                f"t_index must >= f_index, but got {f_index} and {t_index}."
            )

        dst.mkdir(parents=True, exist_ok=True)
        
        with mon.get_progress_bar() as pbar:
            for i in pbar.track(
                sequence    = range(len(image_files)),
                total       = len(image_files),
                description = f"[bright_yellow] Converting {src.name}"
            ):
                if not f_index <= i <= t_index or i % skip != 0:
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
